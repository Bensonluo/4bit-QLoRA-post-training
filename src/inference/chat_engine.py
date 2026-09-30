"""加载模型并对话：底座 + LoRA adapter（可选）→ 合并 → chat template 生成。

对齐 LLaMA-Factory WebUI Chat 的默认形态（HF 引擎，
``model_name_or_path`` + ``adapter_name_or_path``）：底座走本仓库平台感知的
``load_model_and_tokenizer``，adapter 用 peft 包装后 ``merge_and_unload``
（等效「合并导出后部署」，生成更快）。

生成走 **流式**（HF ``TextIteratorStreamer`` + 后台线程，Streamlit
``st.write_stream`` 的标准接法，见官方 docs 与 streamers.py 源码）：
``stream_reply`` 逐块 yield 新增文本，``skip_prompt=True`` 由 streamer
负责只给生成段——chat template 时代不再需要字符串前缀剥离。

重依赖（torch/transformers/peft）一律在函数内延迟导入：模块本身可被
无 GPU 环境与单测安全 import（与 ``src/data/preflight.py`` 同一守卫约定）。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def load_chat_model(base_model: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    """加载底座（不量化，方便合并）+ 可选 adapter，返回 (model, tokenizer）。"""
    from pathlib import Path

    from peft import PeftModel

    from config.base import ModelConfig
    from src.models.loader import load_model_and_tokenizer
    from src.utils.hf_refs import expand_user_ref

    adapter_ref: str | None = None
    if adapter_path:
        # adapter 路径 ~ 展开（R115 obs-2 → R116 收口；先例
        # domains/medical_entity/eval/models.py RealFinetunedModel._load）：
        # 手输 ~/... 不展开会被 PeftModel 当 repo id 拒收；HF 名逐字节透传无损。
        adapter_ref = expand_user_ref(adapter_path)
        # 绝对路径 = 本机目录引用 → 预检 fail-fast（R117）：底座下载/加载是
        # 分钟级等待，坏 adapter 路径必须在它之前报错（st.cache_resource 不
        # 缓存失败，重试每次重付等待）。仅用 Path 做谓词（is_absolute/is_dir/
        # exists），不做字符串变异——R115 教训：str(Path(x)) 会毁 HF repo id。
        # 非绝对路径（HF repo id/相对路径）透传不检查，迟到报错维持。
        local_ref = Path(adapter_ref)
        if local_ref.is_absolute():
            if not local_ref.is_dir():
                # 「或不是目录」：用户可能粘贴的是文件(如 adapter_model.safetensors)
                # 而非 adapter 目录——r117-reviewer nit-2 文案精确性采纳
                raise FileNotFoundError(
                    f"LoRA adapter 路径不存在或不是目录：{adapter_path}"
                    " —— 请检查手输路径，或回到上方列表选择本机训练产物。"
                )
            if not (local_ref / "adapter_config.json").exists():
                raise ValueError(
                    f"该目录不是 LoRA adapter（缺 adapter_config.json）：{adapter_path}"
                    " —— 已合并的完整模型请改填「底座模型」框，不填 adapter。"
                )

    # ~ 展开成 home；HF 名逐字节透传（不过 Path()——Windows 下 / 会被换成 \
    # 毁 repo id，R115 expand_user_ref）
    base_ref = expand_user_ref(base_model)
    config = ModelConfig(name=base_ref, quantization_bits=None)
    model, tokenizer = load_model_and_tokenizer(config)

    if adapter_ref:
        peft_model: Any = PeftModel.from_pretrained(model, adapter_ref)
        model = peft_model.merge_and_unload()

    model.eval()
    return model, tokenizer


def stream_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
    timeout: float = 300.0,
) -> Iterator[str]:
    """流式一轮对话生成：逐块 yield 新增文本，结束即返回。

    ``enable_thinking`` 透传给 chat template（Qwen3 系列支持；其他模板
    的 jinja 会忽略未使用变量，安全）。``timeout`` 是 streamer 队列的取数
    超时——生成线程若异常死亡，队列永远收不到停止信号，靠它打破僵局
    （transformers streamers.py 文档明示该用途）。
    """
    import queue
    import threading

    from transformers import TextIteratorStreamer

    # 新版 transformers 默认 return_dict=True：返回 BatchEncoding（dict-like），
    # 直接 .shape 会走 __getattr__ 抛空消息 AttributeError——必须按键取值
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=timeout
    )
    thread_error: list[BaseException] = []

    def _worker() -> None:
        import torch

        try:
            with torch.no_grad():
                model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except BaseException as exc:  # 线程内异常不会自己传到主线程——记录供诊断
            thread_error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield from streamer
    except queue.Empty:
        reason = (
            repr(thread_error[0]) if thread_error else f"Generation timed out after {timeout:.0f}s"
        )
        raise RuntimeError(f"生成失败：{reason}") from None
    finally:
        thread.join(timeout=1.0)
    if thread_error:
        # 迭代结束但线程标记了异常（如 generate 尾部抛错）——如实上报
        raise RuntimeError(f"生成失败：{thread_error[0]!r}")


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
) -> str:
    """非流式便捷封装：拼接 ``stream_reply`` 的全部文本块。"""
    return "".join(
        stream_reply(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
    )
