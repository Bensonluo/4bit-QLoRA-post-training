"""加载模型并对话：底座 + LoRA adapter（可选）→ 合并 → chat template 生成。

对齐 LLaMA-Factory WebUI Chat 的默认形态（HF 引擎，
``model_name_or_path`` + ``adapter_name_or_path``）：底座走本仓库平台感知的
``load_model_and_tokenizer``，adapter 用 peft 包装后 ``merge_and_unload``
（等效「合并导出后部署」，生成更快）。生成只解码新增 token——chat
template 时代不再需要字符串前缀剥离。

重依赖（torch/transformers/peft）一律在函数内延迟导入：模块本身可被
无 GPU 环境与单测安全 import（与 ``src/data/preflight.py`` 同一守卫约定）。
"""

from __future__ import annotations

from typing import Any


def load_chat_model(base_model: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    """加载底座（不量化，方便合并）+ 可选 adapter，返回 (model, tokenizer)。"""
    from pathlib import Path

    from peft import PeftModel

    from config.base import ModelConfig
    from src.models.loader import load_model_and_tokenizer

    # 本地路径里的 ~ 展开成 home；HF 名不含 ~，展开后原样返回
    base_ref = str(Path(base_model).expanduser())
    config = ModelConfig(name=base_ref, quantization_bits=None)
    model, tokenizer = load_model_and_tokenizer(config)

    if adapter_path:
        peft_model: Any = PeftModel.from_pretrained(model, adapter_path)
        model = peft_model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
) -> str:
    """一轮对话生成。messages 为 chat 格式（role/content），返回新增文本。

    ``enable_thinking`` 透传给 chat template（Qwen3 系列支持；其他模板
    的 jinja 会忽略未使用变量，安全）。
    """
    import torch

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
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只解码生成段：prompt_len 之后的 token 即模型新写的内容
    new_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return new_text if isinstance(new_text, str) else str(new_text)
