"""HF 引用（repo id 或本地路径）的 ~ 展开。"""

from pathlib import Path


def expand_user_ref(ref: str) -> str:
    """Expand a leading ``~``; return every other input byte-identical.

    ``str(Path(x).expanduser())`` 在 Windows 把 ``/`` 换成 ``\\``（毁掉
    ``Qwen/Qwen2.5`` 这类 HF repo id），并折叠 ``a/./b`` / ``a//b`` / 尾斜杠——
    这些归一化对可能是 HF 引用的值一概不要。~ 展开保留（HF 拒收 ``~/...``
    形态）。纯 stdlib，无 src 内依赖。
    """
    if not ref.startswith("~"):
        return ref
    return str(Path(ref).expanduser())
