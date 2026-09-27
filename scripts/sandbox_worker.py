"""JSON transform worker. Run only inside the OS/container boundary in sandbox.py."""

import contextlib
import json
import math
import os
import re
import resource
import sys


def main():
    request_path, source_path, output_limit, cpu_limit, memory_mb = sys.argv[1:]
    resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_limit), int(cpu_limit)))
    resource.setrlimit(resource.RLIMIT_FSIZE, (int(output_limit), int(output_limit)))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    memory = int(memory_mb) * 1024 * 1024
    # macOS does not reliably support lowering DATA/AS. The parent monitors RSS;
    # Linux additionally has the container's hard memory boundary.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))
    try:
        with open(request_path, encoding="utf-8") as handle:
            request = json.load(handle)
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
        allowed_modules = {"json": json, "re": re, "math": math}

        def allowed_import(name, *args, **kwargs):
            if name not in allowed_modules:
                raise ImportError("Only re, json and math may be imported by transforms.")
            return allowed_modules[name]

        import builtins

        namespace = {"__builtins__": {**vars(builtins), "__import__": allowed_import}}
        with open(os.devnull, "w") as silent, contextlib.redirect_stdout(silent):
            exec(compile(source, "transform.py", "exec"), namespace)
            rows = namespace["transform"](request["rows"], request["config"])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("transform must return a list of JSON objects.")
        payload = json.dumps({"ok": True, "rows": rows}, ensure_ascii=False, allow_nan=False)
        if len(payload.encode()) > int(output_limit):
            raise ValueError("Transform output exceeds the configured size limit.")
    except BaseException as exc:
        payload = json.dumps({"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:500]})
    sys.stdout.write(payload)


if __name__ == "__main__":
    main()
