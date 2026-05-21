import sys
import io
import traceback
import threading
from contextlib import redirect_stdout, redirect_stderr


_SANDBOX_GLOBALS = {
    "__builtins__": __builtins__,
    # Allow common imports to be used in exec'd code
}


def run_python(code: str, timeout: int = 30) -> dict:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    result = {"stdout": "", "stderr": "", "error": None, "return_value": None}
    namespace = dict(_SANDBOX_GLOBALS)

    def _run():
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<jarvis>", "exec"), namespace)
        except Exception:
            stderr_buf.write(traceback.format_exc())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        result["error"] = f"Timeout after {timeout}s"
    else:
        result["stdout"] = stdout_buf.getvalue()
        result["stderr"] = stderr_buf.getvalue()

    return result
