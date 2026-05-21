import os
from pathlib import Path


def read_file(path: str, max_chars: int = 20000) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: file not found — {path}"
    if p.is_dir():
        return f"Error: {path} is a directory. Use list_directory instead."
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[...truncated at {max_chars} chars. File is {len(text)} chars total.]"
    return text


def write_file(path: str, content: str, append: bool = False) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    try:
        p.write_text(content, encoding="utf-8") if not append else open(p, "a").write(content)
        action = "appended to" if append else "written to"
        return f"Successfully {action} {p} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"


def list_directory(path: str = ".", show_hidden: bool = False) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"Path not found: {path}"}
    if not p.is_dir():
        return {"error": f"Not a directory: {path}"}

    entries = {"path": str(p.resolve()), "dirs": [], "files": []}
    try:
        for item in sorted(p.iterdir()):
            if not show_hidden and item.name.startswith("."):
                continue
            if item.is_dir():
                entries["dirs"].append(item.name + "/")
            else:
                size = item.stat().st_size
                entries["files"].append({"name": item.name, "size_bytes": size})
    except PermissionError:
        entries["error"] = "Permission denied"
    return entries


def get_file_info(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"Not found: {path}"}
    stat = p.stat()
    return {
        "path": str(p.resolve()),
        "type": "directory" if p.is_dir() else "file",
        "size_bytes": stat.st_size,
        "extension": p.suffix,
    }
