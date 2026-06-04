import asyncio
from pathlib import Path


def _validate_path(path: str, allowed_directories: list[str]) -> Path:
    if not path.startswith("/"):
        raise PermissionError(
            f"Path must be absolute (start with /). Got: {path}"
        )
    resolved = Path(path).resolve()
    for allowed in allowed_directories:
        allowed_path = Path(allowed).resolve()
        if str(resolved).startswith(str(allowed_path)):
            return resolved
    raise PermissionError(
        f"Path not in allowed directories. "
        f"Allowed: {', '.join(allowed_directories)}"
    )


async def read_file(config, path: str) -> str:
    try:
        resolved = _validate_path(path, config.tools.allowed_directories)
    except PermissionError as e:
        return str(e)

    if not resolved.exists():
        return f"File not found: {path}"
    if not resolved.is_file():
        return f"Not a file: {path}"

    file_size = resolved.stat().st_size
    if file_size > config.tools.file_max_size_bytes:
        return (
            f"File too large ({file_size // 1024}KB). "
            f"Maximum allowed: {config.tools.file_max_size_bytes // 1024}KB"
        )

    content = await asyncio.to_thread(resolved.read_text, encoding="utf-8")
    if not content.strip():
        return "(empty file)"

    return content[:8000]


async def write_file(config, path: str, content: str) -> str:
    try:
        resolved = _validate_path(path, config.tools.allowed_directories)
    except PermissionError as e:
        return str(e)

    await asyncio.to_thread(resolved.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(resolved.write_text, content, encoding="utf-8")
    size = len(content)
    return f"Written {size} bytes to {path}"
