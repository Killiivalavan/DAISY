import asyncio
import shlex
import time
from datetime import datetime

import psutil


async def get_time_date() -> str:
    now = datetime.now()
    tz = time.tzname
    return f"It is {now.strftime('%A, %B %d, %Y at %I:%M %p')}. Timezone: {tz[0]}"


async def get_system_info(config) -> str:
    cpu = await asyncio.to_thread(psutil.cpu_percent, interval=0.5)
    mem = await asyncio.to_thread(psutil.virtual_memory)
    disk = await asyncio.to_thread(psutil.disk_usage, "/")
    return (
        f"CPU: {cpu}% | RAM: {mem.percent}% "
        f"({mem.used // 1_000_000_000}GB / {mem.total // 1_000_000_000}GB) "
        f"| Disk: {disk.percent}% "
        f"({disk.used // 1_000_000_000}GB / {disk.total // 1_000_000_000}GB)"
    )


def _command_allowed(command: str, allowed_commands: list[str]) -> bool:
    try:
        cmd_name = shlex.split(command)[0]
    except (ValueError, IndexError):
        return False
    for allowed in allowed_commands:
        if cmd_name == allowed:
            return True
    return False


async def run_command(config, command: str) -> str:
    allowed = config.tools.allowed_commands
    if not _command_allowed(command, allowed):
        return (
            f"Command not allowed. "
            f"Allowed commands: {', '.join(allowed)}"
        )

    proc = await asyncio.create_subprocess_exec(
        *shlex.split(command),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=config.tools.default_timeout,
        )
    except asyncio.TimeoutError:
        proc.terminate()
        return "Error: Command timed out."

    output = (stdout or b"").decode(errors="replace").strip()
    error = (stderr or b"").decode(errors="replace").strip()
    result = output or error or "(no output)"
    return result[:8000]


async def set_reminder(config, announcement_queue, duration_seconds: int, message: str) -> str:
    if duration_seconds < 1:
        return "Error: Duration must be at least 1 second."
    if duration_seconds > 86400:
        return "Error: Duration cannot exceed 24 hours (86400 seconds)."

    async def _reminder_worker():
        await asyncio.sleep(duration_seconds)
        await announcement_queue.push({
            "summary": f"Reminder: {message}",
            "type": "reminder",
            "priority": 1,
        })

    asyncio.create_task(_reminder_worker())
    return f"Reminder set for {duration_seconds} seconds from now"
