import asyncio
import json
import time
from pathlib import Path


async def _run_shell_task(config, command: str) -> dict:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=config.tools.max_timeout,
        )
    except asyncio.TimeoutError:
        proc.terminate()
        return {"error": "Command timed out"}

    return {
        "stdout": (stdout or b"").decode(errors="replace")[:3000],
        "stderr": (stderr or b"").decode(errors="replace")[:1000],
        "returncode": proc.returncode,
    }


async def _run_sub_agent(config, llm, system_prompt: str, prompt: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    result_parts = []
    async for token in llm.stream_tokens(messages):
        result_parts.append(token)
    full_result = "".join(result_parts)
    return {"result": full_result}


async def _run_opencode(config, cmd: list) -> dict:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=config.tools.max_timeout,
        )
    except asyncio.TimeoutError:
        proc.terminate()
        return {"error": "OpenCode task timed out"}

    if proc.returncode != 0:
        err = (stderr or b"").decode(errors="replace")[:2000]
        return {"error": f"OpenCode exited with code {proc.returncode}: {err}"}

    output = (stdout or b"").decode(errors="replace").strip()
    if output.startswith("{"):
        try:
            parsed = json.loads(output)
            return {"result": parsed}
        except json.JSONDecodeError:
            pass

    return {"result": output[:5000]}


async def spawn_task(
    config, task_tracker, llm, task_type: str, description: str, payload: dict,
    notify_on_complete: bool = False,
) -> str:
    if task_type == "shell":
        command = payload.get("command")
        if not command:
            return "Error: 'command' required in payload for shell task type."
        coro = _run_shell_task(config, command)

    elif task_type == "sub_agent":
        prompt = payload.get("prompt")
        if not prompt:
            return "Error: 'prompt' required in payload for sub_agent task type."
        system_prompt = payload.get("system_prompt",
            "You are a helpful background worker AI. Complete the task thoroughly.")
        coro = _run_sub_agent(config, llm, system_prompt, prompt)

    else:
        return f"Error: Unknown task type '{task_type}'. Supported: shell, sub_agent."

    task_id = await task_tracker.create_task(
        description=description,
        coro=coro,
        notify_on_complete=notify_on_complete,
    )
    return f"Background task started. ID: {task_id}. Description: {description}"


async def spawn_opencode_task(
    config, task_tracker, prompt: str, project_dir: str = None,
    notify_on_complete: bool = False,
) -> str:
    import shutil

    if not shutil.which("opencode"):
        return "Error: opencode is not installed on this system."

    project_root = config.tools.opencode.project_root
    if project_dir:
        resolved = str(Path(project_dir).resolve())
        if not resolved.startswith(project_root):
            return f"Error: project_dir must be within {project_root}"
    else:
        project_dir = project_root

    cmd = ["opencode", "-p", prompt, "-f", "json", "-q", "-c", project_dir]

    description = f"OpenCode: {prompt[:80]}"
    task_id = await task_tracker.create_task(
        description=description,
        coro=_run_opencode(config, cmd),
        notify_on_complete=notify_on_complete,
    )
    return f"OpenCode task started. ID: {task_id}. Description: {description}"


async def get_task_status(task_tracker, task_id: str) -> str:
    task = await task_tracker.get_task(task_id)
    if not task:
        return f"No task found with ID {task_id}"

    elapsed = task.elapsed if task.status != "running" else time.monotonic() - task.created_at
    status_line = (
        f"Task '{task.description}': **{task.status}**\n"
        f"Elapsed: {elapsed:.0f}s"
    )
    if task.result:
        result_str = str(task.result)
        status_line += f"\nResult: {result_str[:500]}"
    if task.error:
        status_line += f"\nError: {task.error[:500]}"
    return status_line


async def list_tasks(task_tracker) -> str:
    tasks = await task_tracker.list_tasks(limit=10)
    if not tasks:
        return "No background tasks."

    lines = ["Background tasks:"]
    for t in tasks:
        elapsed = t.elapsed if t.status != "running" else time.monotonic() - t.created_at
        lines.append(
            f"• `{t.task_id}` — {t.description[:50]} — "
            f"**{t.status}** ({elapsed:.0f}s)"
        )
    return "\n".join(lines)


async def cancel_task(task_tracker, task_id: str) -> str:
    task = await task_tracker.get_task(task_id)
    if not task:
        return f"No task found with ID {task_id}"
    if task.status != "running":
        return f"Task '{task.description}' is already {task.status}"

    success = await task_tracker.cancel_task(task_id)
    if success:
        return f"Cancelled task: {task.description}"
    return f"Failed to cancel task: {task.description}"
