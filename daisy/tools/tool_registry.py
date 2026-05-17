from daisy.tools import system_tools, web_tools, file_tools, background_tools

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_time_date",
            "description": "Get the current time, date, and timezone",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get CPU, RAM, and disk usage of the server",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command on the server. Only pre-approved commands are allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a timer that will announce a message after a duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Seconds from now to fire the reminder (max 86400)",
                    },
                    "message": {
                        "type": "string",
                        "description": "The reminder message",
                    },
                },
                "required": ["duration_seconds", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Returns titles, snippets, and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (max 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_url",
            "description": "Fetch a URL and extract its readable content",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem. Only files in allowed directories can be read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Only files in allowed directories can be written.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_task",
            "description": "Start a background task for time-consuming work like research or shell commands. "
                           "You can check progress later with get_task_status. "
                           "If the user says 'notify me' or 'let me know', set notify_on_complete to true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["shell", "sub_agent"],
                        "description": "shell: run a shell command. sub_agent: spawn a worker AI for research",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable task description",
                    },
                    "payload": {
                        "type": "object",
                        "description": "For shell: {'command': '...'}. For sub_agent: {'prompt': '...'}",
                    },
                    "notify_on_complete": {
                        "type": "boolean",
                        "description": "Set to true if the user asked to be notified when done",
                    },
                },
                "required": ["task_type", "description", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_opencode_task",
            "description": "Start a background OpenCode coding task. Use this for writing, fixing, or analyzing code. "
                           "Give a detailed prompt. If the user says 'notify me', set notify_on_complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed prompt for OpenCode",
                    },
                    "project_dir": {
                        "type": "string",
                        "description": "Project directory (must be within allowed paths)",
                    },
                    "notify_on_complete": {
                        "type": "boolean",
                        "description": "Set to true if the user asked to be notified when done",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_status",
            "description": "Check the status and result of a background task by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID returned when the task was spawned",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all recent background tasks and their statuses",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_task",
            "description": "Cancel a running background task",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to cancel",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
]


def build_handlers(config, task_tracker, announcement_queue, llm):
    return {
        "get_time_date": lambda: system_tools.get_time_date(),
        "get_system_info": lambda: system_tools.get_system_info(config),
        "run_command": lambda command: system_tools.run_command(config, command),
        "set_reminder": lambda duration_seconds, message: system_tools.set_reminder(
            config, announcement_queue, duration_seconds, message,
        ),
        "web_search": lambda query, max_results=5: web_tools.web_search(query, max_results),
        "browse_url": lambda url: web_tools.browse_url(url),
        "read_file": lambda path: file_tools.read_file(config, path),
        "write_file": lambda path, content: file_tools.write_file(config, path, content),
        "spawn_task": lambda task_type, description, payload, notify_on_complete=False: (
            background_tools.spawn_task(
                config, task_tracker, llm, task_type, description, payload, notify_on_complete,
            )
        ),
        "spawn_opencode_task": lambda prompt, project_dir=None, notify_on_complete=False: (
            background_tools.spawn_opencode_task(
                config, task_tracker, prompt, project_dir, notify_on_complete,
            )
        ),
        "get_task_status": lambda task_id: background_tools.get_task_status(task_tracker, task_id),
        "list_tasks": lambda: background_tools.list_tasks(task_tracker),
        "cancel_task": lambda task_id: background_tools.cancel_task(task_tracker, task_id),
    }
