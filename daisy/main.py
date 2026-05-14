import asyncio
import signal
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from daisy.utils.config_loader import load_config
from daisy.core.pipeline import Pipeline


async def main():
    load_dotenv()
    config = load_config("config.yaml")
    pipeline = Pipeline(config)
    await pipeline.start()

    shutdown_event = asyncio.Event()

    def signal_handler():
        print("\nShutting down...", file=sys.stderr)
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    print("D.A.I.S.Y. v2 ready. Listening...")
    try:
        while not shutdown_event.is_set():
            print("  [main] Listening...", file=sys.stderr)
            turn_task = asyncio.create_task(pipeline.run_turn())
            done, pending = await asyncio.wait(
                [turn_task, asyncio.create_task(shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if turn_task in done:
                exc = turn_task.exception()
                if exc:
                    print(f"  [main] Error: {exc}", file=sys.stderr)
    finally:
        await pipeline.stop()
        print("D.A.I.S.Y. stopped.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
