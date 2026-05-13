import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from daisy.utils.config_loader import load_config
from daisy.core.pipeline import Pipeline


async def main():
    config = load_config("config.yaml")
    pipeline = Pipeline(config)
    await pipeline.start()

    def shutdown():
        asyncio.create_task(pipeline.stop())

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    print("D.A.I.S.Y. v2 ready. Listening...")
    while True:
        try:
            await pipeline.run_turn()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
