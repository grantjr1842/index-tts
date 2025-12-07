from serve_tars import lifespan, app
import asyncio
import sys

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

async def test():
    print("Starting test")
    async with lifespan(app):
        print("Lifespan entered")
        # Keep it alive for a bit
        await asyncio.sleep(5)
    print("Lifespan exited")

if __name__ == "__main__":
    asyncio.run(test())
