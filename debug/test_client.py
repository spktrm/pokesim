import asyncio
import aiofiles
import yaml
import sys
import os
from random import choices
from typing import Any, Dict


def weighted_random_sample(weights: list[int], k: int) -> int:
    total = len(weights)
    return choices(range(total), weights=weights, k=k)[0]


async def read_config(file_path: str) -> Dict[str, Any]:
    async with aiofiles.open(file_path, mode="r") as file:
        content = await file.read()
    return yaml.safe_load(content)


async def worker(worker_index: int, socket_path: str):
    reader, writer = await asyncio.open_unix_connection(socket_path)

    print(f"Worker {worker_index} connected to server!")

    try:
        while True:
            buffer = await reader.read(526)
            if not buffer:
                print("Disconnected from server")
                break

            player_index = buffer[1]
            legal_mask = buffer[-10:]
            random_action = weighted_random_sample(list(legal_mask), 1)
            message = f"{player_index}|d\n".encode()
            writer.write(message)
            await writer.drain()

    except Exception as e:
        print(e)
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    max_workers = max(int(sys.argv[1]) if len(sys.argv) > 1 else 1, 1)
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    config = await read_config(config_path)
    print(config)
    socket_path = config["socket_path"]

    await asyncio.gather(*(worker(i, socket_path) for i in range(max_workers)))


if __name__ == "__main__":
    asyncio.run(main())
