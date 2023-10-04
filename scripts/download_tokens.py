import json
import aiohttp
import asyncio
import uvloop

from typing import List, Mapping, Any
from tqdm import tqdm

BASE_URL = "https://raw.githubusercontent.com/pkmn/ps/main/dex/data/{}.json"
KEYS = ["abilities", "species", "types", "items", "moves"]
URLS = [BASE_URL.format(key) for key in KEYS]


def aggregate(mapping: Mapping[str, Any]):
    all = set()
    for gen, items in mapping.items():
        all.update(set(items))
    return list(sorted(all))


async def fetch(url: str, session: aiohttp.ClientSession, progress: tqdm):
    async with session.get(url) as response:
        text = await response.json(content_type=None)
        progress.update(1)
        return text


async def download_all(urls: List[str]):
    async with aiohttp.ClientSession() as session:
        progress = tqdm(total=len(urls))
        tasks = [fetch(url, session, progress) for url in urls]
        return await asyncio.gather(*tasks)


def main():
    uvloop.install()
    data = {key: value for key, value in zip(KEYS, asyncio.run(download_all(URLS)))}
    extra_moves = [
        f"hiddenpower{type}{bp}"
        for bp in [30, 60, 70]
        for type in aggregate(data["types"])
    ]
    extra_moves += [f"return102"]
    for key, values in data.items():
        if key == "moves":
            values["extra"] = {key: {} for key in extra_moves}
        data[key] = {key: index for index, key in enumerate(aggregate(values))}
    return data


if __name__ == "__main__":
    data = main()

    with open("src/tokens.json", "w") as f:
        json.dump(data, f)
