import os
import asyncio
import streetlevel.streetview
from streetlevel.dataclasses import Size
import gt_mapmaker
import tqdm
from aiohttp.client import ClientSession

GEN3_COUNTRIES = [
    "do",
    "bo",
    "uy",
    "tn",
    "rs",
    "me",
    "al",
    "mk",
    "ua",
    "ru",
    "il",
    "ps",
    "jo",
    "kg",
    "bt",
    "my",
    "gh",
    "ug",
    "bw",
    "ls",
    "sz",
    "gi",
    "pm",
    "gu",
    "as",
    "mp",
    "cw",
    "sk"
]  # a lot of gen3
DROPS_PER_COUNTRY = 20
RETRIES = 50
MAX_CONCURRENT = 128


async def get_gen3_pano(trigrid, sem, session, country):
    for _ in range(RETRIES):
        lon, lat = trigrid.rand_point()
        pano = await streetlevel.streetview.find_panorama_async(
            lat, lon, session, radius=10000
        )
        if not pano:
            continue

        # filter out trekker / gen4 / gen1
        if pano.source == "launch" and pano.image_sizes[-1] == Size(13312, 6656):
            # async with sem:  # limit simultaneous downloads
            try:
                image = await streetlevel.streetview.get_panorama_async(pano, session)
                if image:
                    with image:
                        image.save(f"samples/{country}/{pano.id}.jpg")
                return pano, image
            except Exception as e:
                print(f"download failed, retrying: {e}")
                continue
    return None, None


async def process_country(country, trigrid, session):
    print("generating", country)
    os.makedirs(f"samples/{country}", exist_ok=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [get_gen3_pano(trigrid, sem, session, country) for _ in range(DROPS_PER_COUNTRY)]

    results = []
    for coro in tqdm.tqdm(asyncio.as_completed(tasks), total=DROPS_PER_COUNTRY):
        pano, image = await coro
        if pano:
            results.append(pano.id)
    return results


async def main():
    gen3_trigrids = {
        c: gt_mapmaker.load_country_trigrids(c.upper()) for c in GEN3_COUNTRIES
    }
    session = ClientSession()
    async with session:
        for country in GEN3_COUNTRIES:
            await process_country(country, gen3_trigrids[country], session)


if __name__ == "__main__":
    asyncio.run(main())
