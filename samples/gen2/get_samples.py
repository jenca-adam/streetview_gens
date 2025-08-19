import os
import asyncio
import streetlevel.streetview
from streetlevel.dataclasses import Size
import gt_mapmaker
import tqdm
from aiohttp.client import ClientSession

GEN2_COUNTRIES = ["ie", "im", "je", "mc", "mo"][1:]  # NEEDS MANUAL REVIEW
DROPS_PER_COUNTRY = 200
RETRIES = 50
MAX_CONCURRENT = 128


async def get_gen2_pano(trigrid, sem, session, country):
    for _ in range(RETRIES):
        lon, lat = trigrid.rand_point()
        pano = await streetlevel.streetview.find_panorama_async(
            lat, lon, session, radius=1000
        )
        if not pano:
            continue

        # filter out trekker / gen4 / gen1
        if pano.source == "launch" and pano.image_sizes[-1] == Size(13312, 6656):
            # async with sem:  # limit simultaneous downloads
            try:
                image = await streetlevel.streetview.get_panorama_async(pano, session)
                if not image:
                    return pano, None
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
    tasks = [
        get_gen2_pano(trigrid, sem, session, country) for _ in range(DROPS_PER_COUNTRY)
    ]

    results = []
    for coro in tqdm.tqdm(asyncio.as_completed(tasks), total=DROPS_PER_COUNTRY):
        pano, image = await coro
        if pano and image:
            with image:
                image.save(f"samples/{country}/{pano.id}.jpg")
            results.append(pano.id)
    return results


async def main():
    gen2_trigrids = {
        c: gt_mapmaker.load_country_trigrids(c.upper()) for c in GEN2_COUNTRIES
    }
    session = ClientSession()
    async with session:
        for country in GEN2_COUNTRIES:
            await process_country(country, gen2_trigrids[country], session)


if __name__ == "__main__":
    asyncio.run(main())
