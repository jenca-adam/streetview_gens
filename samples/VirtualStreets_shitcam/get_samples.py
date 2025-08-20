import os
import asyncio
import streetlevel.streetview
from streetlevel.dataclasses import Size
import tqdm
from aiohttp.client import ClientSession
import json


async def get_shitcam_pano(panoid, session):
    pano = await streetlevel.streetview.find_panorama_by_id_async(
        panoid, session
    )
    if not pano:
        return None, None
    # filter out trekker / smallcam
    if pano.source=="launch" and pano.image_sizes[-1]==Size(13312, 6656):
        # async with sem:  # limit simultaneous downloads
        try:
            image = await streetlevel.streetview.get_panorama_async(pano, session)
            if pano and image:
                with image:
                    image.save(f"samples/dub/{pano.id}.jpg")
            return pano, image
        except Exception as e:
            print(f"download failed, retrying: {e}")
    return None, None



async def main():
    os.makedirs("samples/dub", exist_ok=True)
    session = ClientSession()
    with open("panoids.json") as f:
        panoids = json.load(f)
    tasks = [get_shitcam_pano(panoid, session) for panoid in panoids]
    async with session:
        for coro in tqdm.tqdm(asyncio.as_completed(tasks), total=len(panoids)):
            await coro

if __name__ == "__main__":
    asyncio.run(main())

