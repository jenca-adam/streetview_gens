import json
import requests
import tqdm
import streetlevel.streetview
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
PANOID_PATTERN=re.compile(r"panoid\%3D([^%]+)\%")
def get_pano(link):
    try:
        redirected = requests.get(link, allow_redirects=False).headers["location"]
        search_result = PANOID_PATTERN.search(redirected)
        if search_result:
            return search_result.group(1)
    except Exception as e:
        print(e.__class__.__name__, e)
def main():
    with open("links.json") as f:
        links = json.load(f)
    panoids = []
    with ThreadPoolExecutor(max_workers=128) as executor:
        for pano in tqdm.tqdm(executor.map(get_pano, links), total=len(links)):
            if pano:
                panoids.append(pano)
                with open("panoids.json",'w') as f:
                    json.dump(panoids, f)

if __name__=="__main__":
    main()
