import os
import httpx
import yaml

token = os.getenv("FISHAUDIO_API_KEY")

def run():
    content = open_tts_yaml("tts.yaml")
    update_yaml(content=content,dst="tts.yaml")

def fetch_from_fishaudio():
    uri = "https://api.fish.audio/model"
    params = {
        "page_size": 20,
        "page_number": 1,
        "sort_by": "score",
        "language": "zh",
        "title_language": "zh"
    }

    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    r = httpx.get(uri, params=params, headers=headers)
    if r.status_code != 200:
        r.raise_for_status()
    return r.json()

def open_tts_yaml(filename):
    with open(filename) as stream:
        try:
            content = yaml.safe_load(stream=stream)
        except Exception as e:
            raise e
    return content

def update_yaml(content ,dst):
    new_voices = fetch_from_fishaudio()
    voices = []
    for v in new_voices["items"]:
        voices.append({
            "mode": v["_id"],
            "name": v["title"],
            "Language": v["languages"]
        })
    content["model_properties"]["voices"] = voices
    content["model_properties"]["default_voice"] = voices[0]["mode"]

    with open(dst, "w") as f:
        yaml.safe_dump(content, f)
    

if __name__ == "__main__":
    run()
