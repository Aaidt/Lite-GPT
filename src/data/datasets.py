import requests
from pathlib import Path
from omegaconf import OmegaConf

shakespeare_conf = OmegaConf.load("../../configs/data/shakespeare.yaml")
url = shakespeare_conf.url
raw_file = Path(shakespeare_conf.raw_file)

raw_file.parent.mkdir(parents=True, exist_ok=True)

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.text

    with open(raw_file, "w") as f:
        f.write(data)

except requests.RequestException as e:
    print("Error fetching shakespeare dataset", e)