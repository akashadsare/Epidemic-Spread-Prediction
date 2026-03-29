import os
import requests
import logging
import sys

# Ensure src is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_URLS = {
    "jhu_cases": "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
    "owid": "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
    "google_mobility": "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv",
}


def download_file(url, target_path):
    if os.path.exists(target_path):
        logger.info(f"File already exists: {target_path}")
        return

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    logger.info(f"Downloading {url} to {target_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Successfully downloaded {target_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")


def main():
    logger.info("Starting data download...")
    # Map friendly names to the centralized PathConfig attributes
    download_map = {
        "jhu_cases": PATHS.jhu_cases,
        "owid": PATHS.owid,
        "google_mobility": PATHS.google_mobility,
    }
    
    for key, url in DATA_URLS.items():
        download_file(url, download_map[key])
    logger.info("Data download process completed.")


if __name__ == "__main__":
    main()
