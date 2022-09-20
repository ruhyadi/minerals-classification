"""
Download dataset from source (google drive)
"""

import os
import gdown
import dotenv

def get_dataset(unzip: bool = True):
    """ Download dataset from google drive """
    dotenv.load_dotenv()
    endpoint = os.getenv("ENDPOINT")
    url = f'https://drive.google.com/uc?id={endpoint}'
    output = 'data/dataset.zip'
    gdown.download(url, output, quiet=False)

    # unzip dataset
    os.system('unzip -q data/dataset.zip -d data/') if unzip else None

if __name__ == '__main__':
    get_dataset()