import os
import urllib.request
import json
from zipfile import ZipFile

def download_data_and_class_json(data_dir:str):
    print("Download fine-tuing data... ", end="")

    class_json_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    hymenoptera_zip_url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

    os.makedirs(data_dir, exist_ok=True)

    # imagenet_class_index.json
    save_path = os.path.join(data_dir, "imagenet_class_index.json")
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(class_json_url, save_path)

    # hymenoptera_data.zip
    save_path = os.path.join(data_dir, "hymenoptera_data.zip")
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(hymenoptera_zip_url, save_path)

        zip = ZipFile(save_path)
        zip.extractall(data_dir)

        os.remove(save_path)

    print("Done")
    
class ClassIndexLookUp():
    def __init__(self, data_dir:str) -> str:
        json_path = os.path.join(data_dir, 'imagenet_class_index.json')
        with open(json_path, 'r') as f:
            self.class_name_list = json.load(f)

    def __call__(self, idx:int):
        return self.class_name_list[str(idx)][1]

if __name__ == "__main__":
    download_data_and_class_json(data_dir='./data')