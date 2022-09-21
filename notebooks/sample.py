# import numpy as np
# import cv2
# from PIL import Image
# import torch
# from torchvision import transforms

# preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# img = Image.open("data/minet/biotite/0001.jpg")
# print(img.size)

# img = preprocess(img).to("cuda")
# print(img.shape)

# img = torch.unsqueeze(img, 0)

# # img = img.reshape(1, *img.shape).to('cuda')
# print(img.shape)

from pathlib import Path

img_path = "data/minet/biotite/0001.jpg"

# check if extension is jpg
if Path(img_path).suffix == ".jpg":
    print("yes")
