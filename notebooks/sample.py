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

# from pathlib import Path

# img_path = "data/minet/biotite/0001.jpg"

# # check if extension is jpg
# if Path(img_path).suffix == ".jpg":
#     print("yes")


from torchmetrics import PrecisionRecallCurve
from sklearn.metrics import precision_recall_curve
import torch
import numpy as np
import matplotlib.pyplot as plt

pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
                     [0.05, 0.75, 0.05, 0.05, 0.05],
                     [0.05, 0.05, 0.75, 0.05, 0.05],
                     [0.05, 0.05, 0.05, 0.75, 0.05]])
target = torch.tensor([0, 1, 3, 2])
pr_curve = PrecisionRecallCurve(num_classes=5)
precision, recall, thresholds = pr_curve(pred, target)

for i in range(5):
    plt.plot(recall[i], precision[i], label=f"class {i}")
plt.legend()
plt.savefig("pr_curve.png")


# # plot matrics
# plt.plot(recall, precision)
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")

# # save
# plt.savefig("pr_curve.png")

print(precision)
print(recall)
print(thresholds)