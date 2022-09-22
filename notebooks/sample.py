# # import numpy as np
# # import cv2
# # from PIL import Image
# # import torch
# # from torchvision import transforms

# # preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# # img = Image.open("data/minet/biotite/0001.jpg")
# # print(img.size)

# # img = preprocess(img).to("cuda")
# # print(img.shape)

# # img = torch.unsqueeze(img, 0)

# # # img = img.reshape(1, *img.shape).to('cuda')
# # print(img.shape)

# # from pathlib import Path

# # img_path = "data/minet/biotite/0001.jpg"

# # # check if extension is jpg
# # if Path(img_path).suffix == ".jpg":
# #     print("yes")


# # from torchmetrics import PrecisionRecallCurve
# # from sklearn.metrics import precision_recall_curve
# # import torch
# # import numpy as np
# # import matplotlib.pyplot as plt

# # pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
# #                      [0.05, 0.75, 0.05, 0.05, 0.05],
# #                      [0.05, 0.05, 0.75, 0.05, 0.05],
# #                      [0.05, 0.05, 0.05, 0.75, 0.05]])
# # target = torch.tensor([0, 1, 3, 2])
# # pr_curve = PrecisionRecallCurve(num_classes=5)
# # precision, recall, thresholds = pr_curve(pred, target)

# # for i in range(5):
# #     plt.plot(recall[i], precision[i], label=f"class {i}")
# # plt.legend()
# # plt.savefig("pr_curve.png")


# # # plot matrics
# # plt.plot(recall, precision)
# # plt.xlabel("Recall")
# # plt.ylabel("Precision")
# # plt.title("Precision-Recall Curve")

# # # save
# # plt.savefig("pr_curve.png")

# # print(precision)
# # print(recall)
# # print(thresholds)

# # confusion matrix
# from torchmetrics import ConfusionMatrix
# import torch
# import pandas as pd
# import seaborn as sn
# import matplotlib.pyplot as plt

# confusion_matrix = ConfusionMatrix(num_classes=5)
# pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
#                         [0.05, 0.75, 0.05, 0.05, 0.05],
#                         [0.05, 0.05, 0.75, 0.05, 0.05],
#                         [0.05, 0.05, 0.05, 0.75, 0.05]])
# target = torch.tensor([0, 1, 3, 2])
# confusion_matrix(pred, target)
# cfm = confusion_matrix.compute().numpy()

# cfm_df = pd.DataFrame(cfm, index=["biotite", "bornite", "chrysocolla", "malachite", "muscovite"], columns=["biotite", "bornite", "chrysocolla", "malachite", "muscovite"])
# img = sn.heatmap(cfm_df, annot=True, fmt="d")
# # convert to numpy array
# img = img.get_figure()

# print(type(img.get_figure()))

# # # log confusion matrix
# # confmat = ConfusionMatrix(num_classes=self.hparams.num_classes, normalize="true").to(self.device)
# # confmat(preds, targets)
# # confmat_df = pd.DataFrame(confmat.compute().cpu().numpy(), columns=categories, index=categories)
# # confmat_img = sns.heatmap(confmat_df, annot=True, fmt=".2f").get_figure()
# # wandb_logger.log({"val/confmat": wandb.Image(confmat_img)})
# # plt.clf() # reset confusion matrix chart


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch

# pr curve
from torchmetrics import PrecisionRecallCurve

pr_curve = PrecisionRecallCurve(num_classes=5)
pred = torch.tensor(
    [
        [0.75, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.75, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.75, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.75, 0.05],
    ]
).to("cuda")
target = torch.tensor([0, 1, 3, 2]).to("cuda")
pr_curve(pred, target)
precision, recall, thresholds = pr_curve.compute()

# plot pr curve
for i in range(5):
    # plot with seaborn
    plt.plot(recall[i], precision[i], label=f"class {i}")
plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

# save
plt.savefig("pr_curve.png")

print(precision)
print(recall)
print(thresholds)
