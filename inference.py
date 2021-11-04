import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.modeling import VisionTransformer, CONFIGS


class testDataset(Dataset):
    def __init__(self, label_txt, img_dir, transform=None):
        self.img_dir = img_dir
        self.imgs = open(label_txt, 'r').read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,
                self.imgs[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, self.imgs[idx]


def main():
    test_transform = A.Compose(
        [
            #A.Resize(height=384, width=384),
            A.SmallestMaxSize(max_size=450),
            A.CenterCrop(height=384, width=384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = open('classes.txt', 'r').read().splitlines()

    # modify before evaluation
    label_txt = './testing_img_order.txt'
    img_dir = './testImg/'
    test_data = testDataset(label_txt, img_dir, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    config = CONFIGS["ViT-B_16"]
    net = VisionTransformer(config, 384, zero_head=True, num_classes=200, smoothing_value=0)
    net.load_state_dict(torch.load('./hw1_transFG')['model_state_dict'])
    net = net.to(dev)
    net.eval()

    f = open('answer.txt', 'w')
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(dev)
            pred = torch.argmax(net(x), dim=1)
            s = str(y[0]) + ' ' + classes[pred]
            f.write(s + '\n')
    f.close()
        
if __name__ == '__main__':
    main()
