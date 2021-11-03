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


class trainDataset(Dataset):
    def __init__(self, label_txt, img_dir, transform=None, mode='all'):
        # read labels
        self.labels = []
        train_txt = open(label_txt, 'r').readlines()
        imgs, labels = zip(*(line.split() for line in train_txt))
        for s in list(labels):
            # class idx from 0 to 199
            self.labels.append(int(s[:3])-1)
        
        self.img_dir = img_dir
        self.imgs = list(imgs)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.imgs[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, self.labels[idx]

def main():    
    train_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=450),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=384, width=384),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.GaussNoise(),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # modify before training
    label_txt = './training_labels.txt'
    img_dir = './trainImg/'
    train_data = trainDataset(label_txt, img_dir, transform=train_transform)
    train_dl = DataLoader(train_data, batch_size=16, shuffle=True)

    config = CONFIGS["ViT-B_16"]
    net = VisionTransformer(config, 384, zero_head=True, num_classes=200, smoothing_value=0)
    net = net.to(dev)

    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5*1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    criterion = nn.CrossEntropyLoss()
    epochs = 200
    total_loss = 0

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            # transFG returns loss and output logits
            loss, logits = net(xb, yb)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(epoch+1, total_loss)

    torch.save(net.state_dict(), './hw1_transFG')

if __name__ == '__main__':
    main()