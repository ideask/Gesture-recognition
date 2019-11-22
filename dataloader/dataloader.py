# -*- coding:utf-8 -*-
import os
import cv2
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt


class HandDataloader(data.Dataset):

    def __init__(self, file_path, transforms=None, train=True):
        self.train = train
        self._read_txt_file(file_path)


        if transforms is None:
            # mean(R,G,B) std(R,G,B),Normalized_image=(image-mean)/std
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def _read_txt_file(self, file_path):
        self.images_path = []
        self.images_labels = []

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, int(label)

    def __len__(self):
        return len(self.images_path)


if __name__ == '__main__':
    train_label = '/home/kenny/Desktop/Gesture-recognition/data/train.txt'
    train_data = HandDataloader(train_label, transforms=None, train=True)
    batch_size = 5
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    for images, labels in train_dataloader:
        plt.figure(figsize=(20, 5))
        num = 0
        for image, label in zip(images, labels):
            image = image.numpy()
            image = image.transpose((1, 2, 0))
            num += 1
            plt.subplot(1, batch_size, num)
            plt.imshow(image)
            plt.title('Prediction number:  {}'.format(label))
        plt.show()
