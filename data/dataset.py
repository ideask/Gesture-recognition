# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt

class Hand(data.Dataset):
    
    def __init__(self, transforms=None, train=True):
        '''
        Get images, divide into train/val set
        '''

        self.train = train
        self.images_root = os.path.dirname(os.path.realpath(__file__))

        self._read_txt_file()

        print(" os.getcwd()", os.getcwd())
        print(" real cwd()",os.path.dirname(os.path.realpath(__file__)))
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
                
    def _read_txt_file(self):
        '''
        Get images_path, images_labels from txt_file
        '''
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = self.images_root + "/images/train.txt"
        else:
            txt_file = self.images_root + "/images/test.txt"

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.images_root+self.images_path[index]
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, int(label)
    
    def __len__(self):
        '''
        return the num of images
        '''
        return len(self.images_path)

if __name__ == '__main__':

    train_data = Hand(transforms=None, train=True)
    batch_size = 5
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    for images, labels in train_dataloader:
        plt.figure(figsize = (20,5))
        num = 0
        for image, label in zip(images, labels):
            image = image.numpy()
            image = image.transpose((1, 2, 0))
            num += 1
            plt.subplot(1,batch_size,num)
            plt.imshow(image)
            plt.title('Prediction number:  {}'.format(label))
        plt.show()
