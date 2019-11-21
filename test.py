#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import cv2
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchnet import meter
from model.resnet import resnet18
from dataloader.dataloader import HandDataloader
import torch.nn.functional as F
def show_result(images, show_size=(1024, 1024), blank_size=2, window_name="merge"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("Total pictures ： %s" % (max_count - count))
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, merge_img)
    cv2.waitKey(0)


def validate(my_val_dataloader, backbone):
    backbone.eval()
    confusion_matrix = meter.ConfusionMeter(6)
    imgshow = []
    with torch.no_grad():
        for img, labels_gt in my_val_dataloader:
            labels_gt.requires_grad = False
            labels_gt = labels_gt.cuda(non_blocking=True)

            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            backbone = backbone.cuda()
            labels = backbone(img)
            score = F.softmax(labels, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            prediction = np.array(prediction[0].cpu().numpy())
            labels_gt = np.array(labels_gt[0].cpu().numpy())
            # confusion_matrix.add(labels.data.squeeze(), labels_gt.data.squeeze())
            show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
            show_img = (show_img * 256).astype(np.uint8)
            np.clip(show_img, 0, 255)

            cv2.imwrite("xxx.jpg", show_img)
            img_clone = cv2.imread("xxx.jpg")
            img_clone = cv2.resize(img_clone, (112, 112))
            img_clone = cv2.putText(img_clone, 'P:' + str(prediction), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            img_clone = cv2.putText(img_clone, 'T:' + str(labels_gt), (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            if labels_gt != prediction:
                img_clone = cv2.putText(img_clone, 'Wrong!', (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (138, 43, 226), 2)
            imgshow.append(img_clone)

        show_result(imgshow)
        # cm_value = confusion_matrix.value()
        # accuracy = 100. * (cm_value[0][0] + cm_value[1][1]
        #                    + cm_value[2][2] + cm_value[3][3]
        #                    + cm_value[4][4] + cm_value[5][5]) / (cm_value.sum())

        # return cm_value, accuracy


def main(args):
    checkpoint = torch.load(args.model_path)

    backbone = resnet18(num_classes=6).cuda()

    backbone.load_state_dict(checkpoint['backbone'])

    my_val_dataset = HandDataloader(args.test_dataset, transforms=None, train=False)
    my_val_dataloader = DataLoader(
        my_val_dataset, batch_size=2, shuffle=True, num_workers=0)

    validate(my_val_dataloader, backbone)


def parse_args():
    parser = argparse.ArgumentParser(description='Gesture recognition Testing')
    parser.add_argument('--model_path', default="./checkpoints/snapshot/checkpoint.pth.tar", type=str)
    parser.add_argument('--test_dataset', default='./data/test.txt', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
