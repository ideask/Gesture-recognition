# -*- coding:utf-8 -*-
import os

def dataprocess(is_train):
    root_path = os.path.dirname(os.path.realpath(__file__))
    train_label_path = os.path.join(root_path, 'images/train.txt')
    test_label_path = os.path.join(root_path, 'images/test.txt')
    output_train_label = os.path.join(root_path, 'train.txt')
    output_test_label = os.path.join(root_path, 'test.txt')

    if is_train:
        label_path = train_label_path
        output_label = output_train_label
    else:
        label_path = test_label_path
        output_label = output_test_label

    labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        path = line.split(' ')[0]
        path = root_path + path
        if os.path.isfile(path):
            print(path)
            sign = line.split(' ')[1]
            label = '{} {}'.format(path, sign)
            labels.append(label)

    with open(output_label, 'w') as f:
        for label in labels:
            f.writelines(label)


if __name__ == '__main__':
    dataprocess(is_train=True)
    dataprocess(is_train=False)
