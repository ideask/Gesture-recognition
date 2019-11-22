#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataloader.dataloader import HandDataloader
from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torch import nn
from torchnet import meter
# from utils.parallel import DataParallelModel, DataParallelCriterion

def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, backbone, criterion, optimizer, loss_meter, confusion_matrix, cur_epoch):
    loss_meter.reset()
    confusion_matrix.reset()

    for img, labels_gt in train_loader:

        img.requires_grad = False
        img = img.cuda(non_blocking=True)

        labels_gt.requires_grad = False
        labels_gt = labels_gt.cuda(non_blocking=True)

        backbone = backbone.cuda()
        labels = backbone(img)
        loss = criterion(labels, labels_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.item())
        confusion_matrix.add(labels.data.squeeze(), labels_gt.data.squeeze())
    loss = loss_meter.value()[0]

    return loss, confusion_matrix.value()


def validate(my_val_dataloader, backbone, criterion, epoch):
    backbone.eval()
    confusion_matrix = meter.ConfusionMeter(6)
    with torch.no_grad():
        for img, labels_gt in my_val_dataloader:
            labels_gt.requires_grad = False
            labels_gt = labels_gt.cuda(non_blocking=True)

            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            backbone = backbone.cuda()
            labels = backbone(img)
            confusion_matrix.add(labels.data.squeeze(), labels_gt.data.squeeze())

        cm_value = confusion_matrix.value()
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]
                           + cm_value[2][2] + cm_value[3][3]
                           + cm_value[4][4] + cm_value[5][5]) / (cm_value.sum())
        return cm_value, accuracy


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    backbone = resnet18(num_classes=6).cuda()
    # backbone = resnet34(num_classes=6).cuda()
    # backbone = resnet50(num_classes=6).cuda()
    # backbone = resnet101(num_classes=6).cuda()
    # backbone = resnet152(num_classes=6).cuda()

    if args.resume != '':
        logging.info('Load the checkpoint:{}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        logging.info('Last train epoch:{}'.format(checkpoint['epoch']))
        backbone.load_state_dict(checkpoint['backbone'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [{
            'params': backbone.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)

    # step 3: data
    # argumetion
    mydataset = HandDataloader(args.dataroot, transforms=None, train=True)
    dataloader = DataLoader(
        mydataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    my_val_dataset = HandDataloader(args.val_dataroot, transforms=None, train=False)
    my_val_dataloader = DataLoader(
        my_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    # 平滑处理之后的损失
    loss_meter = meter.AverageValueMeter()
    # 混淆矩阵
    confusion_matrix = meter.ConfusionMeter(6)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        loss, train_cm = train(dataloader, backbone, criterion, optimizer, loss_meter, confusion_matrix, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'backbone': backbone.state_dict(),
        }, filename)

        test_cm, accuracy = validate(my_val_dataloader, backbone, criterion, epoch)
        logging.info("epoch:{epoch},lr:{lr},loss:{loss}\ntrain_cm:\n{train_cm}\nval_cm:\n{val_cm}".format(
                epoch=epoch, loss=str(loss), val_cm=str(test_cm),
                train_cm=str(train_cm), lr=get_learning_rates(optimizer)))
        scheduler.step(loss, epoch)
        # 第一个参数可以简单理解为保存图的名称，第二个参数是可以理解为Y轴数据，第三个参数可以理解为X轴数据
        writer.add_scalars('accuracy/loss', {'accuracy': accuracy, 'train loss': loss}, epoch)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Gesture Recognition Training')
    # general
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    # -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=1000, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoints/snapshot/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoints/train.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoints/tensorboard", type=str)
    # -- load snapshot
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./data/train.txt',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='./data/test.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=128, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
