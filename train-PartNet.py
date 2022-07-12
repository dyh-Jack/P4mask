from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from scheduler import WarmupMultiStepLR

from datasets.PartNet_ego import *
import models.PartNet as Models


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, mode='ego'):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for clip_ego, clip_trans, label in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()

        label = label.to(device)
        if mode == 'ego':
            clip_ego = clip_ego.to(device)
            output = model(clip_ego, None).transpose(1,2)
        if mode == 'trans':
            clip_trans = clip_trans.to(device)
            output = model(clip_trans, None).transpose(1,2)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        lr_scheduler.step()
        sys.stdout.flush()

def evaluate(model, criterion, data_loader, device, print_freq, mode = 'ego'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_loss = 0
    total_correct = 0
    total_pred_class = [0] * 30
    total_correct_class = [0] * 30
    total_class = [0] * 30

    with torch.no_grad():
        for clip_ego, clip_trans, label in metric_logger.log_every(data_loader, print_freq, header):
            label = label.to(device)
            if mode == 'ego':
                clip_ego = clip_ego.to(device)
                output = model(clip_ego, None).transpose(1,2)
            if mode == 'trans':
                clip_trans = clip_trans.to(device)
                output = model(clip_trans, None).transpose(1,2)
            loss = criterion(output, label)

            output = output.cpu().numpy()
            pred = np.argmax(output,1)
            label = label.cpu().numpy()
            correct = np.sum(pred == label)
            total_correct += correct
            for c in range(30):
                total_pred_class[c] += np.sum((pred == c)|(label == c))
                total_correct_class[c] += np.sum((pred == c) & (label == c))
                total_class[c] += np.sum((label == c))

            metric_logger.update(loss=loss.item())

    ACCs = []
    for c in range(30):
        acc = total_correct_class[c] / float(total_class[c])
        if total_class[c] == 0:
            acc = 0
        print('eval acc of %s:\t %f'%(c, acc))
        ACCs.append(acc)
    print(' * Eval accuracy: %f'% (np.mean(np.array(ACCs))))

    IoUs = []
    for c in range(30):
        iou = total_correct_class[c] / float(total_pred_class[c])
        if total_pred_class[c] == 0:
            iou = 0
        print('eval mIoU of %s:\t %f'%(c, iou))
        IoUs.append(iou)
    print(' * Eval mIoU:\t %f'%(np.mean(np.array(IoUs))))

def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')
    # device = torch.device('cpu')


    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = PartNet_ego(
            data_root=args.data_path,
            mode = 'train'
    )

    dataset_test = PartNet_ego(
            data_root=args.data_path,
            mode = 'test'
    )

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)

    print("Creating model")
    
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, num_classes=30)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion_train = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion_train, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq, mode='ego')

        evaluate(model, criterion_test, data_loader_test, device=device, print_freq=args.print_freq, mode='ego')

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Transformer Model Training')

    parser.add_argument('--data-path', default='/share/zhuoyang/PartNet_ego/Chair-2', help='data path')
    
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # P4D
    parser.add_argument('--radius', default=0.9, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=2, type=int, help='transformer depth')
    parser.add_argument('--head', default=4, type=int, help='transformer head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=48, type=int)
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[30, 40, 50], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
