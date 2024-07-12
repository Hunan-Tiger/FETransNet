import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss,BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        #inputs = torch.softmax(inputs, dim=1)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# snapshot_path : 输出路径
def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    # transform.Normalize()把 0-1 变换到 (-1,1)
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size,norm_x_transform=x_transforms, norm_y_transform=y_transforms)

    print("The length of train set is: {}".format(len(db_train)))

    # -------------------------

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # ============ for local ======================
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # ==================================

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    focal_loss = FocalLoss()
    dice_loss = DiceLoss(num_classes)
    edge_loss = FocalLoss()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)  # tqdm 进度条

    # ---------------- begin train -----------------
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("data shape-------------", image_batch.shape, label_batch.shape， edge_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()

            edge_batch = sampled_batch['edge']
            edge_batch = edge_batch.squeeze(1).cuda()

            outputs, edges = model(image_batch, edge=True)  # edge 24 1 224 224
            # outputs = model(image_batch)
            
            loss_focal = focal_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            loss_edge = 0.0
            
            for k in range(len(edges)):
                loss_edge += (edge_loss(edges[k], edge_batch.long()))

            loss_seg = 0.4 * loss_focal + 0.6 * loss_dice
            
            loss = 0.8 * loss_seg + 0.2 * loss_edge
            #loss = loss_seg

            # print("loss---------------", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9  # lr 逐渐减小
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            if loss_edge !=0.0:
                writer.add_scalar('info/loss_edge', loss_edge, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            
            if loss_edge !=0.0:
                logging.info('iteration %d : loss : %f, loss_edge: %f, loss_focal: %f, loss_dice: %f' % (
                iter_num, loss.item(),loss_edge.item(),loss_focal.item(),
                loss_dice.item()))
            else:
                logging.info('iteration %d : loss : %f, loss_focal: %f, loss_dice: %f' % (
                iter_num, loss.item(), loss_focal.item(),
                loss_dice.item())) 

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())  # 规 0-1
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        # 进度过半并且是 50 的倍数
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

    # os.system("shutdown")