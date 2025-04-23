import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
from datetime import datetime
import nibabel as nib
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import json



# from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.dsc import DiceScoreCoefficient
# from loss_functions.focal_loss import FocalLoss
from loss_functions.metrics import dice_pytorch, SegmentationMetric

from loss_functions.unified_loss import (
    symmetric_unified_focal_multiclass,
    asymmetric_unified_focal_multiclass)

from models import sam_seg_gabor_model_registry2
from dataset import generate_dataset, generate_test_loader
from evaluate import test_synapse, test_acdc, test_brats, test_material


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--model_type', type=str, default="vit_l", help='path to splits file')
parser.add_argument('--src_dir', type=str, default=None, help='path to splits file')
parser.add_argument('--data_dir', type=str, default=None, help='path to datafolder')
parser.add_argument('--out_dir', type=str, default="output_experiment", help='path to output folder')
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--classes", type=int, default=8)
parser.add_argument("--do_contrast", default=False, action='store_true')
parser.add_argument("--slice_threshold", type=float, default=0.05)
parser.add_argument("--num_classes", type=int, default=14)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--tr_size", type=int, default=1)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--load_saved_model", action='store_true',
                        help='whether freeze encoder of the segmenter')
parser.add_argument("--saved_model_path", type=str, default=None)
parser.add_argument("--load_pseudo_label", default=False, action='store_true')
parser.add_argument("--dataset", type=str, default="synapse")
parser.add_argument('--loss_type',
                    choices=['sym','asym'],
                    default='sym',
                    help="Which unified focal loss to use")
parser.add_argument('--uf_weight', type=float, default=0.5,
                    help="λ weight between FTL and FL (unified focal)")
parser.add_argument('--uf_delta',  type=float, default=0.6,
                    help="δ parameter (balance of FP/FN)")
parser.add_argument('--uf_gamma',  type=float, default=0.5,
                    help="γ focal parameter (hard‐example emphasis)")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model

    if args.model_type=='vit_h':
        model_checkpoint = 'cp/sam_vit_h_4b8939.pth'
    elif args.model_type == 'vit_l':
        model_checkpoint = 'cp/sam_vit_l_0b3195.pth'
    elif args.model_type == 'vit_b':
        model_checkpoint = 'cp/sam_vit_b_01ec64.pth'

    model = sam_seg_gabor_model_registry2[args.model_type](num_classes=args.num_classes, checkpoint=model_checkpoint)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    


    # freeze weights in the image_encoder
    for name, param in model.named_parameters():
        if param.requires_grad and "image_encoder" in name or "iou" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
        # param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler = generate_dataset(args)

 

    now = datetime.now()
    # args.save_dir = "output_experiment/Sam_h_seg_distributed_tr" + str(args.tr_size) # + str(now)[:-7]
    args.save_dir = os.path.join(args.out_dir, args.save_dir)
    print(args.save_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard' + str(gpu)))

    
    best_dice = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch, args, writer)
        val_dice = validate(val_loader, model, epoch, args, writer)

        # Adjust learning rate based on validation loss
        if epoch >= 10:
            scheduler.step(val_dice)

        is_best = val_dice > best_dice
        best_dice = max(best_dice, val_dice)


        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            filename = os.path.join(args.save_dir, 'checkpoint_b%d.pth.tar' % (args.batch_size))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': (model.module if hasattr(model, 'module') else model).mask_decoder.state_dict(),
                'pe_layer_state_dict': (model.module if hasattr(model, 'module') else model).pe_layer.state_dict(),
                'tex_conv_state_dict':   (model.module if hasattr(model, 'module') else model).tex_proj.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=filename)

    # —————— At the end of training, save a “final” checkpoint ——————
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):

        final_fname = os.path.join(args.save_dir, 'checkpoint_final_b%d.pth.tar' % args.batch_size)
        save_checkpoint({
            'epoch': args.epochs,
            'mask_decoder_state_dict': (model.module if hasattr(model, 'module') else model).mask_decoder.state_dict(),
            'pe_layer_state_dict': (model.module if hasattr(model, 'module') else model).pe_layer.state_dict(),
            'tex_conv_state_dict':   (model.module if hasattr(model, 'module') else model).tex_proj.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=True, filename=final_fname)

    test(model, args)
    if args.dataset == 'synapse':
        test_synapse(args)
    elif args.dataset == 'ACDC' or args.dataset == 'acdc':
        test_acdc(args)
    elif args.dataset == 'brats':
        test_brats(args)
    elif args.dataset == 'material':
        test_material(args)


def train(train_loader, model, optimizer, scheduler, epoch, args, writer):

    # choose and instantiate the multiclass unified focal loss
    if args.loss_type == 'sym':
        criterion = symmetric_unified_focal_multiclass(
            delta=args.uf_delta,
            gamma=args.uf_gamma,
            weight=args.uf_weight
        )
        rare_classes = None
    else:
        rare_classes, freqs = compute_rare_classes_percentile(train_loader, args.num_classes)
        for (i, c) in enumerate(rare_classes):
            print(f"Rare class {i}: {c} with frequency {freqs[c]}")
        criterion = asymmetric_unified_focal_multiclass(
            rare_classes=rare_classes,
            delta=args.uf_delta,
            gamma=args.uf_gamma,
            weight=args.uf_weight
        )
    

    # switch to train mode
    model.train()


    end = time.time()
    for i, tup in enumerate(train_loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        if args.gpu is not None:
            img = tup[0].float().cuda(args.gpu, non_blocking=True)
            label = tup[1].long().cuda(args.gpu, non_blocking=True)
        else:
            img = tup[0].float()
            label = tup[1].long()
        b, c, h, w = img.shape

        # compute output
        # mask size: [batch*num_classes, num_multi_class, H, W], iou_pred: [batch*num_classes, 1]
        mask, iou_pred = model(img)
        mask = mask.view(b, -1, h, w)
        iou_pred = iou_pred.squeeze().view(b, -1)

        pred_softmax = F.softmax(mask, dim=1)
        # loss = ce_loss(mask, label.squeeze(1)) + dice_loss(pred_softmax, label.squeeze(1))
               # + dice_loss(pred_softmax, label.squeeze(1))

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss

        # px_loss_value = px_loss(mask, label.squeeze(1))
        # dice_loss_value = dice_loss(pred_softmax, label.squeeze(1))

        # Integrate new parameters by weighting the losses
        # loss = args.px_loss_weight * px_loss_value + args.dice_weight * dice_loss_value

        # 1) logits → probs; 2) one‑hot encode
        y_onehot   = (
            F.one_hot(label.squeeze(1), num_classes=args.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        loss = criterion(pred_softmax, y_onehot)



        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        writer.add_scalar('train_loss', loss, global_step=i + epoch * len(train_loader))

        if i % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss:.4f}'.format(epoch, i, len(train_loader), loss=loss.item()))




def compute_rare_classes_percentile(train_loader, num_classes, percentile=40):
    """
    Identify 'rare' classes as those whose pixel‐frequency falls below
    the given percentile of the class‐frequency distribution.

    Args:
        train_loader (DataLoader): yields tuples (image, label) with
                                  label shape (N,1,H,W) or (N,H,W).
        num_classes (int): total number of classes C.
        percentile (float): e.g. 25 for the 25th percentile.

    Returns:
        rare_classes (List[int]): class indices with freq < percentile cut.
        freqs       (np.ndarray): array of length C with class frequencies.
    """
    # 1) Accumulate pixel counts
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    for _, labels in train_loader:
        lbl = labels.squeeze(1).cpu().numpy().astype(np.int64)  # (N,H,W)
        flat = lbl.reshape(-1)
        hist = np.bincount(flat, minlength=num_classes)
        class_counts += hist
        total_pixels += flat.size

    # 2) Compute frequencies
    freqs = class_counts / total_pixels  # shape (C,)
    print(freqs)

    # 3) Find the cutoff at the given percentile
    cut = np.percentile(freqs, percentile)
    print(f"Percentile {percentile} cut: {cut:.4f}")

    # 4) All classes with freq < cut are "rare"
    rare_classes = [int(c) for c, f in enumerate(freqs) if 0 < f < cut]

    return rare_classes, freqs




def load_class_weights(weight_path, num_classes, device):
    if weight_path and os.path.isfile(weight_path):
        with open(weight_path, "r") as f:
            # Assumes your JSON file is a dictionary with keys "0", "1", ..., "num_classes-1"
            weights_dict = json.load(f)
        # Build list in order of class indices (convert keys to int if necessary)
        weights = [float(weights_dict.get(str(i), 1.0)) for i in range(num_classes)]
        weight_tensor = torch.tensor(weights, device=device)
        print("Loaded class weights:", weight_tensor)
    else:
        print("No valid class weight file provided; using default (None)")
        weight_tensor = None
    return weight_tensor


def compute_class_weights_from_data(train_loader, num_classes, device):
    """
    Compute class weights using median frequency balancing from the training data loader.
    It handles empty labels by assigning a weight of 0.0 to classes not present in the data.

    Args:
        train_loader (DataLoader): PyTorch DataLoader that provides training samples as (image, label).
        num_classes (int): Number of classes.
        device (torch.device): Device for constructing the weight tensor.

    Returns:
        torch.Tensor: A tensor of class weights with shape [num_classes].
    """
    # Initialize an array to store the pixel counts for each class.
    class_counts = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0

    # Iterate over the training loader
    for i, data in enumerate(train_loader):
        # Assuming data is a tuple (image, label)
        _, label = data
        # Ensure label is on CPU and convert to a numpy array.
        labels_np = label.cpu().numpy()
        # Count pixels for each class
        for cls in range(num_classes):
            class_counts[cls] += np.sum(labels_np == cls)
        total_pixels += np.prod(labels_np.shape)

    # Calculate the frequency of each class (proportion of pixels).
    frequencies = class_counts / total_pixels if total_pixels > 0 else np.zeros_like(class_counts)

    # Get frequencies only for classes that are present.
    nonzero_frequencies = frequencies[frequencies > 0]
    if len(nonzero_frequencies) == 0:
        print("No class pixels found in the dataset.")
        # Return a tensor with all zeros.
        weight_tensor = torch.zeros(num_classes, dtype=torch.float32, device=device)
        return weight_tensor

    median_freq = np.median(nonzero_frequencies)

    # Compute weight for each class.
    weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if frequencies[i] > 0:
            weights[i] = median_freq / frequencies[i]
        else:
            # If a class does not exist in the dataset, set its weight to 0.0.
            weights[i] = 0.0

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Calculated class weights from data:", weight_tensor)
    return weight_tensor


def validate(val_loader, model, epoch, args, writer):
    loss_list = []
    dice_list = []
    iou_preds = []
    # dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    dice_score_fn = DiceScoreCoefficient(n_classes=args.num_classes, ignore_index=0)

    if args.gpu is not None:
        dice_score_fn = dice_score_fn.cuda(args.gpu)

    # if rebalance_weights is not None:
    #     dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False, rebalance_weights=rebalance_weights)
    # else:
    #     dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)


    model.eval()

    with torch.no_grad():
        for i, tup in enumerate(val_loader):
            # measure data loading time

            if args.gpu is not None:
                img = tup[0].float().cuda(args.gpu, non_blocking=True)
                label = tup[1].long().cuda(args.gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]
            b, c, h, w = img.shape

            # compute output
            mask, iou_pred = model(img)
            mask = mask.view(b, -1, h, w)
            iou_pred = iou_pred.squeeze().view(b, -1)
            iou_pred = torch.mean(iou_pred)
            iou_preds.append(iou_pred.item())
            pred_softmax = F.softmax(mask, dim=1)
            dice_score_fn.reset() 
            batch_dsc = dice_score_fn(pred_softmax, label.squeeze(1))  # self.ce_loss(pred, target.squeeze())
            batch_macro = torch.nanmean(batch_dsc).item()
            dice_list.append(batch_macro)

    dice_score_mean = np.mean(dice_list)
    print('Validating: Epoch: %2d DSC: %.4f IoU_pred: %.4f' % (epoch, dice_score_mean, np.mean(iou_preds)))
    writer.add_scalar("val_dsc", dice_score_mean, epoch)
    return dice_score_mean


def test(model, args):
    print('Test')
    join = os.path.join
    if not os.path.exists(join(args.save_dir, "infer")):
        os.mkdir(join(args.save_dir, "infer"))
    if not os.path.exists(join(args.save_dir, "label")):
        os.mkdir(join(args.save_dir, "label"))

    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    test_keys = splits[args.fold]['test']

    model.eval()

    for key in test_keys:
        preds = []
        labels = []
        data_loader = generate_test_loader(key, args)
        with torch.no_grad():
            
            for i, tup in enumerate(data_loader):
                if args.gpu is not None:
                    img = tup[0].float().cuda(args.gpu, non_blocking=True)
                    label = tup[1].long().cuda(args.gpu, non_blocking=True)
                else:
                    img = tup[0]
                    label = tup[1]

                b, c, h, w = img.shape

                mask, iou_pred = model(img)
                mask = mask.view(b, -1, h, w)
                mask_softmax = F.softmax(mask, dim=1)
                mask = torch.argmax(mask_softmax, dim=1)

                preds.append(mask.cpu().numpy())
                labels.append(label.cpu().numpy())

            preds = np.concatenate(preds, axis=0)  # shape: (N, H, W)
            labels = np.concatenate(labels, axis=0)

            if labels.ndim == 4:
                labels = labels[:, 0, :, :]

            if "." in key:
                key = key.split(".")[0]

            # Special case: if only 1 slice
            if preds.shape[0] == 1:
                pred_img = Image.fromarray(preds[0].astype(np.uint8), mode='L')
                label_img = Image.fromarray(labels[0].astype(np.uint8), mode='L')

                pred_img.save(join(args.save_dir, "infer", f"{key}.png"))
                label_img.save(join(args.save_dir, "label", f"{key}.png"))
            else:
                # Save as PNGs (one file per slice)
                for idx in range(preds.shape[0]):
                    pred_img = Image.fromarray(preds[idx].astype(np.uint8), mode='L')
                    label_img = Image.fromarray(labels[idx].astype(np.uint8), mode='L')

                    pred_img.save(join(join(args.save_dir, "infer"), f"{key}_num{idx:02d}.png"))
                    label_img.save(join(join(args.save_dir, "label"), f"{key}_num{idx:02d}.png"))


        print("Finished saving PNGs for:", key)

            


def test_2(data_loader, model, args):
    print('Test')
    metric_val = SegmentationMetric(args.num_classes)
    metric_val.reset()
    model.eval()

    with torch.no_grad():
        for i, tup in enumerate(data_loader):
            # measure data loading time

            if args.gpu is not None:
                img = tup[0].float().cuda(args.gpu, non_blocking=True)
                label = tup[1].long().cuda(args.gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]

            b, c, h, w = img.shape

            # compute output
            mask, iou_pred = model(img)
            mask = mask.view(b, -1, h, w)

            pred_softmax = F.softmax(mask, dim=1)
            metric_val.update(label.squeeze(dim=1), pred_softmax)
            pixAcc, mIoU, Dice = metric_val.get()

            if i % args.print_freq == 0:
                print("Index:%f, mean Dice:%.4f" % (i, Dice))

    _, _, Dice = metric_val.get()
    print("Overall mean dice score is:", Dice)
    print("Finished test")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        # shutil.copyfile(filename, 'model_best.pth.tar')



def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

    # python main_moco.py --data_dir ./data/mmwhs/ --do_contrast --dist-url 'tcp://localhost:10001'
    # --multiprocessing-distributed --world-size 1 --rank 0
