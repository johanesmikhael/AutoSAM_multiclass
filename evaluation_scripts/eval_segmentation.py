import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from PIL import Image

from models import sam_seg_model_registry
from dataset import generate_test_loader
from evaluate import test_material

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    # Evaluation parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to the saved checkpoint')
    parser.add_argument('--model_type', type=str, default="vit_l",
                        help='model type: vit_h, vit_l, or vit_b')
    parser.add_argument('--num_classes', type=int, default=14,
                        help='number of output classes')
    parser.add_argument('--src_dir', type=str, required=True,
                        help='directory containing splits.pkl')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to data folder if required by the loader')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='directory to save evaluation results (infer and label images)')
    parser.add_argument('--fold', type=int, default=0,
                        help='fold number to use from splits.pkl')
    parser.add_argument('--img_size', type=int, default=256,
                        help='image size (if needed by generate_test_loader)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use, if any')
    parser.add_argument("--dataset", type=str, default="ACDC")
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
    return parser.parse_args()

def evaluate_model(args):
    # Set default model checkpoint for the image encoder based on model type.
    if args.model_type == 'vit_h':
        base_checkpoint = 'cp/sam_vit_h_4b8939.pth'
    elif args.model_type == 'vit_l':
        base_checkpoint = 'cp/sam_vit_l_0b3195.pth'
    elif args.model_type == 'vit_b':
        base_checkpoint = 'cp/sam_vit_b_01ec64.pth'
    else:
        raise ValueError("Unsupported model type: {}".format(args.model_type))
    
    # Create model using the registry.
    model = sam_seg_model_registry[args.model_type](num_classes=args.num_classes,
                                                    checkpoint=base_checkpoint)
    
    # Load the saved checkpoint (for the mask_decoder).
    if os.path.isfile(args.checkpoint):
        if args.gpu is None:
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.checkpoint, map_location=loc)
        # The checkpoint stores the mask_decoder's state_dict.
        if hasattr(model, 'module'):
            model.module.mask_decoder.load_state_dict(checkpoint['state_dict'])
        else:
            model.mask_decoder.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(args.checkpoint))
        return

    # If GPU is specified, move the model to GPU.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    
    model.eval()

    # Prepare directories for saving predictions and labels.
    infer_dir = os.path.join(args.save_dir, "infer")
    label_dir = os.path.join(args.save_dir, "label")
    os.makedirs(infer_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Load splits from the provided src_dir.
    split_path = os.path.join(args.src_dir, "splits.pkl")
    with open(split_path, "rb") as f:
        splits = pickle.load(f)
    test_keys = splits[args.fold]['test']

    # Iterate over test keys.
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
                    img = tup[0].float()
                    label = tup[1].long()
                b, c, h, w = img.shape
                mask, _ = model(img)
                mask = mask.view(b, -1, h, w)
                mask_softmax = F.softmax(mask, dim=1)
                mask_pred = torch.argmax(mask_softmax, dim=1)
                preds.append(mask_pred.cpu().numpy())
                labels.append(label.cpu().numpy())
        preds = np.concatenate(preds, axis=0)  # shape: (N, H, W)
        labels = np.concatenate(labels, axis=0)
        if labels.ndim == 4:
            labels = labels[:, 0, :, :]

        key_name = key.split(".")[0] if "." in key else key

        # Save prediction and label images.
        if preds.shape[0] == 1:
            pred_img = Image.fromarray(preds[0].astype(np.uint8), mode='L')
            label_img = Image.fromarray(labels[0].astype(np.uint8), mode='L')
            pred_img.save(os.path.join(infer_dir, f"{key_name}.png"))
            label_img.save(os.path.join(label_dir, f"{key_name}.png"))
        else:
            for idx in range(preds.shape[0]):
                pred_img = Image.fromarray(preds[idx].astype(np.uint8), mode='L')
                label_img = Image.fromarray(labels[idx].astype(np.uint8), mode='L')
                pred_img.save(os.path.join(infer_dir, f"{key_name}_num{idx:02d}.png"))
                label_img.save(os.path.join(label_dir, f"{key_name}_num{idx:02d}.png"))
        print("Finished saving PNGs for:", key)

def main():
    args = parse_args()
    args.distributed = False
    evaluate_model(args)

    if args.dataset == 'material':
        test_material(args)


if __name__ == '__main__':
    print('eval')
    main()