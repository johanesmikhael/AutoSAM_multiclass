import glob
import pickle

import os
import SimpleITK as sitk
import numpy as np
import argparse
# from medpy import metric
import argparse
from PIL import Image

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return np.nan
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
    

def iou(pred, label):
    intersection = np.logical_and(pred, label).sum()
    union = np.logical_or(pred, label).sum()
    if union == 0:
        # If both the prediction and label are empty, return nan.
        return np.nan
    else:
        return intersection / union


# def hd(pred, gt):
#     if pred.sum() > 0 and gt.sum() > 0:
#         hd95 = metric.binary.hd95(pred, gt)
#         return hd95
#     else:
#         return 0


def test_brats(args):
    label_list = sorted(glob.glob(os.path.join(args.save_dir, 'label', '*nii')))
    infer_list = sorted(glob.glob(os.path.join(args.save_dir, 'infer', '*nii')))
    print("loading success...")
    Dice_et = []
    Dice_tc = []
    Dice_wt = []

    # HD_et = []
    # HD_tc = []
    # HD_wt = []

    def process_label(label):
        net = label == 2
        ed = label == 1
        et = label == 3
        ET = et
        TC = net + et
        WT = net + et + ed
        return ET, TC, WT

    fw = open(args.save_dir + '/dice_pre.txt', 'a')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_et, label_tc, label_wt = process_label(label)
        infer_et, infer_tc, infer_wt = process_label(infer)
        Dice_et.append(dice(infer_et, label_et))
        Dice_tc.append(dice(infer_tc, label_tc))
        Dice_wt.append(dice(infer_wt, label_wt))

        # HD_et.append(hd(infer_et, label_et))
        # HD_tc.append(hd(infer_tc, label_tc))
        # HD_wt.append(hd(infer_wt, label_wt))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        # fw.write('hd_et: {:.4f}\n'.format(HD_et[-1]))
        # fw.write('hd_tc: {:.4f}\n'.format(HD_tc[-1]))
        # fw.write('hd_wt: {:.4f}\n'.format(HD_wt[-1]))
        fw.write('*' * 20 + '\n', )
        fw.write('Dice_et: {:.4f}\n'.format(Dice_et[-1]))
        fw.write('Dice_tc: {:.4f}\n'.format(Dice_tc[-1]))
        fw.write('Dice_wt: {:.4f}\n'.format(Dice_wt[-1]))

        # print('dice_et: {:.4f}'.format(np.mean(Dice_et)))
        # print('dice_tc: {:.4f}'.format(np.mean(Dice_tc)))
        # print('dice_wt: {:.4f}'.format(np.mean(Dice_wt)))
    dsc = []
    avg_hd = []
    dsc.append(np.mean(Dice_et))
    dsc.append(np.mean(Dice_tc))
    dsc.append(np.mean(Dice_wt))

    # avg_hd.append(np.mean(HD_et))
    # avg_hd.append(np.mean(HD_tc))
    # avg_hd.append(np.mean(HD_wt))

    fw.write('Dice_et' + str(np.mean(Dice_et)) + ' ' + '\n')
    fw.write('Dice_tc' + str(np.mean(Dice_tc)) + ' ' + '\n')
    fw.write('Dice_wt' + str(np.mean(Dice_wt)) + ' ' + '\n')

    # fw.write('HD_et' + str(np.mean(HD_et)) + ' ' + '\n')
    # fw.write('HD_tc' + str(np.mean(HD_tc)) + ' ' + '\n')
    # fw.write('HD_wt' + str(np.mean(HD_wt)) + ' ' + '\n')

    fw.write('Dice' + str(np.mean(dsc)) + ' ' + '\n')
    # fw.write('HD' + str(np.mean(avg_hd)) + ' ' + '\n')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            print(line)


def test_acdc(args):
    label_list = sorted(glob.glob(os.path.join(args.save_dir, 'label', '*nii')))
    infer_list = sorted(glob.glob(os.path.join(args.save_dir, 'infer', '*nii')))

    Dice_rv = []
    Dice_myo = []
    Dice_lv = []

    # hd_rv = []
    # hd_myo = []
    # hd_lv = []

    def process_label(label):
        rv = label == 1
        myo = label == 2
        lv = label == 3

        return rv, myo, lv

    fw = open(args.save_dir + '/dice_pre.txt', 'a')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label = read_nii(label_path)
        infer = read_nii(infer_path)
        label_rv, label_myo, label_lv = process_label(label)
        infer_rv, infer_myo, infer_lv = process_label(infer)

        Dice_rv.append(dice(infer_rv, label_rv))
        Dice_myo.append(dice(infer_myo, label_myo))
        Dice_lv.append(dice(infer_lv, label_lv))

        # hd_rv.append(hd(infer_rv, label_rv))
        # hd_myo.append(hd(infer_myo, label_myo))
        # hd_lv.append(hd(infer_lv, label_lv))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        # fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        # fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        # fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        # fw.write('*'*20+'\n')
        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_rv: {:.4f}\n'.format(Dice_rv[-1]))
        fw.write('Dice_myo: {:.4f}\n'.format(Dice_myo[-1]))
        fw.write('Dice_lv: {:.4f}\n'.format(Dice_lv[-1]))
        # fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        # fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        # fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        fw.write('*' * 20 + '\n')

    # fw.write('*'*20+'\n')
    # fw.write('Mean_hd\n')
    # fw.write('hd_rv'+str(np.mean(hd_rv))+'\n')
    # fw.write('hd_myo'+str(np.mean(hd_myo))+'\n')
    # fw.write('hd_lv'+str(np.mean(hd_lv))+'\n')
    # fw.write('*'*20+'\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_rv' + str(np.mean(Dice_rv)) + '\n')
    fw.write('Dice_myo' + str(np.mean(Dice_myo)) + '\n')
    fw.write('Dice_lv' + str(np.mean(Dice_lv)) + '\n')
    # fw.write('Mean_HD\n')
    # fw.write('HD_rv' + str(np.mean(hd_rv)) + '\n')
    # fw.write('HD_myo' + str(np.mean(hd_myo)) + '\n')
    # fw.write('HD_lv' + str(np.mean(hd_lv)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_rv))
    dsc.append(np.mean(Dice_myo))
    dsc.append(np.mean(Dice_lv))
    # avg_hd = []
    # avg_hd.append(np.mean(hd_rv))
    # avg_hd.append(np.mean(hd_myo))
    # avg_hd.append(np.mean(hd_lv))
    # fw.write('avg_hd:' + str(np.mean(avg_hd)) + '\n')

    fw.write('DSC:' + str(np.mean(dsc)) + '\n')
    # fw.write('HD:' + str(np.mean(avg_hd)) + '\n')

    print('done')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            print(line)


def test_synapse(args):
    label_list = sorted(glob.glob(os.path.join(args.save_dir, 'label', '*nii')))
    infer_list = sorted(glob.glob(os.path.join(args.save_dir, 'infer', '*nii')))
    Dice_spleen = []
    Dice_right_kidney = []
    Dice_left_kidney = []
    Dice_gallbladder = []
    Dice_liver = []
    Dice_stomach = []
    Dice_aorta = []
    Dice_pancreas = []

    # hd_spleen = []
    # hd_right_kidney = []
    # hd_left_kidney = []
    # hd_gallbladder = []
    # hd_liver = []
    # hd_stomach = []
    # hd_aorta = []
    # hd_pancreas = []

    def process_label(label):
        spleen = label == 1
        right_kidney = label == 2
        left_kidney = label == 3
        gallbladder = label == 4
        liver = label == 6
        stomach = label == 7
        aorta = label == 8
        pancreas = label == 11

        return spleen, right_kidney, left_kidney, gallbladder, liver, stomach, aorta, pancreas

    fw = open(args.save_dir + '/dice_pre.txt', 'a')
    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_spleen, label_right_kidney, label_left_kidney, label_gallbladder, label_liver, \
        label_stomach, label_aorta, label_pancreas = process_label(
            label)
        infer_spleen, infer_right_kidney, infer_left_kidney, infer_gallbladder, infer_liver, \
        infer_stomach, infer_aorta, infer_pancreas = process_label(
            infer)

        Dice_spleen.append(dice(infer_spleen, label_spleen))
        Dice_right_kidney.append(dice(infer_right_kidney, label_right_kidney))
        Dice_left_kidney.append(dice(infer_left_kidney, label_left_kidney))
        Dice_gallbladder.append(dice(infer_gallbladder, label_gallbladder))
        Dice_liver.append(dice(infer_liver, label_liver))
        Dice_stomach.append(dice(infer_stomach, label_stomach))
        Dice_aorta.append(dice(infer_aorta, label_aorta))
        Dice_pancreas.append(dice(infer_pancreas, label_pancreas))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
        fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
        fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
        fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
        fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
        fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))

        # hd_spleen.append(hd(infer_spleen, label_spleen))
        # hd_right_kidney.append(hd(infer_right_kidney, label_right_kidney))
        # hd_left_kidney.append(hd(infer_left_kidney, label_left_kidney))
        # hd_gallbladder.append(hd(infer_gallbladder, label_gallbladder))
        # hd_liver.append(hd(infer_liver, label_liver))
        # hd_stomach.append(hd(infer_stomach, label_stomach))
        # hd_aorta.append(hd(infer_aorta, label_aorta))
        # hd_pancreas.append(hd(infer_pancreas, label_pancreas))

        # fw.write('hd_spleen: {:.4f}\n'.format(hd_spleen[-1]))
        # fw.write('hd_right_kidney: {:.4f}\n'.format(hd_right_kidney[-1]))
        # fw.write('hd_left_kidney: {:.4f}\n'.format(hd_left_kidney[-1]))
        # fw.write('hd_gallbladder: {:.4f}\n'.format(hd_gallbladder[-1]))
        # fw.write('hd_liver: {:.4f}\n'.format(hd_liver[-1]))
        # fw.write('hd_stomach: {:.4f}\n'.format(hd_stomach[-1]))
        # fw.write('hd_aorta: {:.4f}\n'.format(hd_aorta[-1]))
        # fw.write('hd_pancreas: {:.4f}\n'.format(hd_pancreas[-1]))

        dsc = []
        # HD = []
        dsc.append(Dice_spleen[-1])
        dsc.append((Dice_right_kidney[-1]))
        dsc.append(Dice_left_kidney[-1])
        dsc.append(np.mean(Dice_gallbladder[-1]))
        dsc.append(np.mean(Dice_liver[-1]))
        dsc.append(np.mean(Dice_stomach[-1]))
        dsc.append(np.mean(Dice_aorta[-1]))
        dsc.append(np.mean(Dice_pancreas[-1]))
        fw.write('DSC:' + str(np.mean(dsc)) + '\n')

        # HD.append(hd_spleen[-1])
        # HD.append(hd_right_kidney[-1])
        # HD.append(hd_left_kidney[-1])
        # HD.append(hd_gallbladder[-1])
        # HD.append(hd_liver[-1])
        # HD.append(hd_stomach[-1])
        # HD.append(hd_aorta[-1])
        # HD.append(hd_pancreas[-1])
        # fw.write('hd:' + str(np.mean(HD)) + '\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_spleen' + str(np.mean(Dice_spleen)) + '\n')
    fw.write('Dice_right_kidney' + str(np.mean(Dice_right_kidney)) + '\n')
    fw.write('Dice_left_kidney' + str(np.mean(Dice_left_kidney)) + '\n')
    fw.write('Dice_gallbladder' + str(np.mean(Dice_gallbladder)) + '\n')
    fw.write('Dice_liver' + str(np.mean(Dice_liver)) + '\n')
    fw.write('Dice_stomach' + str(np.mean(Dice_stomach)) + '\n')
    fw.write('Dice_aorta' + str(np.mean(Dice_aorta)) + '\n')
    fw.write('Dice_pancreas' + str(np.mean(Dice_pancreas)) + '\n')

    # fw.write('Mean_hd\n')
    # fw.write('hd_spleen' + str(np.mean(hd_spleen)) + '\n')
    # fw.write('hd_right_kidney' + str(np.mean(hd_right_kidney)) + '\n')
    # fw.write('hd_left_kidney' + str(np.mean(hd_left_kidney)) + '\n')
    # fw.write('hd_gallbladder' + str(np.mean(hd_gallbladder)) + '\n')
    # fw.write('hd_liver' + str(np.mean(hd_liver)) + '\n')
    # fw.write('hd_stomach' + str(np.mean(hd_stomach)) + '\n')
    # fw.write('hd_aorta' + str(np.mean(hd_aorta)) + '\n')
    # fw.write('hd_pancreas' + str(np.mean(hd_pancreas)) + '\n')

    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_spleen))
    dsc.append(np.mean(Dice_right_kidney))
    dsc.append(np.mean(Dice_left_kidney))
    dsc.append(np.mean(Dice_gallbladder))
    dsc.append(np.mean(Dice_liver))
    dsc.append(np.mean(Dice_stomach))
    dsc.append(np.mean(Dice_aorta))
    dsc.append(np.mean(Dice_pancreas))
    fw.write('dsc:' + str(np.mean(dsc)) + '\n')

    # HD = []
    # HD.append(np.mean(hd_spleen))
    # HD.append(np.mean(hd_right_kidney))
    # HD.append(np.mean(hd_left_kidney))
    # HD.append(np.mean(hd_gallbladder))
    # HD.append(np.mean(hd_liver))
    # HD.append(np.mean(hd_stomach))
    # HD.append(np.mean(hd_aorta))
    # HD.append(np.mean(hd_pancreas))
    # fw.write('hd:' + str(np.mean(HD)) + '\n')

    print('done')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            print(line)


def test_material(args):
    # Get sorted lists of PNG files for labels and inferences.
    label_list = sorted(glob.glob(os.path.join(args.save_dir, 'label', '*png')))
    infer_list = sorted(glob.glob(os.path.join(args.save_dir, 'infer', '*png')))

    # Define label mapping for material dataset.
    label_names = {
        1: 'reinforced_concrete',
        2: 'unreinforced_concrete',
        3: 'precast_concrete',
        4: 'masonry',
        5: 'slit',
        6: 'xps_insulation',
        7: 'hard_insulation',
        8: 'soft_insulation'
    }
    
    # Initialize dictionary to store Dice scores per label.
    dice_scores = {label: [] for label in label_names.keys()}
    # Initialize dictionary to store IoU scores per label.
    iou_scores = {label: [] for label in label_names.keys()}

    dice_file = os.path.join(args.save_dir, 'dice_pre.txt')
    iou_file = os.path.join(args.save_dir, 'iou_pre.txt')

    # has_gt      = {label: False for label in label_names}   # track presence in GT


    with open(dice_file, 'w') as fw_dice, open(iou_file, 'w') as fw_iou:
        for label_path, infer_path in zip(label_list, infer_list):
            filename = os.path.basename(infer_path)
            print(filename)
            print(filename)
            
            # Read PNG images (assumed to be single-channel with integer label values)
            gt = np.array(Image.open(label_path))
            pred = np.array(Image.open(infer_path))

            
            
            fw_dice.write('*' * 20 + '\n')
            fw_dice.write(filename + '\n')
            fw_dice.write('*' * 20 + '\n')
            fw_dice.write(filename + '\n')
            
            fw_iou.write('*' * 20 + '\n')
            fw_iou.write(filename + '\n')
            fw_iou.write('*' * 20 + '\n')
            fw_iou.write(filename + '\n')
            
            # Process each label
            for label_val, label_name in label_names.items():
                gt_mask = (gt == label_val).astype(np.uint8)
                pred_mask = (pred == label_val).astype(np.uint8)

                # Only calculate Dice score if there is something to segment in the ground truth.
                # your dice() returns 1 if both empty, 0 if GT empty but pred non-empty
                d = dice(pred_mask, gt_mask)
                dice_scores[label_val].append(d)
                if np.isnan(d):
                    fw_dice.write(f"Dice_{label_name}: skipped (no GT & no pred)\n")
                else:
                    fw_dice.write(f"Dice_{label_name}: {d:.4f}\n")

                i = iou(pred_mask, gt_mask)
                iou_scores[label_val].append(i)
                if np.isnan(i):
                    fw_iou.write(f"IoU_{label_name}: skipped (no GT & no pred)\n")
                else:
                    fw_iou.write(f"IoU_{label_name}: {i:.4f}\n")
                # score = dice(pred_mask, gt_mask)
                # dice_scores[label_val].append(score)
                # fw.write(f"Dice_{label_name}: {score:.4f}\n")
            fw_dice.write('*' * 20 + '\n')
            fw_iou.write('*' * 20 + '\n')


        # Write mean Dice per class, skipping absent ones
        fw_dice.write('*' * 20 + '\n')
        fw_dice.write("Mean_Dice\n")

        # build a dict of per‐class means, using NaN for skipped classes
        per_class_mean = {}
        for label_val, label_name in label_names.items():
            per_class_mean[label_val] = np.nanmean(dice_scores[label_val])
            if np.isnan(per_class_mean[label_val]):
                fw_dice.write(f"Dice_{label_name}: skipped (no GT in dataset)\n")
            else:
                fw_dice.write(f"Dice_{label_name}: {per_class_mean[label_val]:.4f}\n")

        # compute the macro‐Dice by ignoring NaNs
        macro_dice = np.nanmean(list(per_class_mean.values()))
        fw_dice.write('*' * 20 + '\n')
        fw_dice.write(f"Macro_Dice skip absent class: {macro_dice:.4f}\n")
        fw_dice.write('*' * 20 + '\n')

        # Write mean IoU per class, skipping absent ones
        fw_iou.write('*' * 20 + '\n')
        fw_iou.write("Mean_IoU\n")

        # build a dict of per‐class means, using NaN for skipped classes
        per_class_iou = {}
        for label_val, label_name in label_names.items():
            per_class_iou[label_val] = np.nanmean(iou_scores[label_val])
            if np.isnan(per_class_iou[label_val]):
                fw_iou.write(f"IoU_{label_name}: skipped (no GT in dataset)\n")
            else:
                fw_iou.write(f"IoU_{label_name}: {per_class_iou[label_val]:.4f}\n")


        # compute the macro‐IoU by ignoring NaNs
        macro_iou = np.nanmean(list(per_class_iou.values()))
        fw_iou.write('*' * 20 + '\n')
        fw_iou.write(f"Macro_IoU skip absent class: {macro_iou:.4f}\n")
        fw_iou.write('*' * 20 + '\n')

    print("Dice Scores:")
    with open(dice_file, 'r') as f:
        for line in f.read().splitlines():
            print(line)
    print("\nIoU Scores:")
    with open(iou_file, 'r') as f:
        for line in f.read().splitlines():
            print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.save_dir = 'output_experiment/sam_unet_seg_ACDC_f0_tr_75'
    test_acdc(args)
