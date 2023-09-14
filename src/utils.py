import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import os


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_image_mask(image, mask):
    fig, axes = plt.subplots()
    axes.imshow(np.array(image))
    ground_truth_seg = np.array(mask)
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"Ground truth mask")
    axes.axis("off")
    plt.savefig("./plots/gt_mask.jpg")
    

def plot_image_mask_dataset(dataset, idx):
    image_path = dataset.img_files[idx]
    mask_path = dataset.mask_files[idx]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    plot_image_mask(image, mask)



def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  idx = np.where(ground_truth_map > 0)
  x_indices = idx[1]
  y_indices = idx[0]
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


def get_list_masks(batch, preds):
    list_gt_msk = []
    list_pred_msk = []
    for k in range (len(batch)):
        list_gt_msk.append(batch[k]["ground_truth_mask"])
        list_pred_msk.append(preds[k]["masks"].squeeze(0).squeeze(0))
        print("GT :", list_gt_msk[-1].shape)
        print("PRED :", list_pred_msk[-1].shape)
    return list_gt_msk, list_pred_msk


def get_max_size(batch):
    max_size_w = 0
    max_size_h = 0
    for elt in batch:
        if elt["ground_truth_mask"].shape[1] > max_size_w:
            max_size_w =  elt["ground_truth_mask"].shape[1]
        if elt["ground_truth_mask"].shape[0] > max_size_h:
            max_size_h =  elt["ground_truth_mask"].shape[0]
            
    return max_size_h, max_size_w


def pad_batch_mask(list_gt_msk, list_pred_msk, max_h, max_w):
    for k in range(len(list_gt_msk)):
        
        list_gt_msk[k] = pad(list_gt_msk[k], pad=(max_w-list_gt_msk[k].shape[1], 0, 0, max_h-list_gt_msk[k].shape[0]))
        list_pred_msk[k] =  pad(list_pred_msk[k], pad=(max_w-list_pred_msk[k].shape[1], 0, 0, max_h-list_pred_msk[k].shape[0]))

    stk_gt_msk = torch.stack(list_gt_msk, dim=0)
    stk_pred_msk = torch.stack(list_pred_msk, dim=0)

    print(stk_gt_msk.shape, stk_pred_msk.shape)
    return stk_gt_msk, stk_pred_msk



def tensor_to_image(gt_masks, pred_msks):
    f, axarr = plt.subplots(2,2)
    for i, (gt_msk, pred_msk) in enumerate(zip(gt_masks, pred_msks)):
        axarr[0, i].imshow(gt_msk[:, :])
        axarr[1, i].imshow(pred_msk[:, :])
    plt.savefig("./plots/comparaison.png")