import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torch
from torch.nn.functional import pad


def show_mask(mask: np.array, ax, random_color=False):
    """
    Plot the mask

    Arguments:
        mask: Array of the binary mask (or float)
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_image_mask(image: PIL.Image, mask: PIL.Image, filename: str):
    """
    Plot the image and the mask superposed

    Arguments:
        image: PIL original image
        mask: PIL original binary mask
    """
    fig, axes = plt.subplots()
    axes.imshow(np.array(image))
    ground_truth_seg = np.array(mask)
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"{filename} predicted mask")
    axes.axis("off")
    plt.savefig("../plots/" + filename + ".jpg")
    

def plot_image_mask_dataset(dataset: torch.utils.data.Dataset, idx: int):
    """
    Take an image from the dataset and plot it

    Arguments:
        dataset: Dataset class loaded with our images
        idx: Index of the data we want
    """
    image_path = dataset.img_files[idx]
    mask_path = dataset.mask_files[idx]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    plot_image_mask(image, mask)


def get_bounding_box(ground_truth_map: np.array) -> list:
  """
  Get the bounding box of the image with the ground truth mask
  
    Arguments:
        ground_truth_map: Take ground truth mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]

  """
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


def get_list_masks(batch: torch.utils.data, preds: list) -> list:
    """
    Take the batch (list(dict)) return a list of preprocessed image tensor, predicted masks tensor and bounding box lists

    Arguments:
        batch: batch (list(dict)) loaded
        preds: list(dict) that are predicted masks of the batch

    Return:
        list_gt_msk: List of ground truth tensors mask tensors of the batch
        list_pred_msk: List of predicted tensors mask tensors from the batch
        list_bbox: List of tensors bounding tensorsboxes from the batch
    """
    list_gt_msk = []
    list_pred_msk = []
    list_bbox = []
    for k in range (len(batch)):
        list_bbox.append(batch[k]["prompt"])
        list_gt_msk.append(batch[k]["ground_truth_mask"])
        list_pred_msk.append(preds[k]["masks"].squeeze(0).squeeze(0))

    return list_gt_msk, list_pred_msk, list_bbox


def get_max_size(batch: torch.utils.data) -> int:
    """
    Get the max size (height and width) between all the images in the batch

    Arguments:
        batch: Batch from dataloader

    Return:
        max_size_h: Maximum height compared from all preprocessed images in the batch
        max_size_w: Maximum width compared from all preprocessed images in the batch
    """
    max_size_w = 0
    max_size_h = 0
    for elt in batch:
        if elt["ground_truth_mask"].shape[1] > max_size_w:
            max_size_w =  elt["ground_truth_mask"].shape[1]
        if elt["ground_truth_mask"].shape[0] > max_size_h:
            max_size_h =  elt["ground_truth_mask"].shape[0]
            
    return max_size_h, max_size_w


def pad_batch_mask(list_gt_msk: list, list_pred_msk: list, max_h: int, max_w: int) -> list:
    """
    Pad all mask tensors ground truth and prediction from sam in the max size of the batch. This is because the monai diceloss requires same size tensors for batches.
    By 0-padding binary mask, we don't loose change the loss function

    Arguments:
        list_gt_msk: List of ground truth masks tensors
        list_pred_msk: List of tensors of predicted masks from SAM 
        max_h: Maximum height compared from all preprocessed images in the batch
        max_w: Maximum width compared from all preprocessed images in the batch
    """
    for k in range(len(list_gt_msk)):
        
        list_gt_msk[k] = pad(list_gt_msk[k], pad=(max_w-list_gt_msk[k].shape[1], 0, 0, max_h-list_gt_msk[k].shape[0]))
        list_pred_msk[k] =  pad(list_pred_msk[k], pad=(max_w-list_pred_msk[k].shape[1], 0, 0, max_h-list_pred_msk[k].shape[0]))

    stk_gt_msk = torch.stack(list_gt_msk, dim=0)
    stk_pred_msk = torch.stack(list_pred_msk, dim=0)

    return stk_gt_msk, stk_pred_msk



def tensor_to_image(gt_masks: list, pred_msks: list, bboxes: list):
    """
    Get tensors of ground truth masks and predicted masks from SAM and plot them to compare

    Arguments:
       gt_masks: List of ground truth masks as tensors
       pred_msks: List of predicted masks as tensors
       bboxes: list of bounding boxes
    """
    f, axarr = plt.subplots(2,2)
    for i, (gt_msk, pred_msk, bbox) in enumerate(zip(gt_masks, pred_msks, bboxes)):
        axarr[1, i].scatter([bbox[0], bbox[2]], [bbox[1], bbox[3]])
        axarr[0, i].imshow(gt_msk[:, :])
        axarr[1, i].imshow(pred_msk[:, :])
        axarr[0, i].set_title('Original Mask', fontdict  = {"fontsize": 8})
        axarr[1, i].set_title('Predicted Mask', fontdict  = {"fontsize": 8})
    plt.savefig("../plots/comparaison.png")

