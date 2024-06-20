import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex
from torchmetrics.classification import Dice # doesn't work with average=None so using F1 instead
from torchmetrics.functional.classification import multiclass_f1_score
from sklearn.metrics import confusion_matrix
from medpy.metric.binary import hd, hd95, assd
import matplotlib.pyplot as plt

from unet.unet_model import UNet
from unet.stochastic_unet import StochasticUNet
from deepmedic.stochastic_deepmedic import StochasticDeepMedic
from utils.get_dataloaders import get_train_loaders, get_test_loader
from mc_loss import SSNLossMCIntegral

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from inference import undo_normalisation, majority_voting, postprocess
from utils.utils import crop_mask


def evaluate_deterministic(model, device, dataloader, num_classes, undo_norm, confusion_matrix=False, filename=None):
    """ Evaluate model on the validation set using Dice and cross-entropy loss

    Args:
        model: trained model
        device: device to run the model on
        dataloader: DataLoader for validation set
        num_classes: number of classes
        eval_ce: whether to evaluate cross-entropy loss

    Returns:
        mean_dice: (macro) average Dice score
        dice_scores_per_class: Numpy array containing the Dice score for each class
        mean_ce_loss: average cross-entropy loss
    """
    # print(f"Length of dataloader: {len(dataloader)}")
    model_name = model.__class__.__name__

    model.eval()

    ############## Metrics ###################
    # Using F1 score to replace macro averaged Dice score as Dice doesn't support average=None
    dice_metric_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    dice_metric_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    dice_metric_micro = MulticlassF1Score(num_classes=num_classes, average='micro').to(device)

    precision_metric_macro = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    precision_metric_micro = MulticlassPrecision(num_classes=num_classes, average='micro').to(device)

    recall_metric_macro = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    recall_metric_micro = MulticlassRecall(num_classes=num_classes, average='micro').to(device)

    iou_metric_macro = MulticlassJaccardIndex(num_classes=num_classes, average='macro').to(device)
    iou_metric_micro = MulticlassJaccardIndex(num_classes=num_classes, average='micro').to(device)
    iou_metric_per_class = MulticlassJaccardIndex(num_classes=num_classes, average=None).to(device)
    
    cross_entropy_loss = nn.CrossEntropyLoss()
    total_ce_loss = 0

    hds = []
    hd95s = []

    all_preds = []
    all_masks = []
    
    ############ Evaluation ##############
    with torch.no_grad():
        for img, mask, _ in dataloader:
            if undo_norm:
                img = undo_normalisation(img.squeeze(0)).unsqueeze(0) # Undo normalisation (assuming batch size is 1)
            
            img = img.to(device) # (B, C, H, W)
            mask = mask.to(device) # (B, H, W)

            preds = model(img) # (B, num_classes, H, W)
            # preds_argmax = torch.argmax(preds, dim=1) # (B, H, W)

            if model_name == 'DeepMedic':
                mask = crop_mask(preds, mask)

            preds = postprocess(preds.argmax(dim=1).squeeze(0).cpu().numpy())
            preds = torch.from_numpy(preds).unsqueeze(0).to(device)

            # Compute Dice metrics
            dice_metric_macro(preds, mask)
            dice_metric_micro(preds, mask)
            dice_metric_per_class(preds, mask)
            # Compute precision and recall
            precision_metric_macro(preds, mask)
            precision_metric_micro(preds, mask)
            recall_metric_macro(preds, mask)
            recall_metric_micro(preds, mask)
            # Compute mean IoU
            iou_metric_macro(preds, mask)
            iou_metric_micro(preds, mask)
            iou_metric_per_class(preds, mask)
            # Compute cross-entropy loss
            # total_ce_loss += cross_entropy_loss(preds, mask).item()
            # Compute distance metrics
            # preds_hd = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy() # Assuming batch size is 1
            preds_hd = preds.squeeze(0).cpu().numpy() # Assuming batch size is 1
            mask_hd = mask.squeeze(0).cpu().numpy()
            hds.append(hd(preds_hd, mask_hd))
            hd95s.append(hd95(preds_hd, mask_hd))

            # Get predictions and masks for confusion matrix
            # all_preds.append(torch.argmax(preds, dim=1).cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    ########### Compute metrics ##############
    mean_dice_macro = dice_metric_macro.compute().item()
    mean_dice_micro = dice_metric_micro.compute().item()
    dice_scores_per_class = dice_metric_per_class.compute().cpu().numpy()
    precision_macro = precision_metric_macro.compute().item()
    precision_micro = precision_metric_micro.compute().item()
    recall_macro = recall_metric_macro.compute().item()
    recall_micro = recall_metric_micro.compute().item()
    iou_macro = iou_metric_macro.compute().item()
    iou_micro = iou_metric_micro.compute().item()
    iou_per_class = iou_metric_per_class.compute().cpu().numpy()
    mean_ce_loss = total_ce_loss / len(dataloader)
    mean_hd = np.mean(hds)
    mean_hd95 = np.mean(hd95s)

    if confusion_matrix:
        plot_confusion_matrix(np.concatenate(all_masks).flatten(), np.concatenate(all_preds).flatten(), num_classes, filename)

    ret = {
        'mean_dice_macro': mean_dice_macro,
        'mean_dice_micro': mean_dice_micro,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'iou_macro': iou_macro,
        'iou_micro': iou_micro,
        'mean_ce_loss': mean_ce_loss,
        'mean_hd': mean_hd,
        'mean_hd95': mean_hd95,
        'dice_scores_per_class': dice_scores_per_class,
        'iou_per_class': iou_per_class,
    }

    # add invididual class scores
    # for i in range(num_classes):
    #     ret[f'dice_only_class_{i}'] = dice_scores_per_class[i]
    #     ret[f'iou_only_class_{i}'] = iou_per_class[i]

    return ret


def evaluate_stochastic(model, device, dataloader, num_classes, edge_weight=1, confusion_matrix=False, filename=None):
    """ Evaluate model on the validation set using SSNLossMCIntegral and Dice loss

    Args:
        model: trained model
        device: device to run the model on
        dataloader: DataLoader for validation set
        num_classes: number of classes
        edge_weight: weight for the edge map
        confusion_matrix: whether to plot the confusion matrix

    Returns:
        mean_dice: (macro) average Dice score
        dice_scores_per_class: Numpy array containing the Dice score for each class
        mean_ssn_loss: average SSNLossMCIntegral
        mean_ce_loss: average cross-entropy loss
    """
    # print(f"Length of dataloader: {len(dataloader)}")
    model_name = model.__class__.__name__

    model.eval()
    
    ############## Metrics ###################
    dice_metric_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    dice_metric_micro = MulticlassF1Score(num_classes=num_classes, average='micro').to(device)
    dice_metric_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    precision_metric_macro = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    precision_metric_micro = MulticlassPrecision(num_classes=num_classes, average='micro').to(device)

    recall_metric_macro = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    recall_metric_micro = MulticlassRecall(num_classes=num_classes, average='micro').to(device)

    iou_metric_macro = MulticlassJaccardIndex(num_classes=num_classes, average='macro').to(device)
    iou_metric_micro = MulticlassJaccardIndex(num_classes=num_classes, average='micro').to(device)
    iou_metric_per_class = MulticlassJaccardIndex(num_classes=num_classes, average=None).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    total_ce_loss = 0

    ssn_loss_fn = SSNLossMCIntegral(num_mc_samples=20)
    total_ssn_loss = 0

    hds = []
    hd95s = []

    total = len(dataloader)
    better = 0

    all_preds = []
    all_masks = []

    ############ Evaluation ##############
    with torch.no_grad():
        for img, mask, edge_map in dataloader:
            img = img.to(device) # (B, C, H, W)
            mask = mask.to(device) # (B, H, W)
            edge_map = edge_map.to(device)
            
            logits, output_dict = model(img) # (B, num_classes, H, W)
            dist = output_dict['distribution']

            if model_name == 'StochasticDeepMedic':
                mask = crop_mask(logits, mask)

            logits = postprocess(logits.argmax(dim=1).squeeze(0).cpu().numpy())
            logits_dice = multiclass_f1_score(torch.from_numpy(logits), mask.squeeze(0).cpu(), num_classes=5, average='macro').item()
            # logits = torch.from_numpy(logits).unsqueeze(0).to(device)

            samples_uncertainty = dist.rsample((5,)).squeeze(1).cpu() # (20, num_classes, H, W)
            samples_uncertainty = samples_uncertainty.argmax(dim=1).numpy() # shape = (num_samples, H, W)

            post_processed_samples = []
            for sample in samples_uncertainty:
                post_processed_samples.append(postprocess(sample))

            # logits = majority_voting_cuda(samples_uncertainty.argmax(dim=1)) # (H, W)
            # logits = postprocess(logits.cpu().numpy()) # (H, W)
            # logits = torch.from_numpy(logits).unsqueeze(0).to(device) # (1, H, W)

            # find the best sample out of the 5
            best_sample = None
            best_dice = 0
            for sample in post_processed_samples:
                dice = multiclass_f1_score(torch.from_numpy(sample), mask.squeeze(0).cpu(), num_classes=5, average='macro').item()

                if dice > logits_dice:
                    better += 1

                if dice > best_dice:
                    best_dice = dice
                    best_sample = sample

            logits = torch.from_numpy(best_sample).unsqueeze(0).to(device)

            # Compute Dice metrics
            dice_metric_macro(logits, mask)
            dice_metric_micro(logits, mask)
            dice_metric_per_class(logits, mask)
            # Compute precision and recall
            precision_metric_macro(logits, mask)
            precision_metric_micro(logits, mask)
            recall_metric_macro(logits, mask)
            recall_metric_micro(logits, mask)
            # Compute IoU
            iou_metric_macro(logits, mask)
            iou_metric_micro(logits, mask)
            iou_metric_per_class(logits, mask)
            # Compute distance metric
            # preds_hd = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() # Assuming batch size is 1
            preds_hd = logits.squeeze(0).cpu().numpy() # Assuming batch size is 1
            mask_hd = mask.squeeze(0).cpu().numpy()
            hds.append(hd(preds_hd, mask_hd))
            hd95s.append(hd95(preds_hd, mask_hd))
            # Compute monte carlo loss
            # total_ssn_loss += ssn_loss_fn(logits, mask, edge_map, edge_weight, output_dict['distribution']).item()
            # Compute cross-entropy loss
            # total_ce_loss += cross_entropy_loss(logits, mask).item()

            # Get predictions and masks for confusion matrix
            # all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_preds.append(logits.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    mean_dice_macro = dice_metric_macro.compute().item()
    mean_dice_micro = dice_metric_micro.compute().item()
    dice_scores_per_class = dice_metric_per_class.compute().cpu().numpy()
    precision_macro = precision_metric_macro.compute().item()
    precision_micro = precision_metric_micro.compute().item()
    recall_macro = recall_metric_macro.compute().item()
    recall_micro = recall_metric_micro.compute().item()
    iou_macro = iou_metric_macro.compute().item()
    iou_micro = iou_metric_micro.compute().item()
    iou_per_class = iou_metric_per_class.compute().cpu().numpy()
    mean_ssn_loss = total_ssn_loss / len(dataloader)
    mean_ce_loss = total_ce_loss / len(dataloader)
    mean_hd = np.mean(hds)
    mean_hd95 = np.mean(hd95s)

    if confusion_matrix:
        plot_confusion_matrix(np.concatenate(all_masks).flatten(), np.concatenate(all_preds).flatten(), num_classes, filename)

    ret = {
        'mean_dice_macro': mean_dice_macro,
        'mean_dice_micro': mean_dice_micro,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'iou_macro': iou_macro,
        'iou_micro': iou_micro,
        'mean_ssn_loss': mean_ssn_loss,
        'mean_ce_loss': mean_ce_loss,
        'mean_hd': mean_hd,
        'mean_hd95': mean_hd95,
        'dice_scores_per_class': dice_scores_per_class,
        'iou_per_class': iou_per_class,
    }

    print(f"Better: {better}/{total}")

    # add invididual class scores   
    # for i in range(num_classes):
    #     ret[f'dice_only_class_{i}'] = dice_scores_per_class[i]

    return ret


def majority_voting_cuda(samples):
    """ Perform majority voting on the samples

    Args:
        samples: Tensor of shape (num_samples, H, W)

    Returns:
        preds: Tensor of shape  (H, W)
    """
    mode_values, _ = torch.mode(samples, dim=0)

    return mode_values


def plot_confusion_matrix(all_masks, all_preds, num_classes, filename):
    """ Plot the confusion matrix

    Args:
        all_masks: Numpy array containing all the true labels
        all_preds: Numpy array containing all the predicted labels
        num_classes: number of classes
    """
    print("Label distribution in true labels:", np.bincount(all_masks))
    print("Label distribution in predictions:", np.bincount(all_preds))

    cm = confusion_matrix(all_masks, all_preds, labels=[i for i in range(num_classes)])
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_normalised = np.divide(cm, row_sums, where=row_sums != 0)
    # print(cm_normalised)

    plt.imshow(cm_normalised, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalised Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, [f'{i}' for i in range(num_classes)], rotation=45)
    plt.yticks(tick_marks, [f'{i}' for i in range(num_classes)])

    fmt = '.2f'
    thresh = cm_normalised.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm_normalised[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm_normalised[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, format='png')
    plt.show()


if __name__ == '__main__':
    # os.system("nvidia-smi")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    SEED = 42
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    in_channels = 3
    num_classes = 5
    
    # Get val DataLoader
    # dataloader = get_train_loaders(batch_size=1)[1]
    test_path = 'labelbox_download/test_data_random.h5'
    dataloader = get_test_loader(test_path)

    # Evaluate UNet model
    # model_path = '/vol/bitbucket/dks20/renal_ssn/checkpoints/base_unet_radam_plateau_dropout20_aug_dice/epoch65_dice_75614.pth'
    # model = UNet(in_channels, num_classes).to(device)
    # model.load_state_dict(torch.load(model_path))

    # metrics = evaluate_deterministic(model, device, dataloader, num_classes, undo_norm=True, confusion_matrix=True, filename="cm_base_unet.png")
    # for key, value in metrics.items():
    #     print(f'{key}: {value}')

    # Evaluate Stochastic UNet model
    model_path = '/vol/bitbucket/dks20/renal_ssn/checkpoints/stoch_unet_adamwcyclic_randomdataset_actualaug_withnorm_dropout20/epoch84_dice_62359.pth'
    model = StochasticUNet(in_channels, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    metrics = evaluate_stochastic(model, device, dataloader, num_classes, confusion_matrix=True, filename="cm_stoch_unet_best5.png")
    for key, value in metrics.items():
        print(f'{key}: {value}')

    # Evaluate the Stochastic DeepMedic model
    # model_path = '/vol/bitbucket/dks20/renal_ssn/checkpoints/stoch_deepmedic_onecycle/epoch19_dice_42631.pth'
    # model = StochasticDeepMedic(in_channels, num_classes).to(device)
    # model.load_state_dict(torch.load(model_path))