import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_f1_score
import numpy as np
import math
from scipy.stats import mode
from scipy.ndimage import label, binary_fill_holes, median_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ipywidgets import interactive, IntSlider, FloatSlider, HBox, VBox, Button, HTML, Layout

from utils.get_dataloaders import get_test_loader
from unet.unet_model import UNet
from unet.stochastic_unet import StochasticUNet
from deepmedic.stochastic_deepmedic import StochasticDeepMedic
from utils.utils import crop_mask, get_cmap


def visualise_preds(model, device, test_path, model_path, undo_norm, num_preds=25):
    """ Visualise the predictions of the model on the test set

    Args:
        model: trained model
        device: device to run the model on
        test_path: path to the test data
        model_path: path to the model
        num_preds: to limit no. of preds to visualise (full test loader length is 22)
    """
    print(f"Visualising preds for model at {model_path}")
    model_name = model.__class__.__name__
    test_loader = get_test_loader(test_path)

    model.eval()

    with torch.no_grad():
        for idx, (img, mask, _) in enumerate(test_loader):
            if undo_norm:
                img = undo_normalisation(img.squeeze(0)).unsqueeze(0) # shape = (1, C, H, W)

            img = img.to(device) # shape = (1, C, H, W)
            mask = mask.to(device) # shape = (1, H, W)

            preds = model(img) # shape = (1, num_classes, H, W)

            if model_name == 'DeepMedic':
                mask = crop_mask(preds, mask)

            ########### Split into smaller images then stitch outputs back together ##########
            top_left = img[:, :, :512, :512] # shape = (1, C, 512, 512)
            top_right = img[:, :, :512, 512:1024]
            bottom_left = img[:, :, 512:1024, :512]
            bottom_right = img[:, :, 512:1024, 512:1024]

            preds_tl = model(top_left) # shape = (1, num_classes, 512, 512)
            preds_tr = model(top_right)
            preds_bl = model(bottom_left)
            preds_br = model(bottom_right)

            top_half = torch.cat((preds_tl, preds_tr), dim=3) # shape = (1, num_classes, 512, 1024)
            bottom_half = torch.cat((preds_bl, preds_br), dim=3)
            preds_stitched = torch.cat((top_half, bottom_half), dim=2) # shape = (1, num_classes, 1024, 1024)

            dice_stitched = multiclass_f1_score(preds_stitched, mask, num_classes=5, average=None)
            mean_dice_stitched = multiclass_f1_score(preds_stitched, mask, num_classes=5, average='macro')

            preds_stitched = torch.argmax(preds_stitched, dim=1) # shape = (1, 1024, 1024)
            preds_stitched = postprocess(preds_stitched.squeeze(0).cpu().numpy())

            ##################################################################################

            dice = multiclass_f1_score(preds, mask, num_classes=5, average=None)
            mean_dice = multiclass_f1_score(preds, mask, num_classes=5, average='macro')

            preds = torch.argmax(preds, dim=1) # shape = (1, H, W)
            post = postprocess(preds.squeeze(0).cpu().numpy())

            # Convert to numpy arrays
            # img = undo_normalisation(img.squeeze(0).cpu()) # shape = (1, C, H, W)
            img = img.squeeze(0).permute(1, 2, 0).cpu().numpy() # shape = (H, W, C)
            mask = mask.squeeze(0).cpu().numpy() # shape = (H, W)
            preds = preds.squeeze(0).cpu().numpy() # shape = (H, W)

            print(f"Mask counts: {np.unique(mask, return_counts=True)}")
            print(f"Preds counts: {np.unique(preds, return_counts=True)}")
            print(f"Mean dice: {mean_dice.item()}, Dice scores per class: {dice}")
            print(f"Mean dice stitched: {mean_dice_stitched.item()}, Dice scores per class stitched: {dice_stitched}")

            ########### Plot the images ############################
            cmap = get_cmap(alpha=1, transparent_background=False)

            fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            ax[0].imshow(img)
            ax[0].set_title(f'Image {idx}')
            ax[0].axis('off')

            ax[1].imshow(mask, cmap=cmap, vmin=0, vmax=4)
            ax[1].set_title('Mask')
            ax[1].axis('off')

            # ax[2].imshow(preds, cmap=cmap, vmin=0, vmax=4)
            # ax[2].set_title('Predictions')
            # ax[2].axis('off')

            ax[2].imshow(post, cmap=cmap, vmin=0, vmax=4)
            ax[2].set_title('Postprocessed')
            ax[2].axis('off')

            ax[3].imshow(preds_stitched, cmap=cmap, vmin=0, vmax=4)
            ax[3].set_title('Stitched')
            ax[3].axis('off')

            plt.show()

            # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            # ax[0, 0].imshow(preds_tl, cmap=cmap, vmin=0, vmax=4)
            # ax[0, 0].set_title('Top left')
            # ax[0, 0].axis('off')

            # ax[0, 1].imshow(preds_tr, cmap=cmap, vmin=0, vmax=4)
            # ax[0, 1].set_title('Top right')
            # ax[0, 1].axis('off')

            # ax[1, 0].imshow(preds_bl, cmap=cmap, vmin=0, vmax=4)
            # ax[1, 0].set_title('Bottom left')
            # ax[1, 0].axis('off')

            # ax[1, 1].imshow(preds_br, cmap=cmap, vmin=0, vmax=4)
            # ax[1, 1].set_title('Bottom right')
            # ax[1, 1].axis('off')

            # plt.show()

            if idx >= num_preds - 1:
                break


def visualise_stochastic_preds(model, device, test_path, model_path, num_samples=5, num_preds=25):
    """ Visualise the predictions of the stochastic model on the test set

    Args:
        model: trained model
        device: device to run the model on
        test_path: path to the test data
        model_path: path to the model
        num_samples: number of samples to draw from the output distribution
        num_preds: to limit no. of preds to visualise (full test loader length is 22)
    """
    print(f"Visualising preds for model at {model_path}")
    test_loader = get_test_loader(test_path)

    model.eval()

    with torch.no_grad():
        for idx, (img, mask, _) in enumerate(test_loader):
            img = img.to(device) # shape = (1, C, H, W)

            logits, output_dict = model(img) # logits shape = (1, num_classes, H, W)
            dist = output_dict['distribution']

            logits = logits.argmax(dim=1).squeeze(0).cpu().numpy() # shape (1, num_classes, H, W) -> (H, W) 
            mask = mask.squeeze(0).cpu() # shape (1, H, W) -> (H, W)

            ########### LOGITS DICE SCORES ################
            print(f"Mask counts: {np.unique(mask, return_counts=True)}")
            print(f"Logits counts: {np.unique(logits, return_counts=True)}")
            logits = postprocess(logits) # shape = (H, W)
            print_dice_scores(torch.from_numpy(logits), mask, num_classes=5, name='Logits')

            ########### UNCERTAINTY ####################
            samples_uncertainty = dist.rsample((20,)).cpu() # shape = (20, 1, num_classes, H, W) (samples for majority voting + uncertainty calc)
            samples_uncertainty = samples_uncertainty.squeeze(1) # shape = (20, num_classes, H, W)

            entropy = calculate_entropy(samples_uncertainty) # shape = (H, W)

            if entropy.min() < 0 or entropy.max() > 1:
                print(f"WARNING entropy not between 0 and 1, Entropy min: {entropy.min()}, max: {entropy.max()}")

            # majority_vote = majority_voting(samples_uncertainty) # shape = (H, W)
            # majority_vote = postprocess(majority_vote.numpy()) # shape = (H, W)
            # print_dice_scores(torch.from_numpy(majority_vote), mask, num_classes=5, name='Majority vote')

            ########### GET SAMPLES TO DISPLAY ########################
            samples_display = dist.rsample((num_samples,)).squeeze(1).cpu() # shape = (num_samples, num_classes, H, W)
            samples_display = samples_display.argmax(dim=1).numpy() # shape = (num_samples, H, W)

            postprocessed_samples = []
            for i in range(num_samples):
                postprocessed_samples.append(postprocess(samples_display[i]))

            # samples_display = torch.stack(postprocessed_samples) # shape = (num_samples, H, W)

            dice_samples = []
            for i in range(num_samples):
                dice_samples.append(multiclass_f1_score(torch.from_numpy(samples_display[i]), mask, num_classes=5, average='macro').item())

            print(f"Macro dice scores for samples: {dice_samples}")
            print(f"Best sample: {np.argmax(dice_samples) + 1}, wih dice: {np.max(dice_samples)}")

            # calculate dice score for majority vote
            # dice_majority = multiclass_f1_score(majority_vote.to(device), mask.squeeze(0), num_classes=5, average='macro').item() 

            # Convert to numpy arrays for plotting
            # img = img / 255.0
            img = img.squeeze(0).cpu()
            img = undo_normalisation(img)
            img = img.permute(1, 2, 0).numpy() # shape = (H, W, C)
            mask = mask.cpu().numpy() # shape = (H, W)

            # Create a color map
            cmap = get_cmap(alpha=1, transparent_background=False)

            # Plot the images
            fig, ax = plt.subplots(2, num_samples, figsize=(40, 10))
            ax[0, 0].imshow(img, cmap=cmap, vmin=0, vmax=4)
            ax[0, 0].set_title(f'Image {idx}')
            ax[0, 0].axis('off')

            ax[0, 1].imshow(mask, cmap=cmap, vmin=0, vmax=4)
            ax[0, 1].set_title('Mask')
            ax[0, 1].axis('off')

            ax[0, 2].imshow(logits, cmap=cmap, vmin=0, vmax=4)
            ax[0, 2].set_title('Logits')
            ax[0, 2].axis('off')

            # ax[0, 3].imshow(majority_vote, cmap=cmap, vmin=0, vmax=4)
            # ax[0, 3].set_title('Majority vote (20 samples)')
            # ax[0, 3].axis('off')

            heatmap = ax[0, 3].imshow(entropy, cmap='viridis')
            cbar = fig.colorbar(heatmap)
            cbar.set_label('Entropy (0=Certainty, 1=Max Uncertainty)')
            ax[0, 3].set_title('Entropy (20 samples)')
            ax[0, 3].axis('off')
    
            for i in range(num_samples):
                ax[1, i].imshow(postprocessed_samples[i], cmap=cmap, vmin=0, vmax=4)
                ax[1, i].set_title(f'Sample {i + 1}')
                ax[1, i].axis('off')

            # OVERLAY
            cmap, colors = get_cmap(alpha=0.9, transparent_background=True, return_colours=True)

            adjusted_colors = np.zeros((*entropy.shape, 4))  # Create an RGBA image
            for i in range(1, len(colors)):
                adjusted_colors[logits == i, :3] = colors[i][:3]  # Color
                adjusted_colors[logits == i, 3] = (1 - entropy[logits == i]) * colors[i][3]  # Alpha, modulated by uncertainty

            ax[0, 4].imshow(img)
            ax[0, 4].set_title('Overlay')
            ax[0, 4].imshow(adjusted_colors)
            ax[0, 4].axis('off')

            plt.show()

            if idx >= num_preds - 1:
                break


def majority_voting(samples):
    """ Perform majority voting on the samples and calculate the standard deviation

    Args:
        samples: Tensor of array of shape (num_samples, num_classes, H, W)

    Returns:
        tensor of shape (H, W) with the majority vote labels
        tensor of shape (H, W) with the standard deviation as a measure of uncertainty
    """
    samples = samples.argmax(dim=1) # shape = (num_samples, H, W)
    majority_vote, _ = torch.mode(samples, dim=0)
    # std = np.std(samples, axis=0)

    # return torch.from_numpy(majority_vote), torch.from_numpy(std)
    return majority_vote


def calculate_entropy(samples):
    """ Calculate the entropy of the samples

    Args:
        samples: Tensor of array of shape (num_samples, num_classes, H, W)

    Returns:
        tensor of shape (H, W) with the entropy values
    """
    num_classes = samples.size(1)
    log_num_classes = math.log(num_classes)

    samples = samples.softmax(dim=1) # shape = (num_samples, num_classes, H, W)
    mean_probs = torch.mean(samples, dim=0) # shape = (num_classes, H, W)

    epsilon = 1e-8
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon) / log_num_classes, dim=0)

    return entropy


def postprocess(labelmap):

    '''
    Inspired by https://github.com/NBouteldja/KidneySegmentation_Histology
    with custom edge smoothing strategy
    '''
    
    # 1. REMOVE SMALL REGIONS
    structure = np.zeros((3, 3), dtype=int)
    structure[1, :] = 1
    structure[:, 1] = 1

    # Tubuli
    labeled, number = label(np.asarray(labelmap == 1, np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, number + 1):
        selection = (labeled == i)
        if selection.sum() < 500:  # remove too small noisy regions
            labelmap[selection] = 0

    # Vessel_indeterminate / Vein
    labeled, number = label(np.asarray(labelmap == 2, np.uint8), structure)
    for i in range(1, number + 1):
        selection = (labeled == i)
        if selection.sum() < 50:
            labelmap[selection] = 0

    # Artery
    labeled, number = label(np.asarray(labelmap == 3, np.uint8), structure)
    for i in range(1, number + 1):
        selection = (labeled == i)
        if selection.sum() < 400:
            labelmap[selection] = 0

    # Glomerui
    labeled, number = label(np.asarray(labelmap == 4, np.uint8), structure)
    for i in range(1, number + 1):
        selection = (labeled == i)
        if selection.sum() < 1500:
            labelmap[selection] = 0

    # 2. FILL HOLES
    # labelmap[binary_fill_holes(labelmap == 1)] = 1
    labelmap[binary_fill_holes(labelmap == 2)] = 2
    labelmap[binary_fill_holes(labelmap == 3)] = 3
    labelmap[binary_fill_holes(labelmap == 4)] = 4

    # 3. EDGE SMOOTHING
    for i in range(1, 5):

        tmp_label_mask = (labelmap == i).astype(int)
        labelmap[tmp_label_mask == 1] = 0

        for j in range(3):
            tmp_label_mask = (median_filter(tmp_label_mask, size=4))

        labelmap[tmp_label_mask == 1] = i

    return labelmap


def print_dice_scores(preds, mask, num_classes, name):
    dice_per_class = multiclass_f1_score(preds, mask, num_classes=num_classes, average=None)
    macro_dice = multiclass_f1_score(preds, mask, num_classes=num_classes, average='macro').item()
    micro_dice = multiclass_f1_score(preds, mask, num_classes=num_classes, average='micro').item()

    print(f"{name} macro dice: {macro_dice}") 
    print(f"{name} micro dice: {micro_dice}")
    print(f"{name} dice scores per class: {dice_per_class}")


def undo_normalisation(img, mean=[0.87770971, 0.80533165, 0.8622918], std=[0.10774853, 0.15939812, 0.10747936]):
    """
    Args:
        img: torch.Tensor (float) (C H W)
        mean: list of mean values
        std: list of std values
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    img = img * std + mean
    return img


if __name__ == '__main__':
    # os.system("nvidia-smi")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    SEED = 42
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    test_path = 'labelbox_download/test_data_random.h5'
    in_channels = 3
    num_classes = 5

    # Visualise predictions for UNet
    # model_path = '/vol/bitbucket/dks20/renal_ssn/checkpoints/base_unet_radam_plateau_dropout20_aug_dice/epoch65_dice_75614.pth'
    # model = UNet(in_channels, num_classes, dropout_prob=0.2).to(device)
    # model.load_state_dict(torch.load(model_path))
    # visualise_preds(model, device, test_path, model_path, undo_norm=True, num_preds=25)
    
    # Visualise predictions for Stochastic UNet
    model_path = 'checkpoints/stoch_unet_adamwcyclic_randomdataset_actualaug_withnorm_dropout20/epoch84_dice_62359.pth'
    model = StochasticUNet(in_channels, num_classes, rank=10).to(device)
    model.load_state_dict(torch.load(model_path))
    visualise_stochastic_preds(model, device, test_path, model_path, num_samples=5, num_preds=20)

    # Visualise predictions for Stochastic Deepmedic
    # model_path = '/vol/bitbucket/dks20/renal_ssn/checkpoints/stoch_deepmedic_onecycle/epoch19_dice_42631.pth'
    # model = StochasticDeepMedic(in_channels, num_classes).to(device)
    # model.load_state_dict(torch.load(model_path))
    # visualise_stochastic_preds(model, device, test_path, model_path, num_samples=5)
