import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from unet.stochastic_unet import StochasticUNet
from deepmedic.stochastic_deepmedic import StochasticDeepMedic
from mc_loss import SSNLossMCIntegral
from evaluate import evaluate_stochastic
from utils.get_dataloaders import get_train_loaders, get_test_loader
from utils.utils import crop_mask, write_metrics_to_tensorboard


def train(model,
          device,
          train_loader,
          val_loader,
          test_loader,
          batch_size,
          lr,
          edge_weight,
          num_epochs,
          num_classes,
          save_dir,
          start_epoch=1):
    """ Train a stochastic model on the kidney dataset

    Args:
        model: stochastic model
        device: device to run the model on
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        batch_size: batch size
        lr: learning rate
        num_epochs: number of epochs
        num_classes: number of classes
        save_dir: directory to save model checkpoints
    """
    model_name = model.__class__.__name__
    print(f'Training {model_name}')
    print(f'Saving checkpoints to {save_dir}')
    print(f'Learning rate: {lr}, Batch size: {batch_size}')
    
    writer = SummaryWriter(comment=f'{model_name}_LR_{lr}_BS_{batch_size}_Save_{save_dir}')

    loss_fn = SSNLossMCIntegral(num_mc_samples=20)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=1e-4, momentum=0.6)
    # optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-5, decoupled_weight_decay=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                  mode='max',
    #                                                  factor=0.2,
    #                                                  patience=5,
    #                                                  threshold=0.002,
    #                                                  min_lr=1e-6)
    # scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs, power=2)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loader))
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/100, max_lr=lr, step_size_up=2500)

    # Early stopping code
    best_dice = 0
    patience = 25
    threshold = 0.002
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + start_epoch}/{num_epochs}, LR {scheduler.get_last_lr()}', unit='batch') as pbar:
            for imgs, masks, edge_maps in train_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                edge_maps = edge_maps.to(device)

                if model_name == "StochasticDeepMedic":
                    masks = crop_mask(imgs, masks)

                optimizer.zero_grad()

                logits, output_dict = model(imgs)

                loss = loss_fn(logits, masks, edge_maps, edge_weight, output_dict['distribution'])

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                scheduler.step()

                pbar.set_postfix(**{'loss (batch)': loss.item(), 'lr': scheduler.get_last_lr()})
                pbar.update()

        writer.add_scalar('Loss/train', epoch_loss, epoch + start_epoch)
        print(f'Epoch {epoch + start_epoch}, Avg Loss: {epoch_loss / len(train_loader):.1f}')

        # Evaluate the model
        metrics = evaluate_stochastic(model, device, val_loader, num_classes, edge_weight)
        # scheduler.step(metrics['mean_dice_macro'])
        write_metrics_to_tensorboard(writer, metrics, epoch + start_epoch, 'val')

        mean_dice = metrics['mean_dice_macro']

        # writer.add_scalar('mean_dice/val', mean_dice, epoch)
        # writer.add_scalar('mean_ssn_loss/val', mean_ssn_loss, epoch)
        # writer.add_scalar('mean_ce_loss/val', mean_ce_loss, epoch)

        # print(f'Epoch {epoch + start_epoch}, Avg eval Dice: {mean_dice:.6f}, Avg eval SSN Loss: {mean_ssn_loss:.6f}, Avg eval CE Loss: {mean_ce_loss:.6f}')
        # print(f'Epoch {epoch + start_epoch}, Dice per class: {dice_scores_per_class}')

        # Save model if it has a better dice score
        if mean_dice > best_dice:
            dice = str(round(mean_dice, 5)).split('.')[1]
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'epoch{epoch + start_epoch}_dice_{dice}.pth')
            torch.save(model.state_dict(), save_path)

        # Early stopping with a small threshold
        if mean_dice > best_dice + threshold:
            best_dice = mean_dice
            counter = 0
        else:
            counter += 1
            if counter > patience:
                print(f'Early stopping at epoch {epoch + start_epoch}')
                break

        # Evaluate the model on test set
        test_metrics = evaluate_stochastic(model, device, test_loader, num_classes)
        write_metrics_to_tensorboard(writer, test_metrics, epoch + start_epoch, 'test')

    writer.close()


if __name__ == '__main__':
    # Check GPU, hyperparams (inc. oversampling), model and save_dir
    # os.system("nvidia-smi")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    SEED = 42
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_path = 'labelbox_download/train_data_random.h5'
    test_path = 'labelbox_download/test_data_random.h5'
    in_channels = 3
    num_classes = 5

    # Hyperparameters
    lr = 0.01
    batch_size = 10
    num_epochs = 250
    oversample = True
    oversample_classes = [2, 3]
    oversample_weight = 4
    downsample_empty = True
    downsample_weight = 0.5
    augment = True
    edge_weight = 1

    train_loader, val_loader = get_train_loaders(train_path=train_path,
                                                 batch_size=batch_size,
                                                 augment=augment,
                                                 calc_edge_map=edge_weight > 1,
                                                 oversample=oversample, 
                                                 oversample_classes=oversample_classes, 
                                                 oversample_weight=oversample_weight,
                                                 downsample_empty=downsample_empty,
                                                 downsample_weight=downsample_weight)
    
    test_loader = get_test_loader(test_path=test_path)

    # Train stochastic UNet
    model = StochasticUNet(in_channels, num_classes, rank=10, dropout_prob=0.2).to(device)
    save_dir = 'checkpoints/stoch_unet_adamwcyclic_randomdataset_actualaug_withnorm_dropout20'
    # model_path = "/vol/bitbucket/dks20/renal_ssn/checkpoints/stoch_unet_adamwcyclic_randomdataset_actualaug_nonorm_dropout20/epoch45_dice_53093.pth"
    # model.load_state_dict(torch.load(model_path))
    start_epoch = 1

    # Train stochastic DeepMedic
    # model = StochasticDeepMedic(in_channels, num_classes).to(device)
    # save_dir = 'checkpoints/stoch_deepmedic_onecycle_edge_weights'
    # start_epoch = 1

    train(model=model,
          device=device,
          train_loader=train_loader,
          val_loader=val_loader,
          test_loader=test_loader,
          batch_size=batch_size,
          lr=lr,
          edge_weight=edge_weight,
          num_epochs=num_epochs,
          num_classes=num_classes,
          save_dir=save_dir,
          start_epoch=start_epoch)