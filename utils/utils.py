import torch
import matplotlib.colors as mcolors

def crop_mask(preds, mask):
    """ Crop the mask to the size of the prediction.
    
    Assumes mask is larger than prediction.
    Assumes both are square.
      
    Args:
        preds (torch.Tensor): shape (B, C, H, W)
        mask (torch.Tensor): shape (B, H, W)
    
    """
    preds_shape = preds.shape[-1]
    mask_shape = mask.shape[-1]
    diff = mask_shape - preds_shape

    mask = mask[:, diff//2:diff+preds_shape, diff//2:diff+preds_shape]

    return mask


def get_cmap(alpha, transparent_background, return_colours=False):

    bg_alpha = 0 if transparent_background else alpha

    # colors = [(128/255, 0, 128/255, bg_alpha), # purple
    #           (1, 0, 0, alpha), # red
    #           (1, 1, 0, alpha), # yellow
    #           (0, 1, 0, alpha), # green
    #           (0, 0, 1, alpha)] # blue
    
    colors = [(128/255, 0, 128/255, bg_alpha), # purple (background)
              (1, 0, 0, alpha), # red (tubules)
              (1, 1, 0, alpha), # yellow (indeterminate)
              (50/255, 200/255, 50/255, alpha), # green (artery)
              (0, 0, 200/255, alpha)] # blue (glomeruli)
    
    cmap = mcolors.ListedColormap(colors)
    
    if return_colours:
        return cmap, colors
    else:
        return cmap
    

def write_metrics_to_tensorboard(writer, metrics, epoch, prefix):
    for key, value in metrics.items():
        if 'per_class' not in key:
            writer.add_scalar(f'{prefix}/{key}', value, epoch)

        if 'only_class' not in key:
            print(f'{key}: {value}')
    


