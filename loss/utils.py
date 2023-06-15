import torch
import torch.nn.functional as F

def mask_to_3d_bbox(mask):
    mask = mask.squeeze()
    bounding_boxes = torch.zeros((6), device=mask.device, dtype=torch.float)

    z, y, x = torch.where(mask != 0)
    bounding_boxes[0] = torch.min(x)
    bounding_boxes[1] = torch.min(y)
    bounding_boxes[2] = torch.min(z)
    bounding_boxes[3] = torch.max(x)
    bounding_boxes[4] = torch.max(y)
    bounding_boxes[5] = torch.max(z)

    return bounding_boxes


def extract_patches(tensor, mask, size=17):
    wc, ws, wa = size, size, size  # window size
    sc, ss, sa = size, size, size  # stride

    x_min, y_min, z_min, x_max, y_max, z_max = mask_to_3d_bbox(mask)

    x_min, y_min, z_min = int(x_min.item()), int(y_min.item()), int(z_min.item())
    x_max, y_max, z_max = int(x_max.item()), int(y_max.item()), int(z_max.item())

    tensor = tensor[:, :, z_min:z_max, y_min:y_max, x_min:x_max]

    # Pad the input such that it is divisible by the window size
    padding_values = []
    for dim_size in tensor.shape[2:]:
        remainder = dim_size % wc
        if remainder != 0:
            padding = wc - remainder
        else:
            padding = 0
        padding_values.extend([padding // 2, padding - padding // 2])

    padding_values.reverse()
    padded = F.pad(tensor, padding_values, 'constant')

    # Create the patches of wc fixed ws fixed wa
    patches = padded.unfold(2, wc, sc).unfold(3, ws, ss).unfold(4, wa, sa)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(-1, wc, ws, wa)

    return patches.unsqueeze(1)
