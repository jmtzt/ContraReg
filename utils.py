""" Dataset helper functions """
import random
import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
import omegaconf

from matplotlib.patches import Rectangle
from omegaconf.listconfig import ListConfig
from torchio.visualization import rotate

from image import crop_and_pad, normalise_intensity
from image_io import load_nifti


def check_transform_history(history, show=False):
    if len(history) == 1 or len(history) == 2:
        # only ToCanonical and Tfm applied to both t1/t2
        # or only ToCanonical applied to both t1/t2
        return True

    t1_hist = history[1]
    t2_hist = history[2]

    if show:
        print(f'T1-specific tfms: {t1_hist}')
        print(f'T2-specific tfms: {t2_hist}')
    # Check if base transforms were the same for t1 and t2
    if t1_hist.name == t2_hist.name and t1_hist.__dict__ == t2_hist.__dict__:
        if show:
            print(f"{t1_hist.name} was applied to both t1 and t2 w/ the same params!")  # a.k.a. positive pair (yi=-1)
            print(f'T1 & T2 transform params:')
            pprint.pprint(t1_hist.__dict__, width=1)
        return True
    elif t1_hist.name == t2_hist.name and t1_hist.__dict__ != t2_hist.__dict__:
        if show:
            print(
                f"{t1_hist.name} was applied to both t1 and t2 but w/ different params!")  # a.k.a. negative pair (yi=1)
            print(f'T1 transform params:')
            pprint.pprint(t1_hist.__dict__, width=1)
            print(f'T2 transform params:')
            pprint.pprint(t2_hist.__dict__, width=1)
        return False
    else:
        if show:
            print(f"{t1_hist.name} was applied to t1 and {t2_hist.name} to t2!")  # a.k.a. negative pair (yi=1)
            print(f'T1 transform params:')
            pprint.pprint(t1_hist.__dict__, width=1)
            print(f'T2 transform params:')
            pprint.pprint(t2_hist.__dict__, width=1)
        return False


def plot_patch_subj(image, patch, out, title, ch=-1, show=True):
    x_min, x_max = patch.get_bounds()[0]
    y_min, y_max = patch.get_bounds()[1]
    z_min, z_max = patch.get_bounds()[2]

    patch_size = x_max - x_min

    args = (2, 3)
    fig, axes = plt.subplots(*args, )

    data = image.data[ch].squeeze()
    patch_data = patch.data[ch].squeeze()

    image_axes = axes[0]
    xlabels = len(axes) - 1

    sag_axis, cor_axis, axi_axis = image_axes
    sag_patch_axis, cor_patch_axis, axi_patch_axis = axes[1]

    indices = np.array(patch.get_center()).astype(int)
    i, j, k = indices
    slice_x = rotate(data[i, :, :], radiological=True)
    slice_y = rotate(data[:, j, :], radiological=True)
    slice_z = rotate(data[:, :, k], radiological=True)

    patch_indices = np.array(patch_data.shape) // 2
    p_i, p_j, p_k = patch_indices

    patch_slice_x = rotate(patch_data[p_i, :, :], radiological=True)
    patch_slice_y = rotate(patch_data[:, p_j, :], radiological=True)
    patch_slice_z = rotate(patch_data[:, :, p_k], radiological=True)

    kwargs = {}
    kwargs_patch = {}

    sr, sa, ss = image.spacing
    psr, psa, pss = patch.spacing

    kwargs['origin'] = 'lower'
    kwargs_patch['origin'] = 'lower'
    kwargs['cmap'] = 'gray'
    kwargs_patch['cmap'] = 'gray'

    p1, p2 = np.percentile(data, (0.5, 99.5))
    kwargs['vmin'] = p1
    kwargs['vmax'] = p2

    pp1, pp2 = np.percentile(patch_data, (0.5, 99.5))
    kwargs_patch['vmin'] = pp1
    kwargs_patch['vmax'] = pp2

    sag_aspect = ss / sa
    sag_axis.imshow(slice_x, aspect=sag_aspect, **kwargs)
    sag_rect = Rectangle((y_min, z_min), (y_max - y_min), (z_max - z_min),
                         linewidth=1, edgecolor='r', facecolor='none')
    sag_axis.add_patch(sag_rect)
    if xlabels:
        sag_axis.set_xlabel('A')
    sag_axis.set_ylabel('S')
    sag_axis.invert_xaxis()
    sag_axis.set_title('Sagittal')

    patch_sag_aspect = pss / psa
    sag_patch_axis.imshow(patch_slice_x, aspect=patch_sag_aspect, **kwargs)
    if xlabels:
        sag_patch_axis.set_xlabel('A')
    sag_patch_axis.set_ylabel('S')
    sag_patch_axis.invert_xaxis()
    sag_patch_axis.set_title('Sagittal - patch')

    cor_aspect = ss / sr
    cor_axis.imshow(slice_y, aspect=cor_aspect, **kwargs)
    cor_rect = Rectangle((x_min, z_min), (x_max - x_min), (z_max - z_min),
                         linewidth=1, edgecolor='r', facecolor='none')
    cor_axis.add_patch(cor_rect)
    if xlabels:
        cor_axis.set_xlabel('R')
    cor_axis.set_ylabel('S')
    cor_axis.invert_xaxis()
    cor_axis.set_title('Coronal')

    patch_cor_aspect = pss / psr
    cor_patch_axis.imshow(patch_slice_y, aspect=patch_cor_aspect, **kwargs)
    if xlabels:
        cor_patch_axis.set_xlabel('R')
    cor_patch_axis.set_ylabel('S')
    cor_patch_axis.invert_xaxis()
    cor_patch_axis.set_title('Coronal - patch')

    axi_aspect = sa / sr
    axi_axis.imshow(slice_z, aspect=axi_aspect, **kwargs)
    axi_rect = Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
                         linewidth=1, edgecolor='r', facecolor='none')
    axi_axis.add_patch(axi_rect)
    if xlabels:
        axi_axis.set_xlabel('R')
    axi_axis.set_ylabel('A')
    axi_axis.invert_xaxis()
    axi_axis.set_title('Axial')

    patch_axi_aspect = psa / psr
    axi_patch_axis.imshow(patch_slice_z, aspect=patch_axi_aspect, **kwargs)
    if xlabels:
        axi_patch_axis.set_xlabel('R')
    axi_patch_axis.set_ylabel('A')
    axi_patch_axis.invert_xaxis()
    axi_patch_axis.set_title('Axial - patch')

    plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    # fig.savefig(out)


def _to_tensor(data_dict):
    # cast to Pytorch Tensor
    for name, data in data_dict.items():
        data_dict[name] = torch.from_numpy(data).float()
    return data_dict


def _crop_and_pad(data_dict, crop_size):
    # cropping and padding
    for name, data in data_dict.items():
        data_dict[name] = crop_and_pad(data, new_size=crop_size)
    return data_dict


def _normalise_intensity(data_dict, keys=None, vmin=0., vmax=1.):
    """ Normalise intensity of data in `data_dict` with `keys` """
    if keys is None:
        keys = {'target', 'source', 'target_original'}

    # images in one pairing should be normalised using the same scaling
    vmin_in = np.amin(np.array([data_dict[k] for k in keys]))
    vmax_in = np.amax(np.array([data_dict[k] for k in keys]))

    for k, x in data_dict.items():
        if k in keys:
            data_dict[k] = normalise_intensity(x,
                                               min_in=vmin_in, max_in=vmax_in,
                                               min_out=vmin, max_out=vmax,
                                               mode="minmax", clip=True)
    return data_dict


def _shape_checker(data_dict):
    """Check if all data points have the same shape
    if so return the common shape, if not print data type"""
    data_shapes_dict = {n: x.shape for n, x in data_dict.items()}
    shapes = [x for _, x in data_shapes_dict.items()]
    if all([s == shapes[0] for s in shapes]):
        common_shape = shapes[0]
        return common_shape
    else:
        raise AssertionError(f'Not all data points have the same shape, {data_shapes_dict}')


def _magic_slicer(data_dict, slice_range=None, slicing=None):
    """Select all slices, one random slice, or some slices
    within `slice_range`, according to `slicing`
    """
    # slice selection
    num_slices = _shape_checker(data_dict)[0]

    # set range for slicing
    if slice_range is None:
        # all slices if not specified
        slice_range = (0, num_slices)
    else:
        # check slice_range
        assert isinstance(slice_range, (tuple, list, ListConfig))
        assert len(slice_range) == 2
        assert all(isinstance(s, int) for s in slice_range)
        assert slice_range[0] < slice_range[1]
        assert all(0 <= s <= num_slices for s in slice_range)

    # select slice(s)
    if slicing is None:
        # all slices within slice_range
        slicer = slice(slice_range[0], slice_range[1])

    elif slicing == 'random':
        # randomly choose one slice within range
        z = random.randint(slice_range[0], slice_range[1]-1)
        slicer = slice(z, z + 1)  # use slicer to keep dim

    elif isinstance(slicing, (list, tuple, ListConfig)):
        # slice several slices specified by slicing
        assert all(0 <= i <= 1 for i in slicing), f'Relative slice positions {slicing} need to be within [0, 1]'
        slicer = tuple(int(i * (slice_range[1] - slice_range[0])) + slice_range[0] for i in slicing)

    else:
        raise ValueError(f'Slicing mode {slicing} not recognised.')

    # slicing
    for name, data in data_dict.items():
        data_dict[name] = data[slicer, ...]  # (N, H, W)

    return data_dict


def _load2d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, N) ->  (N, H, W)
        data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
    return data_dict


def _load3d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
        data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
    return data_dict

def _load3d_custom(data_path_dict, mod_axis):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        if '_seg' in name:
            # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
            data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
        elif 'source' in name:
            # image is saved in shape (H, W, D) -> (ch=mod_axis, H, W, D)
            data_dict[name] = load_nifti(data_path).transpose(3, 0, 1 ,2)[mod_axis][np.newaxis, ...]
        else:
            # The default modality for target is 1
            data_dict[name] = load_nifti(data_path).transpose(3, 0, 1 ,2)[1][np.newaxis, ...]
    return data_dict


def plot_subj(tensor, title=''):
    tensor = tensor.squeeze().detach().cpu().numpy()

    ch = 50
    plt.subplot(131)
    plt.imshow(tensor[:, ch, :], cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(tensor[:, :, ch], cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(tensor[ch, :, :], cmap='gray')
    plt.axis('off')
    plt.suptitle(title)
    plt.show()


def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension

    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param
