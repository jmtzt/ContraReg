import math
import torch
import torch.nn as nn
import numpy as np
from torch import nn as nn

from torch.nn import functional as F
from packaging import version

from loss.utils import extract_patches
from model.model import PatchSampleF
from utils.utils import param_ndim_setup


class LNCCLoss(nn.Module):
    """
    Local Normalized Cross Correlation loss
    Adapted from VoxelMorph implementation:
    https://github.com/voxelmorph/voxelmorph/blob/5273132227c4a41f793903f1ae7e27c5829485c8/voxelmorph/torch/losses.py#L7
    """

    def __init__(self, window_size=7):
        super(LNCCLoss, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        # products and squares
        xsq = x * x
        ysq = y * y
        xy = x * y

        # set window size
        ndim = x.dim() - 2
        window_size = param_ndim_setup(self.window_size, ndim)

        # summation filter for convolution
        sum_filt = torch.ones(1, 1, *window_size).type_as(x)

        # set stride and padding
        stride = (1,) * ndim
        padding = tuple([math.floor(window_size[i] / 2) for i in range(ndim)])

        # get convolution function of the correct dimension
        conv_fn = getattr(F, f'conv{ndim}d')

        # summing over window by convolution
        x_sum = conv_fn(x, sum_filt, stride=stride, padding=padding)
        y_sum = conv_fn(y, sum_filt, stride=stride, padding=padding)
        xsq_sum = conv_fn(xsq, sum_filt, stride=stride, padding=padding)
        ysq_sum = conv_fn(ysq, sum_filt, stride=stride, padding=padding)
        xy_sum = conv_fn(xy, sum_filt, stride=stride, padding=padding)

        window_num_points = np.prod(window_size)
        x_mu = x_sum / window_num_points
        y_mu = y_sum / window_num_points

        cov = xy_sum - y_mu * x_sum - x_mu * y_sum + x_mu * y_mu * window_num_points
        x_var = xsq_sum - 2 * x_mu * x_sum + x_mu * x_mu * window_num_points
        y_var = ysq_sum - 2 * y_mu * y_sum + y_mu * y_mu * window_num_points

        lncc = cov * cov / (x_var * y_var + 1e-5)

        return -torch.mean(lncc)


class PatchNCELoss(nn.Module):
    def __init__(self, temperature = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     # reshape features as if they are all negatives of minibatch of size 1.
        #     batch_dim_for_bmm = 1
        # else:
        #     batch_dim_for_bmm = self.opt.batch_size
        batch_dim_for_bmm = 1

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class CRLoss(nn.Module):
    """
    Contrastive loss function from the ContraReg paper
    """

    def __init__(self,
                 t1_ae,
                 t2_ae,
                 num_patches=256,
                 lambda_NCE=1.0,):
        super(CRLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t1_ae = t1_ae
        self.t2_ae = t2_ae
        self.t1_ae.freeze()
        self.t2_ae.freeze()
        self.num_patches = num_patches
        self.lambda_NCE = lambda_NCE

    def forward(self, fixed, moving):
        t1_out, t1_features = self.t1_ae(fixed)
        t2_out, t2_features = self.t2_ae(moving)

        mlp_model = PatchSampleF().to(self.device)

        criterionNCE = []
        for _ in range(len(t1_features)):
            criterionNCE.append(PatchNCELoss().to(self.device))

        feat_t1_pool, sample_ids = mlp_model(t1_features, self.num_patches, None)
        feat_t2_pool, _ = mlp_model(t2_features, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_t1, f_t2, crit, in zip(feat_t1_pool, feat_t2_pool, criterionNCE):
            loss = crit(f_t1, f_t2) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / len(t1_features)


class DMMRLoss(nn.Module):

    def __init__(self,
                 patch_size=17,
                 batch_size=256,
                 model_path='/vol/alan/users/toz/midir-pycharm/dmmr_models/complete_camcan_tanh_hinge.pt',
                 ):
        super(DMMRLoss, self).__init__()
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path).to(self.device)

    def forward(self, fixed, moving):
        binary_mask = torch.zeros_like(fixed).to(self.device)
        binary_mask[fixed > 0] = 1

        fixed_patches = extract_patches(fixed, binary_mask, size=self.patch_size)
        moving_patches = extract_patches(moving, binary_mask, size=self.patch_size)

        concat_patches = torch.cat((fixed_patches, moving_patches), dim=1)

        keep_mask = torch.zeros(concat_patches.shape[0])
        for i, patch in enumerate(concat_patches):
            patch = patch[0].squeeze()
            zero_percentage = torch.mean((patch.squeeze() == 0).float()).item()
            if zero_percentage > 0.15:
                keep_mask[i] = 0
            else:
                keep_mask[i] = 1

        concat_patches = concat_patches[keep_mask.bool()]
        concat_patches_dset = torch.utils.data.TensorDataset(concat_patches)
        concat_patches_loader = torch.utils.data.DataLoader(concat_patches_dset,
                                                            batch_size=self.batch_size)

        outputs = torch.zeros(0).to(self.device)
        for batch in concat_patches_loader:
            fixed_patches_batch, moving_patches_batch = torch.unbind(batch[0], dim=1)
            fixed_patches_batch = fixed_patches_batch.unsqueeze(1).to(self.device)
            moving_patches_batch = moving_patches_batch.unsqueeze(1).to(self.device)
            out = self.model(fixed_patches_batch, moving_patches_batch)
            labels = out.clone()
            outputs = torch.cat((outputs, torch.mean(labels).view(-1)), dim=0)

        value = torch.mean(outputs)

        return value
