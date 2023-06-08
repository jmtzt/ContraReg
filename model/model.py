import torch
import torch.nn as nn
import numpy as np

from model.utils import init_net, Normalize


def ae_conv_block(in_channels, out_channels, kernel_size, stride, padding, instance_norm=True, leaky_relu=True):
    layers = []
    layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
    if instance_norm:
        layers.append(nn.InstanceNorm3d(out_channels))
    if leaky_relu:
        layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = ae_conv_block(1, 32, 3, 2, 1)
        self.conv2 = ae_conv_block(32, 64, 3, 2, 1)
        self.conv3 = ae_conv_block(64, 128, 3, 2, 1)

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ae_conv_block(128, 64, 3, 1, 1)
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ae_conv_block(64, 32, 3, 1, 1)
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ae_conv_block(32, 32, 3, 1, 1)
        )

        self.conv4 = ae_conv_block(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv3d(64, 1, 3, 1, 1)

    def forward(self, x):
        out_features = []
        x = self.conv1(x)
        out_features.append(x)
        x = self.conv2(x)
        out_features.append(x)
        x = self.conv3(x)
        out_features.append(x)
        x = self.upconv1(x)
        out_features.append(x)
        x = self.upconv2(x)
        out_features.append(x)
        x = self.upconv3(x)
        out_features.append(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x, out_features


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[0]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 4, 1).flatten(1, 3)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                if isinstance(patch_id, torch.Tensor):
                    patch_id = patch_id.clone()
                else:
                    patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


def test_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Testing AutoEncoder model...')
    ae_model = AutoEncoder().to(device)
    x = torch.randn(4, 1, 32, 32, 32).to(device)
    y, features = ae_model(x)
    print(f'x: {x.shape}, y: {y.shape}')
    for i, f in enumerate(features):
        print(f'features[{i}]: {f.shape}')
    features = [f.to(device) for f in features]

    print('Testing MLP model...')
    mlp_model = PatchSampleF().to(device)
    feature_projections, _ = mlp_model(features, num_patches=256, patch_ids=None)

    for i, f in enumerate(feature_projections):
        print(f'feature_projections[{i}]: {f.shape}')


if __name__=='__main__':
    test_models()