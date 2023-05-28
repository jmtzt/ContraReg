import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size, stride, padding, instance_norm=True, leaky_relu=True):
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

        self.conv1 = conv_block(1, 32, 3, 2, 1)
        self.conv2 = conv_block(32, 64, 3, 2, 1)
        self.conv3 = conv_block(64, 128, 3, 2, 1)

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_block(128, 64, 3, 1, 1)
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_block(64, 32, 3, 1, 1)
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_block(32, 32, 3, 1, 1)
        )

        self.conv4 = conv_block(32, 64, 3, 1, 1)
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

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder().to(device)
    print(model)
    x = torch.randn(4, 1, 128, 128, 128).to(device)
    y, features = model(x)
    print(f'x: {x.shape}, y: {y.shape}')
    for i, f in enumerate(features):
        print(f'features[{i}]: {f.shape}')