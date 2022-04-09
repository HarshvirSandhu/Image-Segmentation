import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# [(i-k+2p)/s] + 1


class Conv_op(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv_op, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                      kernel_size=(3, 3), padding=1, stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels,
                      kernel_size=(3, 3), padding=1, stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# a = torch.rand(size=(32, 3, 100, 100))
# ob = Conv_op(input_channels=3, output_channels=10)
# b = ob(a)
# print(b.shape)

class UNET(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()

        # Down scaling part of U-Net
        for feature in features:
            self.down_layers.append(Conv_op(input_channels=input_channels, output_channels=feature))
            input_channels = feature

            # Shape after Conv transpose: [(i-1)*s + k - 2p]

        for feature in reversed(features):
            self.up_layers.append(
                nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=(2, 2), stride=(2, 2))
            )
            self.up_layers.append(Conv_op(input_channels=feature*2, output_channels=feature))

        self.bottle_neck = Conv_op(features[-1], features[-1]*2)
        self.conv_last = nn.Conv2d(in_channels=features[0], out_channels=output_channels, kernel_size=(1, 1))

    def forward(self, x):
        skip_connections = []

        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.up_layers), 2):
            x = self.up_layers[i](x)
            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_layers[i+1](concat_skip)

        return self.conv_last(x)


# img = torch.rand((1, 3, 100, 100))
# model = UNET()
# a = model(img)
# print(a.shape)
