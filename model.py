import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    # input size == (3, 224, 224)

    def __init__(self):
        super().__init__()
        alexnet = models.alexnet(pretrained=True)
        self.feature = list(alexnet.children())[0]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        # x.shape == batch, 256, 1, 1
        return x


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        # (256, 6, 6) pool
        self.pool1 = nn.AdaptiveAvgPool2d((13, 13))
        # (256, 13, 13) conv, no size change
        self.conv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        # (256, 13, 13) conv, no size change
        self.conv2 = nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1)
        # (384, 13, 13) conv, no size change
        self.conv3 = nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1)
        # (192, 13, 13) pool, upscale to 27, 27
        self.pool2 = nn.AdaptiveAvgPool2d((27, 27))
        # (192, 27, 27) conv, no size change
        self.conv4 = nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2)
        # (64, 27, 27) pool, upscale to 55, 55
        self.pool3 = nn.AdaptiveAvgPool2d((55, 55))
        # (64, 55, 55) conv, upscale to 3, 224, 224,
        self.conv5 = nn.ConvTranspose2d(
            64, 3, kernel_size=10, stride=4, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5(x))
        x = self.sigmoid(x)
        return x


def build_model(gpus=[0]):
    model = nn.Sequential()
    model.add_module("Encoder", Encoder())
    model.add_module("Decoder", Decoder())
    if gpus[0] != -1:
        model = model.to(f"cuda:{gpus[0]}")
        if len(gpus) > 1:
            model = nn.DataParallel(
                model, device_ids=gpus, output_device=gpus[0])
    return model


if __name__ == "__main__":
    from torchinfo import summary
    net = nn.Sequential()
    net.add_module("Encoder", Encoder())
    net.add_module("Decoder", Decoder())
    summary(net, (3, 224, 224))
