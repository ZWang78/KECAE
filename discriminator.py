import torch
import torch.nn as nn

def weights_init_uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.1)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.1)

class Block32(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Block64(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Block128(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Block256(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class Siamese_VGG(nn.Module):
    def __init__(self, drop, use_w_init=True):
        super().__init__()
        self.block32 = Block32()
        self.block64 = Block64()
        self.block128 = Block128()
        self.block256 = Block256()

        self.final = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(256, 2)
        )

        if use_w_init:
            self.apply(weights_init_uniform)

        self.attention32 = ChannelAttention(32)
        self.attention64 = ChannelAttention(64)
        self.attention128 = ChannelAttention(128)
        self.attention256 = ChannelAttention(256)

    def forward_one(self, x):
        x = self.block32(x)
        x = x * self.attention32(x)
        x = self.block64(x)
        x = x * self.attention64(x)
        x = self.block128(x)
        x = x * self.attention128(x)
        x = self.block256(x)
        x = x * self.attention256(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        # For the sake of simplicity, we just add the features.
        # In a more complex model, you might concatenate or use a more sophisticated merging strategy.
        feats_final = out1 + out2
        return self.final(feats_final), feats_final