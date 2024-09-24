import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder1 = nn.Sequential(
            # input is 1 x 299 x 299
            nn.Conv2d(1, 32, 3, 2, 0, bias=False),  # 32*149*149
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 0, bias=False),  # 64*74*74
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 0, bias=False),  # 128*36*36
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 0, bias=False),  # 256*17*17
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 0, bias=False),  # 512*8*8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 3, 2, 0, bias=False),  # 1024*3*3
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 2048, 3, 2, 0, bias=False),  # 2048*1*1
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder2 = nn.Sequential(
            # input is 1 x 299 x 299
            nn.Conv2d(1, 32, 3, 2, 0, bias=False),  # 32*149*149
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 0, bias=False),  # 64*74*74
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 0, bias=False),  # 128*36*36
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 0, bias=False),  # 256*17*17
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 0, bias=False),  # 512*8*8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 3, 2, 0, bias=False),  # 1024*3*3
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 2048, 3, 2, 0, bias=False),  # 2048*1*1
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img):
        output1 = self.encoder1(img)
        output2 = self.encoder2(img)
        return output1, output2