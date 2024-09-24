import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import KneeGradingDataset, CenterCrop
from encoder import Encoder
from decoder import Decoder
from discriminator import Siamese_VGG
from loss import distance_loss

# Configuration
CUDA = 0
cuda = True if torch.cuda.is_available() else False
img_shape = (1, 128, 128)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_pair_Tensor(I):
    tensor_l = torch.flip(I[:,:,100:228,171:299][0].cpu(), [-1]).unsqueeze(1)
    tensor_m = I[:,:,100:228,0:128][0].unsqueeze(1)
    for i in range(I.shape[0]-1):
        tensor_l = torch.cat((tensor_l, torch.flip(I[:,:,100:228,171:299][i+1].cpu(), [-1]).unsqueeze(1)), 0)
        tensor_m = torch.cat((tensor_m, I[:,:,100:228,0:128][i+1].unsqueeze(1)), 0)
    return tensor_l,tensor_m

# Initialize models
discriminator2 = Siamese_VGG(0.2)
encoder = Encoder()
decoder = Decoder()

if cuda:
    discriminator2.cuda(CUDA)
    discriminator2.apply(weights_init)
    encoder.cuda(CUDA)
    encoder.apply(weights_init)
    decoder.cuda(CUDA)
    decoder.apply(weights_init)

# Data loader
transf_tens = transforms.Compose([
    transforms.ToTensor()])
augment_transforms = transforms.Compose([
    CenterCrop(299)])
train_ds = KneeGradingDataset('./OAI_m', transform=transf_tens, augment=augment_transforms, stage='train')
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Optimizers
optimizer_D_ = torch.optim.Adam(discriminator2.parameters(),lr=0.0001)
optimizer_E = torch.optim.Adam(encoder.parameters(),lr=0.0001)
optimizer_DE = torch.optim.Adam(decoder.parameters(),lr=0.0001)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Losses
mse = nn.MSELoss()
loss_Dis = distance_loss()

# Training
batches_done = 0
device = torch.device("cuda:" + str(CUDA) if (torch.cuda.is_available()) else "cpu")

print("Starting Training Loop...")
for epoch in range(100):
    for i, (img, img2, l, m, l2, m2, label_0, label_2) in enumerate(train_loader):
        img0 = Variable(img.cuda(CUDA))
        img2 = Variable(img2.cuda(CUDA))
        l0 = Variable(l.cuda(CUDA))
        l2 = Variable(l2.cuda(CUDA))
        m0 = Variable(m.cuda(CUDA))
        m2 = Variable(m2.cuda(CUDA))
        label_0 = Variable(label_0.long().cuda(CUDA))
        label_2 = Variable(label_2.long().cuda(CUDA))

        # Forward pass
        vector_Key0, vector_Unrelated0 = encoder(img0)
        vector_img0 = vector_Key0 + vector_Unrelated0

        vector_Key2, vector_Unrelated2 = encoder(img2)
        vector_img2 = vector_Key2 + vector_Unrelated2

        vector_Key_Exchange_img0 = vector_Key2 + vector_Unrelated0
        vector_Key_Exchange_img2 = vector_Key0 + vector_Unrelated2

        R_img0 = decoder(vector_img0)
        R_img2 = decoder(vector_img2)

        Key_Exchange_img0 = decoder(vector_Key_Exchange_img0)
        Key_Exchange_img2 = decoder(vector_Key_Exchange_img2)

        Key_Exchange_img0_l, Key_Exchange_img0_m = get_pair_Tensor(Key_Exchange_img0)
        Key_Exchange_img0_l = Key_Exchange_img0_l.cuda(CUDA)
        Key_Exchange_img0_m = Key_Exchange_img0_m.cuda(CUDA)


        Key_Exchange_img2_l, Key_Exchange_img2_m = get_pair_Tensor(Key_Exchange_img2)
        Key_Exchange_img2_l = Key_Exchange_img2_l.cuda(CUDA)
        Key_Exchange_img2_m = Key_Exchange_img2_m.cuda(CUDA)


        # Train Discriminator
        optimizer_D_.zero_grad()
        for p in discriminator2.parameters():
            p.requires_grad = True

        loss_D1 = F.cross_entropy(discriminator2(l0,m0)[0],label_0)
        loss_D2 = F.cross_entropy(discriminator2(l2,m2)[0],label_2)
        loss_D = loss_D1 + loss_D2
        loss_D.backward()
        optimizer_D_.step()

        # Train Encoder and Decoder
        optimizer_E.zero_grad()
        optimizer_DE.zero_grad()
        for p in discriminator2.parameters():
            p.requires_grad = False

        loss_distance0 = loss_Dis(vector_Key0, vector_Unrelated0)
        loss_distance2 = loss_Dis(vector_Key2, vector_Unrelated2)
        loss_distance = loss_distance0 + loss_distance2

        loss_R0 = mse(img0, R_img0)
        loss_R2 = mse(img2, R_img2)
        loss_R = loss_R0 + loss_R2

        loss_ce0 = F.cross_entropy(discriminator2(Key_Exchange_img0_l, Key_Exchange_img0_m)[0],label_2)
        loss_ce2 = F.cross_entropy(discriminator2(Key_Exchange_img2_l, Key_Exchange_img2_m)[0],label_0)
        loss_ce = loss_ce0 + loss_ce2

        loss_total = loss_R + 0.001 * loss_distance + 0.01 * loss_ce
        loss_total.backward()

        optimizer_E.step()
        optimizer_DE.step()

        print(f"[Epoch {epoch}/100] [Batch {i}/{len(train_loader)}] "
              f"[Reconstruction Loss: {loss_R.item():.4f}] [Discriminator Loss: {loss_D.item():.4f}] "
              f"[Distance Loss: {loss_distance.item():.4f}] [CE Loss: {loss_ce.item():.4f}]")
        batches_done += 1

    if epoch % 10 == 0 and epoch != 0:
        torch.save(encoder.state_dict(), f'KECAE_encoder_epoch_{epoch}.pth')
        torch.save(decoder.state_dict(), f'KECAE_decoder_epoch_{epoch}.pth')
        torch.save(discriminator2.state_dict(), f'KECAE_siamese_epoch_{epoch}.pth')