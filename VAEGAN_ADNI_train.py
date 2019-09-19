import numpy as np
import torch
import os

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from skimage.transform import resize
from nilearn import plotting
from ADNI_dataset import *
from BRATS_dataset import *
from ATLAS_dataset import *
from Model_VAEGAN import *




#%% md
# Configuration
#%%
BATCH_SIZE=4
max_epoch = 100
gpu = True
workers = 4

reg = 5e-10

gamma = 20
beta = 10

Use_BRATS=False
Use_ATLAS = False

#setting latent variable sizes
latent_dim = 1000

# %%
trainset = ADNIdataset(augmentation=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=workers)
if Use_BRATS:
    # 'flair' or 't2' or 't1ce'
    trainset = BRATSdataset(imgtype='flair')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=workers)
if Use_ATLAS:
    trainset = ATLASdataset(augmentation=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=workers)
# %%
G = Generator(noise=latent_dim)
D = Discriminator()
E = Encoder()

G.cuda()
D.cuda()
E.cuda()
# %%
g_optimizer = optim.Adam(G.parameters(), lr=0.0001)
d_optimizer = optim.Adam(D.parameters(), lr=0.0001)
e_optimizer = optim.Adam(E.parameters(), lr=0.0001)
# %% md
# Training
# %%
N_EPOCH = 100

real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda())
fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda())
criterion_bce = nn.BCELoss()
criterion_l1 = nn.L1Loss()

# %%
for epoch in range(N_EPOCH):
    for step, real_images in enumerate(train_loader):
        _batch_size = real_images.size(0)
        real_images = Variable(real_images, requires_grad=False).cuda()
        z_rand = Variable(torch.randn((_batch_size, latent_dim)), requires_grad=False).cuda()
        mean, logvar, code = E(real_images)
        x_rec = G(code)
        x_rand = G(z_rand)
        ###############################################
        # Train D
        ###############################################
        d_optimizer.zero_grad()

        d_real_loss = criterion_bce(D(real_images), real_y[:_batch_size])
        d_recon_loss = criterion_bce(D(x_rec), fake_y[:_batch_size])
        d_fake_loss = criterion_bce(D(x_rand), fake_y[:_batch_size])

        dis_loss = d_recon_loss + d_real_loss + d_fake_loss
        dis_loss.backward(retain_graph=True)

        d_optimizer.step()

        ###############################################
        # Train G
        ###############################################
        g_optimizer.zero_grad()
        output = D(real_images)
        d_real_loss = criterion_bce(output, real_y[:_batch_size])
        output = D(x_rec)
        d_recon_loss = criterion_bce(output, fake_y[:_batch_size])
        output = D(x_rand)
        d_fake_loss = criterion_bce(output, fake_y[:_batch_size])

        d_img_loss = d_real_loss + d_recon_loss + d_fake_loss
        gen_img_loss = -d_img_loss

        rec_loss = ((x_rec - real_images) ** 2).mean()

        err_dec = gamma * rec_loss + gen_img_loss

        err_dec.backward(retain_graph=True)
        g_optimizer.step()
        ###############################################
        # Train E
        ###############################################
        prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
        prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
        err_enc = prior_loss + beta * rec_loss

        e_optimizer.zero_grad()
        err_enc.backward()
        e_optimizer.step()
        ###############################################
        # Visualization
        ###############################################

        if step % 10 == 0:
            print('[{}/{}]'.format(epoch, N_EPOCH),
                  'D: {:<8.3}'.format(dis_loss.data[0].cpu().numpy()),
                  'En: {:<8.3}'.format(err_enc.data[0].cpu().numpy()),
                  'De: {:<8.3}'.format(err_dec.data[0].cpu().numpy())
                  )

            featmask = np.squeeze((0.5 * real_images[0] + 0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask, affine=np.eye(4))
            plotting.plot_img(featmask, title="X_Real")
            plotting.show()

            featmask = np.squeeze((0.5 * x_rec[0] + 0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask, affine=np.eye(4))
            plotting.plot_img(featmask, title="X_DEC")
            plotting.show()

            featmask = np.squeeze((0.5 * x_rand[0] + 0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask, affine=np.eye(4))
            plotting.plot_img(featmask, title="X_rand")
            plotting.show()

    torch.save(G.state_dict(), './chechpoint/G_VG_ep_' + str(epoch + 1) + '.pth')
    torch.save(D.state_dict(), './chechpoint/D_VG_ep_' + str(epoch + 1) + '.pth')
    torch.save(E.state_dict(), './chechpoint/E_VG_ep_' + str(epoch + 1) + '.pth')

