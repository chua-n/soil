import time
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from particle.nn.gan.dcgan import *
from particle.pipeline import Sand
from particle.mayaviOffScreen import mlab


torch.manual_seed(3.14)

LR = 0.0002
BETA = 0.5
N_EPOCH = 65
BS = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_path = '/home/chuan/soil/data/train_set.npy'
source = torch.from_numpy(np.load(source_path))   # device(type='cpu')
train_set = DataLoader(TensorDataset(source), batch_size=BS, shuffle=True)

net_D = Discriminator().to(device)
net_G = Generator().to(device)

# net_D.weights_init()
# net_G.weights_init()

optim_D = torch.optim.Adam(net_D.parameters(), lr=LR, betas=(BETA, 0.999))
optim_G = torch.optim.Adam(net_G.parameters(), lr=LR, betas=(BETA, 0.999))

# 要知道：G是通过优化D来间接提升自己的，故两个网络只需一个loss criterion
criterion = nn.BCELoss()

# Create a batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(5, n_latent, 1, 1, 1, device=device)
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

begin = time.time()
for epoch in range(N_EPOCH):
    for i, (x,) in enumerate(train_set):
        x = x.to(dtype=torch.float, device=device)
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # 判真
        label = torch.full((x.size(0),), real_label,
                           device=device, dtype=torch.float)
        net_D.zero_grad()
        judgement_real = net_D(x).view(-1)
        loss_D_real = criterion(judgement_real, label)
        loss_D_real.backward()
        D_x = judgement_real.mean().item()
        # 判假
        noise = torch.randn(x.size(0), n_latent, 1, 1, 1, device=device)
        fake = net_G(noise)
        label.fill_(fake_label)
        judgement_fake = net_D(fake.detach()).view(-1)
        loss_D_fake = criterion(judgement_fake, label)
        loss_D_fake.backward()
        D_G_z1 = judgement_fake.mean().item()
        loss_D = loss_D_real + loss_D_fake
        optim_D.step()

        # (2) Update G network: maximize log(D(G(z)))
        net_G.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        judgement = net_D(fake).view(-1)
        loss_G = criterion(judgement, label)
        loss_G.backward()
        D_G_z2 = judgement.mean().item()
        optim_G.step()

        if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
            time_cost = int(time.time() - begin)
            print('Time cost so far: {}h {}min {}s'.format(
                time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
            print("Epoch[{}/{}], Step [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f} / {:.4f}".
                  format(epoch + 1, N_EPOCH, i + 1, len(train_set), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

    # 每轮结束保存一次模型参数
    torch.save({
        'source_size': source.shape,
        'batch_size': BS,
        'epoch': '{}/{}'.format(epoch + 1, N_EPOCH),
        'step': '{}/{}'.format(i + 1, len(train_set)),
        'n_latent': n_latent,
        'discriminator_state_dict': net_D.state_dict(),
        'generator_state_dict': net_G.state_dict(),
        'optim_D_state_dict': optim_D.state_dict(),
        'optim_G_state_dict': optim_G.state_dict(),
        'loss_D': loss_D,
        'loss_G': loss_G,
        'D(x)': D_x,
        'D(G(z))': "{:.4f} / {:.4f}".format(D_G_z1, D_G_z2)}, 'output/gan/param/state_dict.tar')

    net_G.eval()
    with torch.no_grad():
        cubes = net_G(fixed_noise).to('cpu').numpy()
        for i, cube in enumerate(cubes):
            cube = cube[0]
            sand = Sand(cube)
            sand.visualize(voxel=True, glyph='point', scale_mode='scalar')
            mlab.outline()
            mlab.axes()
            mlab.savefig(f'output/gan/process/{epoch + 1}-{i + 1}.png')
            mlab.close()
            # x, y, z = np.nonzero(cube)
            # flatten = cube.reshape(-1)
            # val = flatten[np.nonzero(flatten)]
            # ax = plt.axes(projection='3d')
            # ax.scatter(x, y, z, s=val)
            # plt.axis('off')
            # plt.savefig('{}-{}.png'.format(epoch + 1, i + 1), dpi=200)
    net_G.train()

# 以下为查看当前生成效果
# checkpoint = torch.load('state_dict_20200206.tar')
# net_G.load_state_dict(checkpoint['generator_state_dict'])
# net_G.eval()
# with torch.no_grad():
#     vec = torch.randn(1, n_latent, 1, 1, 1, device=device)
#     cube = net_G(vec).to('cpu').numpy()[0, 0]
#     x, y, z = np.nonzero(cube)
#     flatten = cube.reshape(-1)
#     val = flatten[np.nonzero(flatten)]
#     # mlab.points3d(x, y, z, val)
#     # mlab.show()
#     ax = plt.axes(projection='3d')
#     # plot voxels
#     # cube[cube > 0.7] = 1
#     # cube[cube <= 0.7] = 0
#     # ax.voxels(cube)
#     ax.scatter(x, y, z, s=val)
#     plt.axis('off')
#     plt.show()
