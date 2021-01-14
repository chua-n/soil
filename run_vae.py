import time
import os.path as op
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from particle.nn.vae import VAE

begin = time.time()
N_LATENT = 32
LAMB = 40
LR = 1e-3
N_EPOCH = 2
BS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
source_path = '../data/all.npy'
source = torch.from_numpy(np.load(source_path))   # device(type='cpu')
train_set = DataLoader(TensorDataset(source), batch_size=BS, shuffle=True)
# source和train_set共享了内存
vae = VAE(n_latent=N_LATENT, lamb=LAMB, lr=LR).to(device)
vae.initialize()
optimizer = vae.optimizer()
for epoch in range(N_EPOCH):
    for i, (x,) in enumerate(train_set):
        x = x.to(dtype=torch.float, device=device)
        x_re, mu, log_sigma = vae(x)
        loss_re, loss_kl, loss = vae.criterion(x_re, x, mu, log_sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
            time_cost = int(time.time() - begin)
            print('Time cost so far: {}h {}min {}s'.format(
                time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
            print("Epoch[{}/{}], Step [{}/{}], Loss_re: {:.4f}, Loss_kl: {:.4f}, Loss: {:.4f}".
                  format(epoch + 1, N_EPOCH, i + 1, len(train_set), loss_re.item(), loss_kl.item(), loss.item()))
    torch.save({  # 每轮结束保存一次模型数据
        'source_path': op.abspath(source_path),
        'source_size': source.shape,
        'batch_size': BS,
        'epoch': '{}/{}'.format(epoch + 1, N_EPOCH),
        'step': '{}/{}'.format(i + 1, len(train_set)),
        'n_latent': vae.n_latent,
        'lamb': vae.lamb,
        'lr': vae.lr,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_re': loss_re,
        'loss_kl': loss_kl,
        'loss': loss}, '../params/state_dict.tar')
time_cost = int(time.time() - begin)
print('Total time cost: {}h {}min {}s'.format(
    time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
