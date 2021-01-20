import time
import os.path as op
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from particle.nn.vae import VAE


def run():
    begin = time.time()
    N_LATENT = 32
    LAMB = 40
    LR = 1e-3
    N_EPOCH = 200
    BS = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    source_path = './data/train_set.npy'
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
            'loss': loss}, './output/vae/state_dict.tar')
    time_cost = int(time.time() - begin)
    print('Total time cost: {}h {}min {}s'.format(
        time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))


def study():
    from mayavi import mlab
    n_latent = 8
    vae = VAE(n_latent, 40)
    checkpoint = torch.load(
        '../params/state_dict({}, 40, 1e-3, 200, 128).tar'.format(n_latent), map_location=torch.device('cpu'))
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()

    source = torch.from_numpy(np.load('../data/samples.npy'))

    torch.manual_seed(4)
    coding = torch.zeros(1, n_latent)
    print(coding)
    # vae.random_generate(coding, voxel=True)
    fig1 = vae.random_generate(coding, opacity=1)
    coding[0, 4:5] = 1
    print(coding)
    fig2 = vae.random_generate(coding, color=(0, 1, 1), opacity=0.8)
    mlab.show()

    # np.random.seed(31425)
    # for i in range(10):
    #     index = np.random.randint(0, source.shape[0])
    #     sample = source[index:index+1].to(dtype=torch.float)
    #     print('Sample shape:', sample.shape)
    #     fig1, fig2 = vae.contrast(sample)
    #     time.sleep(1)
    #     mlab.savefig('../result/{}-original.png'.format(i+1), figure=fig1)
    #     mlab.savefig('../result/{}-fake.png'.format(i+1), figure=fig2)
    #     mlab.close(all=True)


def test():
    N_LATENT = 32
    latent = N_LATENT
    LAMB = 40
    LR = 1e-3
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # source_path = '/home/chuan/soil/data/train_set.npy'
    # source = torch.from_numpy(np.load(source_path))   # device(type='cpu')
    # x = source[20:30].to(dtype=torch.float)
    torch.manual_seed(3.14)
    vec = torch.randn(3000, latent, device=device)

    vae = VAE(n_latent=latent, lamb=LAMB, lr=LR).to(device)
    # state_dict(2, 40, 1e-3, 200, 128).tar
    # param = torch.load(
    #     f'output/vae/params/state_dict({latent}, 40, 1e-3, 200, 128).tar', map_location='cpu')
    param = torch.load('output/vae/state_dict.tar', map_location='cpu')
    vae.load_state_dict(param['model_state_dict'])
    # x_re = vae.contrast(x)
    # mlab.show()
    vae.eval()
    with torch.no_grad():
        target = vae.decoder(vec)
        target = target.numpy()
        np.save('output/vae.npy', target)


if __name__ == "__main__":
    run()
    # study()
    # test()
