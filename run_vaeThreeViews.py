import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from particle.nn.threeViews import Reconstructor
from particle.nn.vaeThreeViews import VAE


def run():
    torch.manual_seed(3.14)

    N_LATENT = 3
    LAMB = 40
    LR = 1e-3
    N_EPOCH = 80
    BS = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_path = './data'
    cwd = os.getcwd()
    os.chdir(source_path)
    source_train = torch.from_numpy(np.load('train_set.npy'))
    source_test = torch.from_numpy(np.load('test_set.npy'))
    os.chdir(cwd)
    projection_train = Reconstructor.get_projection_set(source_train)
    projection_test = Reconstructor.get_projection_set(source_test)
    train_set = DataLoader(TensorDataset(projection_train),
                           batch_size=BS, shuffle=True)
    test_set = DataLoader(TensorDataset(projection_test),
                          batch_size=2*BS, shuffle=False)

    vae = VAE(n_latent=N_LATENT, lamb=LAMB, lr=LR).to(device)
    vae.initialize()
    optim = vae.optimizer()
    print(vae)

    losses = []
    test_losses = []
    time_begin = time.time()
    for epoch in range(N_EPOCH):
        vae.train()
        for i, (x,) in enumerate(train_set):
            x = x.to(dtype=torch.float32, device=device)
            x_re, mu, log_sigma = vae(x)
            loss_re, loss_kl, loss = vae.criterion(x_re, x, mu, log_sigma)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
                time_cost = int(time.time() - time_begin)
                print('Time cost so far: {}h {}min {}s'.format(
                    time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
                print("Epoch[{}/{}], Step [{}/{}], Loss_re: {:.4f}, Loss_kl: {:.4f}, Loss: {:.4f}".
                      format(epoch + 1, N_EPOCH, i + 1, len(train_set), loss_re.item(), loss_kl.item(), loss.item()))

        # 评估在测试集上的损失
        # vae.eval()
        # with torch.no_grad():
        #     # 为啥autopep8非要把我的lambda表达式给换成def函数形式？？？
        #     def transfer(x): return x.to(dtype=torch.float32, device=device)
        #     # sum函数将其内部视为生成器表达式？？？
        #     test_loss = sum(vae.criterion(
        #         vae(transfer(x)), transfer(y)) for x, y in test_set)
        #     test_loss /= len(test_set)  # 这里取平均数
        #     test_losses.append(test_loss)
        # time_cost = int(time.time() - time_begin)
        # print('Time cost so far: {}h {}min {}s'.format(
        #     time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
        # print('The loss in test set after {}-th epoch is: {:.4f}'.format(
        #     epoch + 1, test_loss))
        # ckpt_name = f"state_dict_{test_loss}.tar" if test_loss < 7300 else "state_dict.tar"
        torch.save({  # 每轮结束保存一次模型数据
            'source_path': os.path.abspath(source_path),
            'train_set_size': source_train.shape,
            'test_set_size': source_test.shape,
            'batch_size': BS,
            'epoch': '{}/{}'.format(epoch + 1, N_EPOCH),
            'step': '{}/{}'.format(i + 1, len(train_set)),
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss_re': loss.item()}, "./state_dict.tar")

    time_cost = int(time.time() - time_begin)
    print('Total time cost: {}h {}min {}s'.format(
        time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))

    # Plot the training losses.
    plt.style.use('seaborn')
    plt.figure()
    plt.title("Reconstruction Loss in Train Set")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.savefig("trainSet")
    # Plot the testing losses.
    # plt.figure()
    # plt.title("Reconstruction Loss in Test Set")
    # plt.xlabel("Epoches")
    # plt.ylabel("Loss")
    # plt.plot(test_losses)
    # plt.savefig("testSet")

    return


if __name__ == "__main__":
    run()
