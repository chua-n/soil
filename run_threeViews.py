import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from particle.nn.threeViews import Reconstructor, get_projection_set, train


def run():
    torch.manual_seed(3.14)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Reconstructor("./particle/nn/config/threeViews.xml",
                          log_dir="output/log", ckpt_dir='output/threeViews')

    source_path = './data'
    cwd = os.getcwd()
    os.chdir(source_path)
    source_train = torch.from_numpy(np.load('train_set.npy'))
    source_test = torch.from_numpy(np.load('test_set.npy'))
    os.chdir(cwd)
    projection_train = get_projection_set(source_train)
    projection_test = get_projection_set(source_test)
    train_set = DataLoader(
        TensorDataset(projection_train, source_train), batch_size=model.hp['bs'], shuffle=True)
    test_set = DataLoader(TensorDataset(
        projection_test, source_test), batch_size=2*model.hp['bs'], shuffle=False)

    losses, test_losses = train(model, train_set, test_set, device)

    # Plot the training losses.
    plt.style.use('seaborn')
    plt.figure()
    plt.title("Reconstruction Loss in Train Set")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.savefig("trainSet")
    # Plot the testing losses.
    plt.figure()
    plt.title("Reconstruction Loss in Test Set")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.plot(test_losses)
    plt.savefig("testSet")

    return


def study():
    from mayavi import mlab
    from skimage import io as skio
    model = Reconstructor("./particle/nn/config/threeViews-test.xml")
    state_dict = torch.load('./state_dict.tar',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    def test_generate(ind):
        cwd = os.getcwd()
        os.chdir('./data/简单几何图形')
        filenames = sorted(os.listdir())
        filename = filenames[ind]
        print(filename)
        image = skio.imread(filename, as_gray=True)
        os.chdir(cwd)
        x = torch.empty((3, 64, 64), dtype=torch.uint8)
        for i in range(3):
            x[i] = torch.from_numpy(image)
        model.generate(x)
        mlab.outline()
        mlab.axes()
        mlab.savefig(filename)
        return

    # test_generate(6)
    train_set = torch.from_numpy(np.load(r'./data/train_set.npy'))
    projection_set = Reconstructor.get_projection_set(train_set)
    ind = 666
    fig1, fig2 = model.contrast(
        projection_set[ind], train_set[ind, 0], voxel=True)
    mlab.savefig("contrast1.png", figure=fig1)
    mlab.savefig("contrast2.png", figure=fig1)

    return


if __name__ == "__main__":
    run()
    # study()
