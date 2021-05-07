import os
import numpy as np
from skimage import io as skio

import torch

from mayavi import mlab
# from particle.mayaviOffScreen import mlab
from particle.nn.tvsnet import VaeTVSNet


def loadMoel():
    model = VaeTVSNet("./particle/nn/config/tvsnet-vae-deep.xml")
    state_dict = torch.load('/home/chuan/soil/output/tvsnet/vae/网络加深/dropout0.5/state_dict-overfit.pt',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def testSimpleGeometry(ind, model=None):
    cwd = os.getcwd()
    os.chdir('./data/简单几何图形')
    filenames = sorted(os.listdir())
    filename = filenames[ind]
    print(filename)
    image = skio.imread(filename, as_gray=True)
    image[image == 255] = 1
    print(image.max())
    os.chdir(cwd)
    x = torch.empty((3, 64, 64), dtype=torch.uint8)
    for i in range(3):
        x[i] = torch.from_numpy(image)
    model.generate(x)
    mlab.outline()
    mlab.axes()
    mlab.show()
    mlab.savefig(filename)
    return


def testGenerate(model):
    src = [skio.imread(f"{i+1}.png", as_gray=True) for i in range(3)]
    src = [src[2], src[0], src[1]]
    src = np.concatenate([view[np.newaxis, :, :] for view in src], axis=0)
    src = torch.from_numpy(src)
    src[src == 255] = 1
    print(src.max(), src.shape)
    model.generate(src)
    mlab.outline()
    mlab.axes()
    mlab.show()


if __name__ == "__main__":
    model = loadMoel()
    # testSimpleGeometry(4, model)
    testGenerate(model)
