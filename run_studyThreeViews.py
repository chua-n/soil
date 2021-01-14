import numpy as np
from particle.mayaviOffScreen import mlab
# from mayavi import mlab
import os
import torch
from particle.nn.threeViews import Reconstructor
from skimage import io as skio
# from particle.utils import project, plot_mayavi, plot_vv

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
fig1, fig2 = model.contrast(projection_set[ind], train_set[ind, 0], voxel=True)
mlab.savefig("contrast1.png", figure=fig1)
mlab.savefig("contrast2.png", figure=fig1)
