import os
import numpy as np
import torch

from particle.utils.dirty import loadNnData
from particle.nn.tvsnet import VaeTVSNet, getProjections


def study():
    from mayavi import mlab
    from skimage import io as skio
    model = VaeTVSNet("./particle/nn/config/tvsnet.xml")
    state_dict = torch.load('output/tvsnet/none/v1/state_dict.pt',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
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
        mlab.show()
        mlab.savefig(filename)
        return

    # test_generate(4)
    source_set = loadNnData("data/liutao/v1/particles.npz", "testSet")
    projection_set = getProjections(source_set)
    for ind in range(200, 300):
        fig1, fig2 = model.contrast(
            projection_set[ind], source_set[ind, 0], voxel=True)
        mlab.show()
    # mlab.savefig("contrast1.png", figure=fig1)
    # mlab.savefig("contrast2.png", figure=fig2)

    return


if __name__ == "__main__":
    study()