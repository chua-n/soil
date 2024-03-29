import numpy as np
import torch

from particle.mayaviOffScreen import mlab
from particle.nn.wgan_gp import *


def test():
    from mayavi import mlab
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    xml = "particle/nn/config/wgan_gp.xml"
    net_G = Generator(xml).to(device)

    # 以下为查看当前生成效果
    checkpoint = torch.load(
        '/home/chuan/soil/output/wgan_gp/state_dict.tar', map_location='cpu')
    net_G.load_state_dict(checkpoint['generator_state_dict'])

    torch.manual_seed(3.14)
    vec = torch.randn(3000, net_G.hp['nLatent'], device=device)

    cubes = generate(net_G, vec)
    cubes = cubes.numpy()
    np.save('output/geometry/wgan_gp.npy', cubes)
    # for cube in cubes:
    #     cube[cube <= 0.5] = 0
    #     cube[cube > 0.5] = 1
    #     sand = Sand(cube)
    #     # sand.visualize(voxel=True, glyph='sphere', scale_mode='scalar')
    #     # sand.visualize(vivid=False)
    #     sand.visualize(vivid=True)
    #     mlab.outline()
    #     mlab.axes()
    #     mlab.show()


if __name__ == "__main__":
    # train wgan_gp
    torch.manual_seed(3.14)
    train()
    # test()
