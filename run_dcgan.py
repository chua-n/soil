import numpy as np
import torch

from particle.mayaviOffScreen import mlab
from particle.pipeline import Sand
from particle.nn.dcgan import Generator, train, generate


def test():
    from mayavi import mlab
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    xml = "particle/nn/config/dcgan.xml"
    net_G = Generator(xml).to(device)

    # 以下为查看当前生成效果
    checkpoint = torch.load(
        '/home/chuan/soil/output/dcgan/state_dict.tar', map_location='cpu')
    # checkpoint = torch.load('/home/chuan/soil/output/gan/nLatent32/param/state_dict.tar', map_location='cpu')
    net_G.load_state_dict(checkpoint['generator_state_dict'])

    torch.manual_seed(3.14)
    vec = torch.randn(3000, net_G.nLatent, device=device)

    cubes = generate(net_G, vec)
    cubes = cubes.numpy()
    np.save('output/geometry/gan.npy', cubes)
    for cube in cubes:
        cube[cube <= 0.5] = 0
        cube[cube > 0.5] = 1
        sand = Sand(cube)
        # sand.visualize(voxel=True, glyph='sphere', scale_mode='scalar')
        # sand.visualize(realistic=False)
        sand.visualize(realistic=True)
        mlab.outline()
        mlab.axes()
        mlab.show()


if __name__ == "__main__":
    # train dcgan
    torch.manual_seed(3.14)
    train()
    # test()
