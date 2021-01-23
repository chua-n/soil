import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from particle.nn.wgan import *


def run():
    mlab.options.offscreen = True
    torch.manual_seed(3.14)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xml = "particle/nn/config/wgan.xml"
    net_D = Critic(xml)
    net_G = Generator(xml)
    hp = net_D.hp
    source_path = '/home/chuan/soil/data/train_set.npy'
    source = torch.from_numpy(np.load(source_path))
    train_set = DataLoader(TensorDataset(source),
                           batch_size=hp['bs'], shuffle=True)

    train(net_D, net_G, train_set, device, img_dir="output/wgan/process",
          ckpt_dir="output/wgan/param", log_dir="output/log")
    return


def test():
    from mayavi import mlab
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    xml = "particle/nn/config/wgan.xml"
    net_G = Generator(xml).to(device)

    # 以下为查看当前生成效果
    checkpoint = torch.load(
        '/home/chuan/soil/output/wgan/param/state_dict.tar', map_location='cpu')
    net_G.load_state_dict(checkpoint['generator_state_dict'])

    torch.manual_seed(3.14)
    vec = torch.randn(3000, net_G.hp['nLatent'], device=device)

    cubes = generate(net_G, vec)
    cubes = cubes.numpy()
    np.save('output/geometry/wgan.npy', cubes)
    # for cube in cubes:
    #     cube[cube <= 0.5] = 0
    #     cube[cube > 0.5] = 1
    #     sand = Sand(cube)
    #     # sand.visualize(voxel=True, glyph='sphere', scale_mode='scalar')
    #     # sand.visualize(realistic=False)
    #     sand.visualize(realistic=True)
    #     mlab.outline()
    #     mlab.axes()
    #     mlab.show()


if __name__ == "__main__":
    # run()
    test()
