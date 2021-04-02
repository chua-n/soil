import torch
from skimage import io
from particle.mayaviOffScreen import mlab
from particle.nn.infogan import generate, train, Generator
from particle.pipeline import Sand


def test():
    xml = "particle/nn/config/infogan.xml"
    ckpt_dir = "output/infogan/state_dict.tar"
    net_G = Generator(xml)
    ckpt = torch.load(ckpt_dir, map_location='cpu')
    net_G.load_state_dict(ckpt['netG_state_dict'])
    vector = torch.randn(net_G.nLatent)
    cube = generate(net_G, vector)
    sand = Sand(cube)
    sand.visualize(voxel=True, glyph="sphere")
    # mlab.points3d(cube, mode="point")
    mlab.savefig('333.png')
    img = mlab.screenshot()
    io.imsave('666.png', img)


if __name__ == "__main__":
    torch.manual_seed(3.14)
    train()
    # test()
