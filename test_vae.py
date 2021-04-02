import numpy as np
import torch
from mayavi import mlab

from particle.pipeline import Sand
from particle.nn.vae import Vae


def test():
    device = torch.device("cpu")
    xml = "particle/nn/config/vae.xml"
    vae = Vae(xml).to(device)
    ckpt = torch.load("output/vae/nLatent64/state_dict.pt", map_location="cpu")
    vae.load_state_dict(ckpt)
    torch.manual_seed(3.14)
    vec = torch.randn(3000, vae.decoder.nLatent, device=device)
    print(vec.shape)
    cubes = vae.generate(vec)
    print(cubes.shape, cubes.dtype)

    np.save("output/geometry/vae.npy", cubes)
    for cube in cubes:
        sand = Sand(cube)
        sand.visualize(vivid=True)
        mlab.outline()
        mlab.axes()
        mlab.show()


if __name__ == "__main__":
    # generate particles
    test()
