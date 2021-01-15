import numpy as np
import torch
from particle.nn.gan.dcgan import *
from mayavi import mlab
from particle.pipeline import Sand

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_D = Discriminator().to(device)
net_G = Generator().to(device)

# 以下为查看当前生成效果
checkpoint = torch.load('output/gan/param/state_dict.tar')
net_G.load_state_dict(checkpoint['generator_state_dict'])
net_G.eval()
with torch.no_grad():
    vec = torch.randn(2, n_latent, 1, 1, 1, device=device)
    cubes = net_G(vec).to('cpu').numpy()[:, 0]
    for cube in cubes:
        sand = Sand(cube)
        sand.visualize(realistic=True)
        mlab.outline()
        mlab.axes()
    mlab.show()
