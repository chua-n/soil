import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mayavi import mlab

from particle.nn.vae import VAE

n_latent = 8
vae = VAE(n_latent, 40)
checkpoint = torch.load(
    '../params/state_dict({}, 40, 1e-3, 200, 128).tar'.format(n_latent), map_location=torch.device('cpu'))
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

source = torch.from_numpy(np.load('../data/samples.npy'))

torch.manual_seed(4)
coding = torch.zeros(1, n_latent)
print(coding)
# vae.random_generate(coding, voxel=True)
fig1 = vae.random_generate(coding, opacity=1)
coding[0,4:5] = 1
print(coding)
fig2 = vae.random_generate(coding, color=(0, 1, 1), opacity=0.8)
mlab.show()

# np.random.seed(31425)
# for i in range(10):
#     index = np.random.randint(0, source.shape[0])
#     sample = source[index:index+1].to(dtype=torch.float)
#     print('Sample shape:', sample.shape)
#     fig1, fig2 = vae.contrast(sample)
#     time.sleep(1)
#     mlab.savefig('../result/{}-original.png'.format(i+1), figure=fig1)
#     mlab.savefig('../result/{}-fake.png'.format(i+1), figure=fig2)
#     mlab.close(all=True)
