import numpy as np
from particle.pipeline import Sand
from mayavi import mlab

# data = np.load("output/geometry/generatedParticles/vae.npy")
# initInd = 281
# for i, particle in enumerate(data[initInd:initInd+30]):
#     Sand(particle).visualize()
#     mlab.savefig(f"/home/chuan/obj/{i+1:02d}.obj")
#     mlab.close(all=True)

data = np.load("data/liutao/v1/particles.npz")["trainSet"]
print(data.shape)
initInd = 114
for i, particle in enumerate(data[initInd:initInd+30]):
    Sand(particle[0]).visualize()
    mlab.savefig(f"/home/chuan/obj/{i+1:02d}.obj")
    mlab.savefig(f"/home/chuan/obj/{i+1:02d}.png")
    mlab.close(all=True)
