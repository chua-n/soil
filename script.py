import numpy as np

from mayavi import mlab

from particle.pipeline import Sand

data = np.load('output/geometry/generated_particles/vae.npy')

data[data<=0.5] = 0
data[data>0.5] = 1
print(data.shape)

for i in range(len(data)):
    cube = data[i, 0]
    sand = Sand(cube)
    sand.visualize()
    mlab.outline()
    mlab.axes()
    mlab.title('particle '+str(i))
    mlab.show()

