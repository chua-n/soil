import os
from multiprocessing import Pool, Process
from tqdm import trange

import numpy as np
from particle.mayaviOffScreen import mlab
from particle.pipeline import Sand

gan = "/home/chuan/soil/output/geometry/generatedParticles/gan.npy"
vae = "/home/chuan/soil/output/geometry/generatedParticles/vae.npy"


def display(dataFile, savePath, thrd=0.5):
    data = np.load(dataFile)
    for i in trange(len(data)):
        cube = data[i]
        cube[cube > thrd] = 1
        cube[cube <= thrd] = 0
        Sand(cube).visualize(vivid=True)
        # mlab.outline()
        # mlab.axes()
        mlab.savefig(os.path.join(savePath, f"{i+1:04d}.png"))
        mlab.close()
    return


ganSavePath = "/home/chuan/soil/output/geometry/generatedParticles/gan"
vaeSavePath = "/home/chuan/soil/output/geometry/generatedParticles/vae"

# 使用Pool
pool = Pool(2)
pool.apply_async(display, args=(vae, vaeSavePath))
pool.apply_async(display, args=(gan, ganSavePath))
pool.close()
pool.join()

# 使用Process
# pGan = Process(target=display, args=(gan, vaeSavePath))
# pVae = Process(target=display, args=(vae, vaeSavePath))
# pGan.start()
# pVae.start()
# pGan.join()
# pVae.join()

print("All tasks have been completed!")
