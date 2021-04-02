import numpy as np
from tqdm import tqdm
from particle.clump.distTrans import build
from particle.pipeline import Sand
# from mayavi import mlab
from particle.mayaviOffScreen import mlab

data = np.load("data/liutao/v1/particles.npz")["testSet"]
print(data.shape)
for i in range(10, 11):
    cube = data[i, 0]
    lamdas = [0.1, 0.2, 0.3, 0.4, 0.5]
    phis = [90, 105, 130, 145, 160, 175]
    for lamda in lamdas:
        for phi in tqdm(phis):
            c, r = build(cube, lamda, phi)

            # 剔除最小的颗粒
            # mask = r != r.min()
            # c = c[mask]
            # r = r[mask]
            # print(f"Total {len(r)} spheres.")

            fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            mlab.points3d(*c.T, r*2, scale_factor=1, opacity=1,
                          resolution=30, mode="sphere", color=(0.65,)*3)
            sand = Sand(cube)
            sand.visualize(vivid=True, figure=fig,
                           color=(0, 0.6, 0.85), opacity=0.5)
            # mlab.title(f"Total {len(r)} spheres.")
            # mlab.outline()
            # mlab.axes()

            mlab.savefig(f"../myClump/{lamda}-{phi}-{len(r)}.png")
            mlab.close(all=True)
            # mlab.show()
