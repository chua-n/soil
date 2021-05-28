import os
import numpy as np
from skimage import io as skio

import torch

from mayavi import mlab
# from particle.mayaviOffScreen import mlab
from particle.pipeline import Sand
from particle.utils.dirty import loadNnData
from particle.nn.tvsnet import VaeTVSNet, TVSNet, getProjections, generateOneParticle, contrast


def loadMoel():
    model = VaeTVSNet(
        "/home/chuan/soil/particle/nn/config/tvsnet-vae-deep.xml")
    # model = TVSNet("/home/chuan/soil/particle/nn/config/tvsnet.xml")
    state_dict = torch.load("/home/chuan/soil/output/tvsnet/vae/网络加深/dropout0.2, 数据增强/精调-rotate0/state_dict-overfit.pt",
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def testSimpleGeometry(ind, model=None):
    cwd = os.getcwd()
    os.chdir('./data/简单几何图形')
    filenames = sorted(os.listdir())
    filename = filenames[ind]
    print(filename)
    image = skio.imread(filename, as_gray=True)
    image[image == 255] = 1
    print(image.max())
    os.chdir(cwd)
    x = torch.empty((3, 64, 64), dtype=torch.uint8)
    for i in range(3):
        x[i] = torch.from_numpy(image)
    cube = generateOneParticle(model, x)
    Sand(cube).visualize()
    mlab.outline()
    mlab.axes()
    mlab.show()
    mlab.savefig(filename)
    return


def loadMyPicture(folder="trial/v1"):
    src = [skio.imread(f"/home/chuan/老板要求/chazhi/4-{i+1}.png",
                       as_gray=True) for i in range(3)]
    src = [src[2], src[0], src[1]]
    src = np.concatenate([view[np.newaxis, :, :] for view in src], axis=0)
    src = torch.from_numpy(src)
    src[src == 255] = 1
    print(src.max(), src.shape)
    return src


def testGenerate(src, model, **kwargs):
    particle = generateOneParticle(model, src)
    sand = Sand(particle)
    from skimage.util import random_noise
    sand.cube = random_noise(
        sand.cube, mode="gaussian", var=0.02, seed=314)
    print("已添加噪声！")
    fig = sand.visualize(**kwargs)
    return fig


if __name__ == "__main__":
    model = loadMoel()
    # 任务1
    # testSimpleGeometry(4, model)

    # 任务2
    src = loadMyPicture(folder="/home/chuan/老板要求/liuyu")
    # src2 = loadMyPicture(folder="v4")
    fig = mlab.figure()
    testGenerate(src, model, figure=fig, color=(0.7, 0.7, 0.7))
    # testGenerate(src2, model, figure=fig, color=(0, 0.6, 0.85), opacity=0.9)
    # mlab.outline()
    # axes = mlab.axes(xlabel='', ylabel='', zlabel='')
    # axes.label_text_property.font_family = "times"
    mlab.show()

    # 任务3
    # particles = loadNnData("/home/chuan/soil/data/liutao/v1/particles.npz",
    #                        "testSet")
    # src = getProjections(particles)
    # for ind in [944, 983, 1218, 1340, 1427]:
    #     testGenerate(src[ind-1], model)
    #     mlab.savefig(f"/home/chuan/{ind}.png")
    #     mlab.savefig(f"/home/chuan/{ind}.obj")
    #     mlab.close(all=True)

    # 任务4
    # ind = 1426
    # testGenerate(src[ind], model)
    # print(src[ind].shape, particles[ind].shape)
    # model.contrast(src[ind], particles[ind, 0])
    # mlab.show()
