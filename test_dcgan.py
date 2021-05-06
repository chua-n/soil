import numpy as np
import torch

# from mayavi import mlab
from particle.mayaviOffScreen import mlab
from particle.pipeline import Sand
from particle.nn.dcgan import Generator, generate


def test():
    device = torch.device("cpu")
    xml = "particle/nn/config/dcgan.xml"
    net_G = Generator(xml).to(device)

    # 以下为查看当前生成效果
    checkpoint = torch.load('/home/chuan/soil/output/dcgan/iterD:iterG=1:2/nLatent64/state_dict-25.tar',
                            map_location='cpu')
    net_G.load_state_dict(checkpoint['generator_state_dict'])

    torch.manual_seed(3.14)
    vec = torch.randn(3000, net_G.nLatent, device=device)
    print(vec.shape)

    cubes = generate(net_G, vec)
    print(cubes.shape, cubes.dtype)
    np.save('output/geometry/gan.npy', cubes)
    for cube in cubes:
        sand = Sand(cube)
        # sand.visualize(voxel=True, glyph='sphere', scale_mode='scalar')
        # sand.visualize(vivid=False)
        sand.visualize(vivid=True)
        mlab.outline()
        mlab.axes()
        mlab.show()


def interpolate(z1, z2, xml="particle/nn/config/dcgan.xml"):
    net_G = Generator(xml)
    ckpt = torch.load("output/dcgan/iterD:iterG=1:2/nLatent64/state_dict-25.tar",
                      map_location="cpu")
    net_G.load_state_dict(ckpt['generator_state_dict'])
    betas = [0, 0.2, 0.4, 0.6, 0.8, 1]
    inters = [(1-beta)*z1+beta*z2 for beta in betas]
    cubes = []
    for beta, z in zip(betas, inters):
        cubes.append(generate(net_G, z))
        sand = Sand(cubes[-1])
        sand.cube = sand.poseNormalization()
        sand.visualize(figure=f"{1-beta:.1f}-{beta:.1f}")
        # mlab.outline()
        # mlab.axes()
        mlab.savefig(f"/home/chuan/{1-beta:.1f}-{beta:.1f}.png")
    mlab.show()


def surround(z, xml="particle/nn/config/dcgan.xml", angleInd=1):
    net_G = Generator(xml)
    ckpt = torch.load("output/dcgan/iterD:iterG=1:2/nLatent64/state_dict-25.tar",
                      map_location="cpu")
    net_G.load_state_dict(ckpt['generator_state_dict'])
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
    noises = [torch.normal(0, sigma, size=z.size()) for sigma in sigmas]
    zs = [z+noise for noise in noises]
    cubes = []
    sigmas = [0] + sigmas
    zs = [z] + zs
    for sigma, z in zip(sigmas, zs):
        cubes.append(generate(net_G, z))
        sand = Sand(cubes[-1])
        # sand.cube = sand.poseNormalization()
        if len(cubes) == 1:
            Sand(cubes[0]).visualize(figure=f"{sigma:.1f}",
                                     color=(0.9, 0.9, 0.9), opacity=1)
        elif len(cubes) > 1:
            sand.visualize(figure=f"{sigma:.1f}",
                           color=(0, 0.6, 0.85), opacity=0.9)
            Sand(cubes[0]).visualize(figure=f"{sigma:.1f}",
                                     color=(1, 1, 1), opacity=1)

        views = mlab.view()
        mlab.gcf().scene.camera.zoom(0.8)
        if angleInd == 1:
            mlab.view(azimuth=views[0], elevation=views[1])
        elif angleInd == 2:
            mlab.view(azimuth=views[0]+120, elevation=views[1]+60)
        elif angleInd == 3:
            mlab.view(azimuth=views[0]+240, elevation=views[1]+120)
        else:
            raise ValueError
        mlab.savefig(f"/home/chuan/{angleInd}-{sigma:.1f}.png",
                     magnification=2)
    # mlab.show()


if __name__ == "__main__":
    # generate particles
    # test()

    # test interpolate
    torch.manual_seed(100)
    z1 = torch.randn(64)
    z2 = torch.randn(64)
    interpolate(z1, z2)

    # test surround
    # torch.manual_seed(512)
    # z = torch.randn(64)
    # surround(z, angleInd=1)
    # surround(z, angleInd=2)
    # surround(z, angleInd=3)

    # 这个位姿归一化有点奇葩
    # data = np.load("data/liutao/v1/particles.npz")["testSet"]
    # sand = Sand(data[158, 0])
    # sand.visualize(sand.poseNormalization())
    # mlab.show()
