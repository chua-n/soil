import random
import numpy as np
from skimage import measure
from tqdm import tqdm

from mayavi import mlab
from particle.pipeline import Sand
from particle.utils import plotter
from particle.clump import distTrans


def build():
    num = 1000
    fakeParticles = np.load("output/geometry/generatedParticles/gan.npy")[:num]
    fakeParticlesTmp = []
    balls = []
    clumps = []
    for particle in tqdm(fakeParticles):
        labeled = measure.label(particle, background=0)
        props = measure.regionprops(labeled)
        props.sort(key=lambda prop: prop.area)
        particle = props[-1].image
        fakeParticlesTmp.append(particle)

        # 外接球
        sphere = Sand(particle).circumscribedSphere()
        balls.append([*sphere[1], sphere[0]])

        # 计算clump
        clump = distTrans.build(particle, 0.4, 160)
        clump = np.hstack([clump[0], clump[1][:, np.newaxis]])
        clumps.append(clump)

    fakeParticles = fakeParticlesTmp
    del fakeParticlesTmp
    fakeParticles = np.array([Sand(particle).toCoords()
                              for particle in fakeParticles], dtype=object)
    np.save("output/clump/pipeline/fakeParticles.npy", fakeParticles)
    np.save("output/clump/pipeline/balls.npy", balls)
    np.save("output/clump/pipeline/clumps.npy", clumps)
    return


def arrange(nLength=7, nWidth=7):
    fakeParticles = np.load("output/clump/pipeline/fakeParticles.npy",
                            allow_pickle=True)
    balls = np.load("output/clump/pipeline/balls.npy")
    clumps = np.load("output/clump/pipeline/clumps.npy",
                     allow_pickle=True)
    nl = nw = 0
    curMaxWidth = curMaxHeight = 0
    lastLengthEnd = lastWidthEnd = lastHeightEnd = 0
    for i, particle in enumerate(fakeParticles):
        sizeL = particle[:, 0].max() - particle[:, 0].min()
        sizeW = particle[:, 1].max() - particle[:, 1].min()
        sizeH = particle[:, 2].max() - particle[:, 2].min()

        particle[:, 0] += lastLengthEnd
        particle[:, 1] += lastWidthEnd
        particle[:, 2] += lastHeightEnd

        balls[i][0] += lastLengthEnd
        balls[i][1] += lastWidthEnd
        balls[i][2] += lastHeightEnd

        clumps[i][:, 0] += lastLengthEnd
        clumps[i][:, 1] += lastWidthEnd
        clumps[i][:, 2] += lastHeightEnd

        if nl < nLength - 1:
            lastLengthEnd += sizeL
            nl += 1
        else:
            lastLengthEnd = 0
            nl = 0
            # 第二维度也将在这步发生变化
            nw += 1
            lastWidthEnd += curMaxWidth
            curMaxWidth = 0

        if nw < nWidth:
            curMaxWidth = max(sizeW, curMaxWidth)
        else:
            lastWidthEnd = 0
            nw = 0

        if nl == nw == 0:
            lastHeightEnd += curMaxHeight
            curMaxHeight = 0
        else:
            curMaxHeight = max(curMaxHeight, sizeH)

    np.save("output/clump/pipeline/fakeParticles-move.npy", fakeParticles)
    np.save("output/clump/pipeline/balls-move.npy", balls)
    np.save("output/clump/pipeline/clumps-move.npy", clumps)
    return


def plotParticles():
    fakeParticles = np.load("output/clump/pipeline/fakeParticles-move.npy",
                            allow_pickle=True)
    fig = mlab.figure()
    for particle in fakeParticles:
        mlab.points3d(particle[:, 0],
                      particle[:, 1],
                      particle[:, 2],
                      mode="point", color=(random.random(), random.random(), random.random()))
    return fig


def plotBalls():
    fig = mlab.figure()
    balls = np.load("output/clump/pipeline/balls-move.npy")
    plotter.sphere(balls[:, :3], balls[:, -1])
    return fig


def plotClumps():
    fig = mlab.figure()
    clumps = np.load("output/clump/pipeline/clumps-move.npy",
                     allow_pickle=True)
    clumps = np.vstack(clumps)
    plotter.sphere(clumps[:, :3], clumps[:, -1], resolution=15)
    mlab.outline()
    mlab.axes()
    return fig


# build()
# arrange()
plotParticles()
# plotBalls()
# plotClumps()
mlab.show()
