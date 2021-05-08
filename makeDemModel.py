import numpy as np
from skimage import measure
from tqdm import tqdm

from mayavi import mlab
from particle.pipeline import Sand
from particle.utils import plotter
from particle.clump import distTrans


def build():
    num = 1000
    fakeParticles = np.load(
        "/home/chuan/soil/output/tvsnet/vae/通道加多/no-dropout, rotation60, 1*1ConvT-no-bn/state_dict.pt")[:num]
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


def arrange(nLength=8, nWidth=8):
    fakeParticles = np.load("output/clump/pipeline/fakeParticles.npy",
                            allow_pickle=True)
    balls = np.load("output/clump/pipeline/balls.npy")
    clumps = np.load("output/clump/pipeline/clumps.npy",
                     allow_pickle=True)
    nl = nw = 0
    curMaxWidth = curMaxHeight = 0
    lastLengthEnd = lastWidthEnd = lastHeightEnd = 0
    for i, particle in tqdm(enumerate(fakeParticles)):
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


def plotParticles(saveObj=True):
    fakeParticles = np.load("output/clump/pipeline/fakeParticles-move.npy",
                            allow_pickle=True)
    fakeParticles = np.vstack(fakeParticles)
    print(fakeParticles.shape)
    bigCubes = np.zeros((
        fakeParticles[:, 0].max() - fakeParticles[:, 0].min() + 1,
        fakeParticles[:, 1].max() - fakeParticles[:, 1].min() + 1,
        fakeParticles[:, 2].max() - fakeParticles[:, 2].min() + 1
    ), dtype=np.uint8)
    bigCubes[tuple(fakeParticles.T)] = 1
    outermostLayerThickness = 4
    newBigCubes = np.zeros(np.array(bigCubes.shape)+outermostLayerThickness*2)
    newBigCubes[outermostLayerThickness:-outermostLayerThickness,
                outermostLayerThickness:-outermostLayerThickness,
                outermostLayerThickness:-outermostLayerThickness] = bigCubes
    bigCubes = newBigCubes
    del newBigCubes
    print(bigCubes.shape)
    fig = Sand(bigCubes).visualize()
    if saveObj:
        mlab.savefig("output/clump/pipeline/fakeParticles.obj")
    return fig


def plotBalls():
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    balls = np.load("output/clump/pipeline/balls-move.npy")
    # plotter.sphere(balls[:, :3], balls[:, -1])
    plotter.sphere(balls[:, :3], balls[:, -1],
                   color=(0.65, 0.65, 0.65))
    return fig


def plotClumps():
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    clumps = np.load("output/clump/pipeline/clumps-move.npy",
                     allow_pickle=True)
    clumps = np.vstack(clumps)
    plotter.sphere(clumps[:, :3], clumps[:, -1], resolution=15,
                   color=(0.65, 0.65, 0.65))
    return fig


# arrange(8, 8)
# plotParticles(saveObj=False)
# plotBalls()
plotClumps()
mlab.outline()
mlab.show()
