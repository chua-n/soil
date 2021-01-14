import numpy as np
import pandas as pd
from particle.clump.mghAlgorithm import *

# datafile = r"E:\Code\VAE\data\test_set.npy"
# data = np.load(datafile)
# sandCube = data[46, 0]
sandCube = np.load("./data/special/132.npy")
sand = Sand(sandCube)
sandPointCloud = sand.point_cloud().T
cells = CellCollection(sandPointCloud, nX=20)
scpMatrix = SCPMatrix(cells)
center, radii = scpMatrix.solver()
io = np.hstack([center, radii[:, np.newaxis]])
io = pd.DataFrame(io, columns="x, y, z, r".split(","))
print(io)
io.to_csv("clump.txt", sep="\t", index=False)
scpMatrix.solverTest()
# plotTest(sand, cells)
mlab.show()
