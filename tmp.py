from tqdm import trange
import numpy as np
from skimage import morphology, segmentation, measure
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

file = "./data/all.npy"
data = np.load(file)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
bad = []
for i in trange(len(data)):
    cube = data[213, 0]  # 185
    distance = ndi.distance_transform_edt(cube)
    coords = peak_local_max(distance, min_distance=3, labels=cube)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=cube)
    label, num = measure.label(labels, return_num=True)
    if num > 3:
        bad.append(i)
    if len(bad) % 500 == 0:
        print("Bingo!")
print(bad)
