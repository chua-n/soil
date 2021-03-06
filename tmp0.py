from tqdm import trange
import numpy as np
from skimage import measure

file = "data/all.npy"
data = np.load(file)
# print(data.shape)
# mask = np.ones(len(data), dtype=np.bool)
# for i in trange(len(data)):
#     cube = data[i, 0]
#     labels, num = measure.label(cube, connectivity=2, return_num=True)
#     if num > 1:
#         mask[i] = False
# print(np.sum(mask == False))

train = "data/train_set.npy"
train = np.load(train)
test = "data/test_set.npy"
test = np.load(test)
rest = "data/the_rest.npy"
rest = np.load(rest)
flag_train = (data[:len(train)] == train).all()
flag_test = (data[len(train):len(train)+len(test)] == test).all()

flag_rest = data[len(train)+len(test):] == rest
flags = np.ones(len(rest), dtype=np.bool)
for i, flag in enumerate(flag_rest):
    flags[i] = flag.all()

print(flag_train, flag_test, flag_rest, sep="\n")
