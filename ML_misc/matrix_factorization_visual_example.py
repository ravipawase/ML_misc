import numpy as np
import matplotlib.pyplot as plt

# create a 9X10 array
or_arr = np.zeros((9,10))

# add values to crate a simple image
or_arr[6:,:] =10    # soil color
or_arr[:6,:] =20    # sky color
or_arr[5:,4:5] =30  # stem of a tree
or_arr[1:2,3:6] =40 # leaves
or_arr[2:3,2:7] =40 # leaves
or_arr[3:4,1:8] =40 # leaves
or_arr[4:5,2:7] =40 # leaves

# create image by assigning colors
print(or_arr)
Fig = plt.figure(figsize=(45, 10), dpi=100, tight_layout=True)
ax = Fig.add_subplot(1, 1, 1)
ax.imshow(or_arr, cmap='rainbow')
Fig.savefig("/home/ravindra/git_repos/ML_misc/or_arr.png", format='png', dpi=200)


# decompose the original array into component matrices each of dimension 1 i.e. row and column
from sklearn.decomposition import NMF
model = NMF(n_components=1, init='random', random_state=0)
W = model.fit_transform(or_arr)
H = model.components_
print(W)
print(H)
rec_arr = np.dot(W, H)
rec_arr_round =  np.round(rec_arr, 1)
print(rec_arr_round)

#create image by assigning colors
Fig = plt.figure(figsize=(45, 10), dpi=100, tight_layout=True)
ax = Fig.add_subplot(1, 1, 1)
ax.imshow(rec_arr, cmap='rainbow')
Fig.savefig("/home/ravindra/git_repos/ML_misc/rec_array.png", format='png', dpi=200)

