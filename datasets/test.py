import h5py
import numpy as np

with h5py.File('/share/zhuoyang/PartNet_ego/Chair-2/train/train-00-0.h5','r') as f:
    A = np.array(f['label'])
    print(np.min(A))
    print(np.max(A))
