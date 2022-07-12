from torch.utils.data import Dataset
import os, glob
import h5py
import torch
import numpy as np

class PartNet_ego(Dataset):
    def __init__(self, data_root, mode = 'train'):
        super(PartNet_ego, self).__init__()

        self.mode = mode
        self.len = 0
        self.clips_ego = []
        self.clips_trans = []
        self.labels = []

        search_path = os.path.join(data_root,mode,'*.h5')
        self.h5_files = glob.glob(search_path)
        self.h5_files.sort()

        for file in self.h5_files:
            with h5py.File(file,"r") as f:
                clips_ego = f['ego_data']
                clips_trans = f['trans_data']
                labels = f['label']
                self.clips_ego.append(np.array(clips_ego))
                self.clips_trans.append(np.array(clips_trans))
                self.labels.append(np.array(labels))
                self.len += f['label'].shape[0]
        

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            if idx <= 20479:
                fir, sec = idx // 1024, idx % 1024
            else:
                idx = idx - 20480
                fir, sec = idx // 393, idx % 393
                fir += 20
        if self.mode == 'test':
            if idx <= 2047:
                fir, sec = idx // 1024, idx % 1024
            else:
                idx = idx - 2048
                fir, sec = idx // 193, idx % 193
                fir += 2

        clip_ego = torch.tensor(self.clips_ego[fir][sec],dtype=torch.float32)
        clip_trans = torch.tensor(self.clips_trans[fir][sec],dtype=torch.float32)
        label = torch.tensor(self.labels[fir][sec],dtype=torch.long)

        return clip_ego, clip_trans, label


if __name__ == '__main__':
    dataset = PartNet_ego(data_root='/share/zhuoyang/PartNet_ego/Chair-2',mode='test')
    print(dataset.__len__())
    # clip_ego, clip_trans, label = dataset[20]
    # print(clip_ego.dtype)
    # print(clip_trans.shape)
    # print(label.shape)
    # clip_ego, clip_trans, label = dataset[30]
    # print(clip_ego.shape)
    # print(clip_trans.shape)
    # print(label.shape)
    # clip_ego, clip_trans, label = dataset[200]
    # print(clip_ego.shape)
    # print(clip_trans.shape)
    # print(label.shape)
    # clip_ego, clip_trans, label = dataset[1200]
    # print(clip_ego.shape)
    # print(clip_trans.shape)
    # print(label.shape)

