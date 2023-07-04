import os
import pickle

import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([transforms.Normalize(mean=[0, 0], std=[1, 1])])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([transforms.Normalize(mean=[0, 0], std=[1, 1])])  # transform it into a torch tensor


class TrainDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, dataset_path, h, w):
        self.plk_dict = pickle.load(open(os.path.join(dataset_path, 'optical_field.pickle'), 'rb'))
        self.sources = list(self.plk_dict.keys())

        self.h = h
        self.w = w

        self.fetcher = nori.Fetcher()

        self.transformer = train_transformer

    def __len__(self):
        # return size of dataset
        return len(self.sources)

    def __getitem__(self, idx):
        try:
            gyro_frames = self.fetcher.get(self.sources[idx])
            cc9 = self.fetcher.get(self.plk_dict[self.sources[idx]])

            gyro_frames = self.bytes2np(gyro_frames, c=8, h=self.h, w=self.w)
            cc9 = self.bytes2np(cc9, c=2, h=self.h, w=self.w)

            gyro = gyro_frames[:2]
            frames = gyro_frames[2:]

            gyro = np.ascontiguousarray(gyro, dtype=np.float32)
            cc9 = np.ascontiguousarray(cc9, dtype=np.float32)
            frames = np.ascontiguousarray(frames, dtype=np.float32)
        except IndexError as e:
            raise e

        return gyro, cc9, frames

    def bytes2np(self, data, c=2, h=270, w=360):
        data = np.fromstring(data, np.float32)
        data = data.reshape((c, h, w))
        return data


class TestDataset(Dataset):
    def __init__(self, dataset_path, h=270, w=360):
        self.plk_dict = np.load(dataset_path, allow_pickle=True)
        self.h = h
        self.w = w

    def __len__(self):
        # return size of dataset
        print(f"TestDataset length is {len(self.plk_dict)}")
        return len(self.plk_dict)

    def __getitem__(self, idx):
        buf = self.plk_dict[idx]
        gyro = buf["gyro"].squeeze()
        frames = buf["frames"].squeeze()

        gyro = np.ascontiguousarray(gyro, dtype=np.float32)
        frames = np.ascontiguousarray(frames, dtype=np.float32)

        return gyro, frames


def fetch_dataloader(types, data_dir, params):
    dataloaders = {}

    for split in ['train', 'valid', 'test']:
        if split in types:
            dataset_path = os.path.join(data_dir, "{}_optical_field/{}".format(split, split))

            if split == 'train':
                dl = DataLoader(TrainDataset(dataset_path, 270, 360),
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=params.num_workers,
                                drop_last=True)
            elif split == 'test':
                dl = DataLoader(TestDataset("/data/cs230-code-examples/pytorch/vision/TestDatasetItem.npy"),
                                batch_size=1,
                                shuffle=False,
                                num_workers=params.num_workers)
            else:
                raise Exception('split type is wrong')

            dataloaders[split] = dl

    return dataloaders
