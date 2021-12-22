import h5py
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LLD_Dataset(Dataset):
    def __init__(self, transform, text_buffer_to):
        print(f"Beginning assembly of LLD_Dataset from HDF5 file LLD-logo.hdf5")
        self.lld_hdf5 = h5py.File('LLD-logo.hdf5', 'r')
        self.primary_transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(0, 1)])
        self.secondary_transform = transform
        self.text_buffer_to = text_buffer_to

        if len(self.lld_hdf5['meta_data']['names']) != self.lld_hdf5['data'].shape[0]:
            raise RuntimeError(f"The data count does not match the label count. HD5F file likely corrupted or damaged.")

    def __len__(self):
        return len(self.lld_hdf5['meta_data']['names'])

    def __getitem__(self, idx):
        image_array = np.array(self.lld_hdf5['data'][idx, :, :self.lld_hdf5['shapes'][idx][1], :self.lld_hdf5['shapes'][idx][2]])
        data_as_image = PIL.Image.fromarray(image_array.transpose(1, 2, 0).astype('uint8'), 'RGB')
        label = torch.FloatTensor([n/126 for n in self.lld_hdf5['meta_data']['names'][idx]])
        label = torch.cat((label, torch.zeros(50-label.shape[0])))

        if self.secondary_transform is not None:
            return self.primary_transform(self.secondary_transform(data_as_image)), label
        else:
            return self.primary_transform(data_as_image), label



