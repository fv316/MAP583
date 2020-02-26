import os
import numpy as np
import torch
import pandas as pd
from loaders.gdd import GoogleDriveDownloader
# from gdd import GoogleDriveDownloader
import pathlib


GOOGLE_FILE_ID = '17Rd4YpGwssSpk4xZAT5AyYskjvs95dAY'
ZIP_NAME = r'../ecg_zip.zip'


idx2label = {
    0: 'Normal',
    1: 'Artial Premature',
    2: 'Premature ventricular contraction',
    3: 'Fusion of ventricular and normal',
    4: 'Fusion of paced and normal'
}

classnames = ['Normal','Artial Premature','Premature ventricular contraction','Fusion of ventricular and normal','Fusion of paced and normal']


class ECGLoader(torch.utils.data.Dataset):

    def __init__(self, data_dir, split, custom_transforms=None, list_dir=None,
                 crop_size=None, num_classes=5, phase=None):
        
        self.data_dir = data_dir
        self.split = split
        self.phase = split if phase is None else phase
        self.idx2label = idx2label
        self.classnames = classnames
        self.transform = custom_transforms
        self.num_classes = num_classes
        self.data, self.ecg_list, self.label_list = None, None, None
        self.read_lists()


    def __getitem__(self, index : int):
        ecg = self.data.iloc[index, :-1].values.astype(np.float32).reshape((1, 187))
        label = self.data.iloc[index, -1]
        if self.transform is not None:
            ecg = self.transform(ecg)
            label = self.transform(label)
        else:
            ecg = torch.tensor(ecg).float()
            label = torch.tensor(label).long()
        return tuple([ecg, label, index])


    def __len__(self):
        return len(self.data)


    def read_lists(self):
        path = None
        if self.split == 'train':
            path = os.path.join(self.data_dir, 'mitbih_train.csv')
        elif self.split == 'val':
            path = os.path.join(self.data_dir, 'mitbih_test.csv')
        else:
            raise Exception('Please chose a valid split type from ["train", "val"]')            
            
        if not os.path.exists(path):
            parent_path = os.path.join(pathlib.Path(self.data_dir).parent, 'ecg_zip.zip')            
            name = os.path.basename(self.data_dir)

            GoogleDriveDownloader.download_file_from_google_drive(
                file_id=GOOGLE_FILE_ID, dest_path=parent_path, unzip=True, showsize=True, del_zip=True)
            
            os.rename(os.path.join(pathlib.Path(self.data_dir).parent, 'ecg_data'), 
                      os.path.join(pathlib.Path(self.data_dir).parent, name))

        self.data = pd.read_csv(path)
