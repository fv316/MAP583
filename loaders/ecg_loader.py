import os
import numpy as np
import torch
import pandas as pd
from loaders.gdd import GoogleDriveDownloader
import pathlib
import shutil


GOOGLE_FILE_ID = '17Rd4YpGwssSpk4xZAT5AyYskjvs95dAY'
ZIP_NAME = 'ecg_zip.zip'


data_sets = ['mitbih_train.csv', 'mitbih_test.csv']


class ECGLoaderBase(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, idx2label, classnames,
                 sh_classnames, class_importance, num_classes,
                 custom_transforms=None,
                 phase=None):

        self.data_dir = data_dir

        data_dir_parent = pathlib.Path(self.data_dir).parent
        data_dir_name = 'ecg'
        self.data_dir = os.path.join(data_dir_parent, data_dir_name)

        self.split = split
        self.phase = split if phase is None else phase
        self.idx2label = idx2label
        self.classnames = classnames
        self.sh_classnames = sh_classnames
        self.transform = custom_transforms
        self.num_classes = num_classes
        self.class_importance = class_importance
        self.data_sets = data_sets
        self.data, self.labels, self.sampler, self.class_weights = None, None, None, None

        self.read_lists()

    def __getitem__(self, index: int):
        ecg = self.data.iloc[index, :-
                             1].values.astype(np.float32).reshape((1, 187))
        label = self.labels[index]
        if self.transform is not None:
            ecg = self.transform(ecg)
            label = self.transform(label)
        else:
            ecg = torch.tensor(ecg).float()
            label = torch.tensor(label).long()
        return tuple([ecg, label, index])

    def __len__(self):
        return len(self.data)

    def extract_labels(self, data):
        pass

    def read_lists(self):
        if self.split == 'train':
            path = os.path.join(self.data_dir, self.data_sets[0])
        # validation data and test data are the same for the time being
        elif self.split == 'val' or self.split == 'test':
            path = os.path.join(self.data_dir, self.data_sets[1])
        else:
            raise Exception(
                'Please chose a valid split type from ["train", "val", "test"]')

        if not os.path.exists(path):
            self._build_dir(path, True, True, True)

        self.data = pd.read_csv(path, header=None)
        self.labels = self.extract_labels(self.data)
        repartition = self.labels.value_counts()
        self.class_weights = np.array(
            [1./repartition[i] for i in range(self.num_classes)])

    def _build_dir(self, path, unzip, showsize, del_zip):
        parent_path = pathlib.Path(self.data_dir).parent
        name = os.path.basename(self.data_dir)  # always ecg

        GoogleDriveDownloader.download_file_from_google_drive(
            file_id=GOOGLE_FILE_ID, dest_path=os.path.join(parent_path, ZIP_NAME), unzip=unzip, showsize=showsize, del_zip=del_zip)
        extracted_folder = os.path.join(parent_path, 'ecg_data')

        for i in self.data_sets:
            shutil.move(os.path.join(extracted_folder, i),
                        os.path.join(parent_path, name))
        os.rmdir(extracted_folder)

    def get_cb_weights(self, cb):
        label_importance = self.get_label_importance(cb)
        return self.class_weights * label_importance

    def get_sampler_weights(self, sampler):
        self.sampler = sampler
        weights = np.zeros(len(self.data))
        label_importance = self.get_label_importance(self.sampler)

        for i, j in enumerate(self.labels.values):
            weights[i] = self.class_weights[j] * label_importance[j]

        return weights

    def get_label_importance(self, scheme):
        if scheme == 'equal':
            l_importance = np.ones(self.class_importance.size)
        elif scheme == 'importance':
            l_importance = self.class_importance
        else:
            raise 'Sampling strategy {} not available'.format(scheme)

        return l_importance


class ECGLoader(ECGLoaderBase):
    def __init__(self, data_dir, split, **kwargs):
        idx2label = {
            0: 'Normal',
            1: 'Artial Premature',
            2: 'Premature ventricular contraction',
            3: 'Fusion of ventricular and normal',
            4: 'Unknown'
        }

        # class names and short hand class names
        classnames = ['Normal', 'Artial Premature', 'Premature ventricular contraction',
                      'Fusion of ventricular and normal', 'Unknown']
        sh_classnames = ['Normal', 'PAC', 'PVC', 'Fusion', 'Unknown']
        class_importance = np.array([1, 2, 2, 2, 0.5])

        super(ECGLoader, self).__init__(data_dir, split, idx2label,
                                        classnames, sh_classnames, class_importance,
                                        num_classes=5, **kwargs)

    def extract_labels(self, data):
        return data[187].astype(int)


class ECGLoader_bin(ECGLoaderBase):
    def __init__(self, data_dir, split, **kwargs):
        idx2label = {
            0: 'Normal',
            1: 'Abnormal'
        }

        # class names and short hand class names
        classnames = ['Normal', 'Abnormal']
        sh_classnames = ['Normal', 'Abnormal']
        class_importance = np.array([1, 2])

        super(ECGLoader_bin, self).__init__(data_dir, split, idx2label,
                                        classnames, sh_classnames, class_importance,
                                        num_classes=2, **kwargs)

    def extract_labels(self, data):
        return pd.Series(np.where(data[187] > 0, 1, 0))
