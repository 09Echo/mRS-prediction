import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import os
import torch
import numpy as np
from monai.transforms import (
    OneOf,
    LoadImage,
    Transpose,
    RandGaussianNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
    RandGaussianSharpen,
    RandHistogramShift,
    NormalizeIntensity,
    Rand3DElastic,
    RandAffine,
    Zoom,
    Compose,
)
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader

def z_score(cta_array):
    mask = cta_array != 0  # 生成前景mask
    voxels = cta_array[mask]
    mean = np.mean(voxels)
    std = np.std(voxels)
    result = np.where(cta_array != 0, ((cta_array - mean) / std), 0)
    return result

class ProVe():
    def __init__(self, fold=0, mode='train',n_splits=5):
        self.mode = mode
        self.fold = fold
        self.n_splits = n_splits
        self.split = []
        self.id = []
        data_info_path = "/home/hubin/Datasets/ISLES2024/Text_information.xlsx"
        label = pd.read_excel(data_info_path)[['labels']].values
        data = pd.read_excel(data_info_path)[['id','patient_prompt']].values
        skf = StratifiedKFold(self.n_splits, shuffle=True ,random_state=42)
        ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
        for train_index, test_index in skf.split(data, label):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            X_train, y_train = ros.fit_resample(X_train, y_train)
            self.split.append({'train': {'path': X_train, 'label': y_train},
                               'test': {'path': X_test, 'label': y_test}})

            self.transform = Compose([
                NormalizeIntensity(),
                #lambda x: z_score(x)
            ])

    def __getitem__(self, item):
        data_info = self.split[self.fold][self.mode]
        data_list = data_info['path'][item]
        patient_text = data_list[1]
        img_path = os.path.join("/home/hubin/Datasets/ISLES2024/MRS_DATA/strip_MNI_MIP_40/", data_list[0]+'_MCA.nii.gz')
        itk = sitk.ReadImage(img_path)
        arr = sitk.GetArrayFromImage(itk).astype(float)
        # arr = arr[16:240,16:240] #Vit
        # arr = self.transform(arr)
        # arr = torch.Tensor(arr).unsqueeze(0)
        # arr = np.concatenate((arr, arr, arr), axis=0)
        # label = int(data_info['label'][item].item())
        # return arr, patient_text, diff_text, label ,data_list[0]

        arr = arr[16:240, 16:240]  # Vit
        arr = torch.Tensor(arr).unsqueeze(0)
        arr = np.concatenate((arr, arr, arr), axis=0)
        arr = self.transform(arr) #tra
        label = int(data_info['label'][item].item())
        return arr, patient_text, label, data_list[0]

    def __len__(self):
        return len(self.split[self.fold][self.mode]['label'])

if __name__ == '__main__':
    dataset = ProVe(fold=2, mode='train', n_splits=5)
    train_dl = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False, num_workers=4)
    for step, data in enumerate(train_dl):
        a,b,c,d,e = data
        print(d)