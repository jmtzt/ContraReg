import os
import random

from torch.utils.data import Dataset
from utils import _load3d, _crop_and_pad, _normalise_intensity, _to_tensor


class _BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, data_dir_path):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir_path
        assert os.path.exists(data_dir_path), f"Data dir does not exist: {data_dir_path}"
        self.subject_list = sorted(os.listdir(self.data_dir))
        self.data_path_dict = dict()

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)


class BrainMRInterSubj3D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 crop_size,
                 evaluate=False,
                 modality='t1t2',
                 atlas_path=None,
                 transform=None):
        super(BrainMRInterSubj3D, self).__init__(data_dir_path)
        self.evaluate = evaluate
        self.crop_size = crop_size
        self.img_keys = ['target', 'source']
        self.modality = modality
        self.atlas_path = atlas_path
        self.transform = transform

    def _set_path(self, index):
        # choose the target and source subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        self.src_subj_id = random.choice(self.subject_list)
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        self.data_path_dict['target'] = f'{self.tar_subj_path}/T1_brain.nii.gz'

        # modality
        if self.modality == 't1t1':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T1_brain.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T2_brain.nii.gz'
        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        # eval data
        if self.evaluate:
            # T1w image of source subject for visualisation
            self.data_path_dict['target_original'] = f'{self.src_subj_path}/T1_brain.nii.gz'
            self.img_keys.append('target_original')
            # segmentation
            self.data_path_dict['target_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
            self.data_path_dict['source_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load3d(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = _to_tensor(data_dict)

        if self.transform:
            # TODO: maybe transform the seg too?
            data_dict['source'] = self.transform(data_dict['source'])
            data_dict['target'] = self.transform(data_dict['target'])

        return data_dict