import h5py

from torch.utils.data import Dataset, DataLoader
from ._transforms import build_transforms
from ._samplers import TwoStreamBatchSampler


class ACDCDataSet(Dataset):

    def __init__(self, root_dir=r'F:/datasets/ACDC/', mode='train', num=None, transforms=None):

        self.root_dir = root_dir
        self.mode = mode

        if isinstance(transforms, list):
            transforms = build_transforms(transforms)
        self.transforms = transforms

        names_file = f'{self.root_dir}/train_slices.list' if self.mode == 'train' else f'{self.root_dir}/val_slices.list'

        with open(names_file, 'r') as f:
            self.sample_list = f.readlines()
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        if num is not None and self.mode == 'train':
            self.sample_list = self.sample_list[:num]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(f'{self.root_dir}/data/slices/{case}.h5', 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transforms:
            sample = self.transforms(sample)
        return sample


def get_acdc_loaders(root_dir=r'F:/datasets/ACDC/', labeled_num=7, labeled_bs=12, batch_size=24, batch_size_val=16,
                     num_workers=4, worker_init_fn=None, train_transforms=None, val_transforms=None):
    ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}

    db_train = ACDCDataSet(root_dir=root_dir, mode="train", transforms=train_transforms)
    db_val = ACDCDataSet(root_dir=root_dir, mode="val", transforms=val_transforms)

    if labeled_bs < batch_size:
        labeled_slice = ref_dict[str(labeled_num)]
        labeled_idxs = list(range(0, labeled_slice))
        unlabeled_idxs = list(range(labeled_slice, len(db_train)))
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(db_train, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
