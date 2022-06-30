from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class BaseDataSet(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> T_co:
        pass

    def __len__(self):
        pass
