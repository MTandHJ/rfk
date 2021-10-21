

from typing import Optional, Tuple, List
from torch.utils.data import Dataset
import torchvision.transforms as T



AUGMENTATIONS = {
    'none' : lambda x: x,
    'tensor': T.ToTensor(),
    'cifar': T.Compose((
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        )),
}



class OrderTransform:

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, data: Tuple) -> List:
        return [transform(item) for item, transform in zip(data, self.transforms)]


class WrapperSet(Dataset):

    def __init__(
        self, dataset: Dataset,
        transforms: Optional[str] = None
    ) -> None:
        """
        Args:
            dataset: dataset;
            transforms: string spilt by ',', such as "tensor,none'
        """
        super().__init__()

        self.data = dataset

        counts = len(self.data[0])
        if transforms is None:
            transforms = ['none'] * counts
        else:
            transforms = transforms.split(',')
        self.transforms = [AUGMENTATIONS[transform] for transform in transforms]
        if counts == 1:
            self.transforms = self.transforms[0]
        else:
            self.transforms = OrderTransform(self.transforms)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        data = self.data[index]
        return self.transforms(data)

