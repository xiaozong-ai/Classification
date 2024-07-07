import torch
from torch.utils.data.dataset import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, img_path_list: list = None, label_list: list = None, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, item):
        img = Image.open(self.img_path_list[item])
        ann = self.label_list[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, ann

    @staticmethod
    def collate_fn(batch):
        images, anns = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        anns = torch.as_tensor(anns)

        return images, anns
