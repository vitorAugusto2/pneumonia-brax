import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CsvImgData(Dataset):
    def __init__(self, csv_path: str, base_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["path_png"]
        img_path = os.path.join(self.base_dir, rel_path)

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        y = int(row["target"])

        return img, y


def get_loaders(base_dir: str, train_csv: str, val_csv: str, test_csv: str, train_transform, eval_transform,
                batch_size: int,num_workers: int):

    train_ds = CsvImgData(train_csv, base_dir=base_dir, transform=train_transform)
    val_ds   = CsvImgData(val_csv,   base_dir=base_dir, transform=eval_transform)
    test_ds  = CsvImgData(test_csv,  base_dir=base_dir, transform=eval_transform)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders