import torch
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from unet_baseline import UNet
from watershed_baseline import watershed_segmentation


class RandomDataset(Dataset):
    def __init__(self, num_samples, height, width):
        self.num_samples = num_samples
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        random_tensor = torch.rand(1, self.height, self.width)
        random_mask = torch.randint(0, 2, (self.height, self.width), dtype=torch.long)

        return random_tensor, random_mask


if __name__ == "__main__":

    random_dataset = RandomDataset(num_samples=50, height=512, width=512)
    dataloader = DataLoader(random_dataset, shuffle=True)

    # Deep Learning Method
    model = UNet()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1
    )
    trainer.fit(model,dataloader)

    # Classical Watershed
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Processing", unit="batch")):
        for idx in range(1):
            segmented_image = watershed_segmentation(images[idx])

