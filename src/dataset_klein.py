import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset import InfiniteSampler


KLEIN_DATASET_NAME = "dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora"
KLEIN_DATASET_CACHE = "/code/dataset/" + KLEIN_DATASET_NAME.split("/")[-1]


########################################################################################################################
#                                      PAIRED IMG2IMG DATASET FOR KLEIN                                               #
########################################################################################################################


class PairedDatasetKlein(Dataset):
    """HuggingFace-backed paired dataset.  Returns pixel tensors in [-1, 1]."""

    def __init__(self, hf_dataset, resolution=512):
        self.dataset = hf_dataset
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        target = example["edited_image"].convert("RGB")
        cond = example["input_image"].convert("RGB")
        return {
            "pixel_values": self.transform(target),
            "cond_pixel_values": self.transform(cond),
        }


def get_loader_klein(batch_size=1, resolution=512):
    hf_dataset = load_dataset(
        KLEIN_DATASET_NAME,
        split="train",
        cache_dir=KLEIN_DATASET_CACHE,
    )
    dataset = PairedDatasetKlein(hf_dataset, resolution=resolution)

    sampler = InfiniteSampler(dataset=dataset, rank=0, shuffle=True, num_replicas=1)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
    )
    return iter(loader), dataset
