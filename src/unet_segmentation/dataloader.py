from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import typer

app = typer.Typer()


# Dataset class for the PH2 dataset
class PH2dataset(Dataset):
    def __init__(self, source_path, labels_path, transform=None):
        self.source_path = source_path
        self.labels_path = labels_path
        self.transform = transform
        self.source_images = sorted([f for f in os.listdir(source_path) if f.endswith('.bmp')])
        self.label_images = sorted([f for f in os.listdir(labels_path) if f.endswith('lesion.bmp')])

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        source_image = os.path.join(self.source_path, self.source_images[idx])
        label_image = os.path.join(self.labels_path, self.label_images[idx])

        image = Image.open(source_image).convert("RGB")
        label = Image.open(label_image).convert("L")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
    
def load_PH2(resolution, batch_size, source_path, labels_path):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()
       
    ])
    
    dataset = PH2dataset(source_path=source_path, labels_path=labels_path, transform=transform)
    generator = torch.Generator().manual_seed(42)  # Set a manual seed for reproducibility
    train_set, val_set, test_set = random_split(dataset, [int(0.7 * len(dataset)), int(0.2 * len(dataset)), int(0.1 * len(dataset))], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print("Loaded %d training images" % len(train_set))
    print("Loaded %d validation images" % len(val_set))
    print("Loaded %d test images" % len(test_set))

    return train_loader, val_loader, test_loader


@app.command()
def main(
    resolution: int = typer.Option(128, help="Image resolution (pixels)"),
    batch_size: int = typer.Option(16, help="Batch size"),
    source_path: str = typer.Option("data/processed/source", help="Path to source images"),
    labels_path: str = typer.Option("data/processed/labels", help="Path to label images")
):
    """
    Load the PH2 dataset and print the shape of the first batch.
    """
    train_loader, val_loader, test_loader = load_PH2(resolution, batch_size, source_path, labels_path)
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}, Batch of labels shape: {labels.shape}")
        break  # Just to check the first batch


if __name__ == "__main__":
    app()