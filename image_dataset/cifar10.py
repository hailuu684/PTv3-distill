from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import clip


def get_cifar10_clip(root, preprocess, batch_size, device):
    # Train and Validation sets
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CIFAR10(root=root, train=False, download=True, transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # CIFAR-10 class names (as text prompts)
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]

    # Step 3: Prepare textual labels for the classes
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10_classes]).to(device)

    return train_loader, val_loader, text_inputs, cifar10_classes

