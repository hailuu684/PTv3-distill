import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import clip
from image_dataset import cifar10
from train.train_clip import train, validate


def train_clip():
    # Step 1: Load CLIP model and its preprocess function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    """
    ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    """

    # Step 2: Use CLIP's preprocess function on CIFAR-10 dataset
    batch_size = 20

    root = '/home/luu/DistilledDataset_ContinualLearning/data'
    train_loader, val_loader, _, cifar10_classes = cifar10.get_cifar10_clip(root=root,
                                                                            preprocess=preprocess,
                                                                            batch_size=batch_size,
                                                                            device=device)

    # Step 4: Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(20):
        model.train()
        train(model, train_loader, cifar10_classes, optimizer, epoch, device)

        # val_acc = validate(model, val_loader, text_inputs, device)
        #
        # print(f"Validation Accuracy after Epoch [{epoch + 1}/10]: {val_acc * 100:.2f}%")

    # # Define CrossEntropyLoss function
    # criterion = nn.CrossEntropyLoss()
    #
    # for idx, batch in enumerate(train_loader):
    #     images, labels = batch
    #     images = images.to(device)
    #     labels = labels.to(device)
    #
    #     # Dynamically create text inputs based on labels for this batch
    #     text_inputs = torch.cat([clip.tokenize(f"a photo of a {cifar10_classes[label]}") for label in labels]).to(
    #         device)
    #
    #     # Encode images and text
    #     image_features = model.encode_image(images)
    #     text_features = model.encode_text(text_inputs)
    #
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #
    #     # Compute the logits as the dot product between image and text features
    #     logits_per_image = image_features @ text_features.T
    #     logits_per_text = text_features @ image_features.T
    #
    #     # Contrastive loss
    #     loss_image = criterion(logits_per_image, labels)
    #     loss_text = criterion(logits_per_text, labels)
    #
    #     print(f"Image features = {image_features}")
    #     print(f"Text features = {text_features}")
    #     print(f"logits_per_image = {logits_per_image}")
    #     print(f"logits_per_text = {logits_per_text}")
    #     print(f"loss_image = {loss_image}")
    #     print(f"loss_text = {loss_text}")
    #     print(f"labels = {labels}")
    #
    #     if idx == 2:
    #         break


if __name__ == '__main__':
    train_clip()
