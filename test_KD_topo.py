import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from gtda.homology import VietorisRipsPersistence
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from gtda.diagrams import PersistenceEntropy
from gpu_main import compute_chamfer_loss
from pytorch3d.loss import chamfer_distance


class BaseNet(nn.Module):
    """Base class for neural network models."""

    def save(self, fname):
        # Extract the directory path from the file name
        dir_path = os.path.dirname(fname)

        # Check if the directory path is not empty
        if dir_path:
            # Check if the directory exists, and create it if it does not
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

        # Save the model
        torch.save(self.state_dict(), fname)

    def load(self, fname, device):
        self.load_state_dict(torch.load(fname, map_location=device))
        self.eval()


class CNN(BaseNet):
    """A simple CNN for CIFAR-10 / CIFAR-100."""

    def __init__(self, num_classes, return_latent_feature=False):
        super().__init__()

        self.return_latent_feature = return_latent_feature
        print("=========> Initializing Teacher Model <=========")
        # network layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding="same")
        self.conv2 = nn.Conv2d(32, 32, 3, padding="same")
        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv4 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv5 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv6 = nn.Conv2d(128, 128, 3, padding="same")

        # Poor man's ResNet ...
        # skip connections (learned 1x1 convolutions with stride=1)
        self.skip2 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.skip4 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.skip6 = nn.Conv2d(128, 128, 1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.drop = nn.Dropout(0.25)

    # forward pass of the data "x"
    def forward(self, x):
        # Poor man's ResNet with residual connections

        # For some reason, residual connections work better in this
        # example with relu() applied before the addition and not after
        # the addition as in the original ResNet paper.

        # Input: 3x32x32
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.skip2(x) + F.leaky_relu(self.bn2(self.conv2(x)))  # residual connection
        x = self.pool(x)
        x = self.drop(x)
        # Output: 32x16x16

        # Input: 32x16x16
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.skip4(x) + F.leaky_relu(self.bn4(self.conv4(x)))  # residual connection
        x = self.pool(x)
        x = self.drop(x)
        # Output: 64x8x8

        # Input: 64x8x8
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        latent_feature = x
        x = self.skip6(x) + F.leaky_relu(self.bn6(self.conv6(x)))  # residual connection
        x = self.pool(x)
        x = self.drop(x)
        # Output: 128x4x4

        # Input: 128x4x4

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        # "softmax" activation will be automatically applied in the cross entropy loss below,
        # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        if self.return_latent_feature:
            return x, latent_feature

        return x


class StudentCNN(nn.Module):
    """A smaller CNN student model for knowledge distillation."""

    def __init__(self, num_classes, return_latent_feature=False):
        super().__init__()

        self.return_latent_feature = return_latent_feature

        print("=========> Initializing Student Model <=========")

        # ✅ Reduce channels in convolution layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding="same")  # 32 → 16 channels
        self.conv2 = nn.Conv2d(16, 16, 3, padding="same")  # 32 → 16
        self.conv3 = nn.Conv2d(16, 32, 3, padding="same")  # 64 → 32
        self.conv4 = nn.Conv2d(32, 32, 3, padding="same")  # 64 → 32

        # ✅ Reduce skip connections
        self.skip2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.skip4 = nn.Conv2d(32, 32, 1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        # ✅ Reduce Fully Connected Layer size
        self.fc1 = nn.Linear(32 * 8 * 8, 64)  # 128 → 64 neurons
        self.fc2 = nn.Linear(64, num_classes)

        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        # ✅ First block
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.skip2(x) + F.leaky_relu(self.bn2(self.conv2(x)))  # Residual
        x = self.pool(x)
        x = self.drop(x)

        # ✅ Second block
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.skip4(x) + F.leaky_relu(self.bn4(self.conv4(x)))  # Residual
        latent_feature = x
        x = self.pool(x)
        x = self.drop(x)

        # ✅ Flatten and Fully Connected
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        if self.return_latent_feature:
            return x, latent_feature

        return x


def get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    root = 'data/user/luutunghai@gmail.com/dataset/cifar10'
    # ✅ Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader


def get_num_params():

    """
    Number of parameters - Model A:  573194
    Number of parameters - Model B:  149962
    :return:
    """
    model_A = CNN(num_classes=10)

    model_B = StudentCNN(num_classes=10)

    params_A = sum(p.numel() for p in model_A.parameters() if p.requires_grad)

    params_B = sum(p.numel() for p in model_B.parameters() if p.requires_grad)

    print("Number of parameters - Model A: ", params_A)
    print("Number of parameters - Model B: ", params_B)


def train_teacher(model, train_loader, test_loader, num_epochs=10, lr=0.001, device="cuda"):
    """
    Train the teacher model on CIFAR-10 dataset.

    Args:
        model (nn.Module): The teacher CNN model.
        train_loader (DataLoader): Training dataset.
        test_loader (DataLoader): Testing dataset.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.
        device (str): Device to train on (default: "cuda").

    Returns:
        model (nn.Module): Trained model.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    print("✅ Training Completed!")
    return model


def evaluate(model, test_loader, device="cuda"):
    """
    Evaluate the trained model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test dataset.
        device (str): Device to evaluate on.

    Returns:
        float: Test accuracy.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if model.return_latent_feature:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def train_teacher_model():

    train_loader, test_loader = get_cifar10()

    # ✅ Initialize & Train Teacher Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = CNN(num_classes=10)  # CNN is the Teacher Model

    train_teacher(teacher_model, train_loader, test_loader, num_epochs=10, lr=0.001, device=device)

    save_path = '/home/luutunghai@gmail.com/projects/PTv3-distill/test_topo_train/teacher_weight.pth'

    teacher_model.save(save_path)


# ✅ Define Persistent Homology Function
def compute_ph(data):
    """
    Compute Persistent Homology for input feature maps.
    """
    ph = VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1, 2])
    # data = data.reshape(1, *data.shape)  # Reshape for GTDA
    diagrams = ph.fit_transform(data)
    return diagrams


def test_compute_ph():

    data = torch.randn(64, 32, 16, 16)
    data = data.view(64, 32, -1)

    diagrams = compute_ph(data)

    print(diagrams)
    print(type(diagrams))


def train_KD(alpha=0.5, beta=0.3, T=2.0, num_epochs=10, lr=0.001):

    save_path = '/home/luutunghai@gmail.com/projects/PTv3-distill/test_topo_train/teacher_weight.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = CNN(num_classes=10, return_latent_feature=True).to(device)  # CNN is the Teacher Model
    teacher_model.load(save_path, device=device)
    print("Loaded teacher weight successfully")

    student_model = StudentCNN(num_classes=10, return_latent_feature=True).to(device)

    # 🔹 Define Optimizer & Loss Functions
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()  # Normal classification loss
    criterion_kd = nn.KLDivLoss(reduction="batchmean")  # KD loss
    PE = PersistenceEntropy()

    train_loader, test_loader = get_cifar10()

    print("✅ Starting KD Training...")
    for epoch in range(num_epochs):
        student_model.train()
        total_loss, correct, total = 0, 0, 0

        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # 🔹 Forward pass (Student & Teacher)
            student_logits, student_features = student_model(images)

            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images)

            # 🔹 Compute Losses
            loss_ce = criterion_ce(student_logits, labels)  # Cross-Entropy Loss
            loss_kd = criterion_kd(F.log_softmax(student_logits / T, dim=1),
                                   F.softmax(teacher_logits / T, dim=1)) * (T * T)  # KD Loss

            student_features = student_features.view(64, 32, -1).cpu().detach()
            teacher_features = teacher_features.view(64, 32, -1).cpu().detach()

            student_features_diagrams = compute_ph(student_features)
            teacher_features_diagrams = compute_ph(teacher_features)

            # ✅ Convert Persistent Topology Features to Torch
            student_features_topo = torch.tensor(PE.fit_transform(student_features_diagrams), dtype=torch.float32,
                                                 device="cuda").unsqueeze(0)  # (64, 3)
            teacher_features_topo = torch.tensor(PE.fit_transform(teacher_features_diagrams), dtype=torch.float32,
                                                 device="cuda").unsqueeze(0)  # (64, 3)

            # print(student_features_topo)
            loss_topo, _ = chamfer_distance(student_features_topo, teacher_features_topo)  # Topology Loss

            # print("Loss topo = ", loss_topo)
            # print("Loss KD = ", loss_kd)
            # print("Loss CE = ", loss_ce)
            # 🔹 Combine Losses
            # loss = loss_ce + alpha * loss_kd + beta * loss_topo
            loss = beta * loss_topo + loss_ce
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate(student_model, test_loader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}")

    # 🔹 Save Student Model
    torch.save(student_model.state_dict(),
               "/home/luutunghai@gmail.com/projects/PTv3-distill/test_topo_train/student_weight.pth")
    print("✅ Student Model Saved!")


# train_KD()


