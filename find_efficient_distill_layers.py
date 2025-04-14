import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from test_KD_topo import CNN, StudentCNN


# Assume CNN and StudentCNN classes are defined as provided earlier

class FeatureHook:
    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output


# KD Loss (Feature Matching + Classification)
def kd_loss(student_logits, teacher_logits, student_feats, teacher_feats, alpha=0.3, temperature=4.0):
    cls_loss = F.cross_entropy(student_logits, torch.argmax(teacher_logits, dim=1))
    distill_loss = F.mse_loss(student_feats, teacher_feats)
    return alpha * cls_loss + (1 - alpha) * distill_loss


# Load CIFAR-10
def get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    return train_loader, test_loader


# Evaluation Function
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)[0]
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100


def main(teacher_layer, student_layer, epochs=20, save_student_path=None):
    save_path = '/home/luutunghai@gmail.com/projects/PTv3-distill/test_topo_train/teacher_weight.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = CNN(num_classes=10, return_latent_feature=True).to(device)
    teacher_model.load_state_dict(torch.load(save_path, map_location=device))
    print("Loaded teacher weight successfully")

    student_model = StudentCNN(num_classes=10, return_latent_feature=True).to(device)

    optimizer = optim.Adam(student_model.parameters(), lr=0.005)
    train_loader, test_loader = get_cifar10()

    teacher_acc = evaluate(teacher_model.eval(), test_loader)
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")

    t_hook = FeatureHook()
    s_hook = FeatureHook()
    teacher_hook = getattr(teacher_model, teacher_layer).register_forward_hook(t_hook.hook_fn)
    student_hook = getattr(student_model, student_layer).register_forward_hook(s_hook.hook_fn)

    # Define channel projection based on teacher layer
    channel_map = {"conv2": 32, "conv4": 64, "conv6": 128}
    proj_conv = nn.Conv2d(32, channel_map[teacher_layer], 1).cuda()  # From student 32 channels to teacher channels
    proj_conv.train()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            with torch.no_grad():
                t_logits, _ = teacher_model(x)
            s_logits, _ = student_model(x)

            t_feats = t_hook.features
            s_feats = s_hook.features
            if t_feats.shape[2:] != s_feats.shape[2:]:
                s_feats = F.interpolate(s_feats, size=t_feats.shape[2:], mode='bilinear', align_corners=False)
            s_feats = proj_conv(s_feats)  # Project student features to match teacher channels

            loss = kd_loss(s_logits, t_logits, s_feats, t_feats)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    teacher_hook.remove()
    student_hook.remove()

    if save_student_path:
        torch.save(student_model.state_dict(), save_student_path)
        print(f"Saved trained student model to {save_student_path}")

    return evaluate(student_model, test_loader), teacher_model, test_loader, student_model


# Mutual Information Between Layer and Input/Output
def compute_layer_mi(teacher, layers, loader, num_bins=20, pca_components=50):
    teacher.eval()
    hooks = {layer: FeatureHook() for layer in layers}
    hook_handles = {layer: getattr(teacher, layer).register_forward_hook(hooks[layer].hook_fn) for layer in layers}

    all_inputs = []
    all_labels = []
    all_features = {layer: [] for layer in layers}

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            teacher(x)

            all_inputs.append(x.view(x.size(0), -1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            for layer in layers:
                if hooks[layer].features is None:
                    raise ValueError(f"Features not captured for layer {layer}")
                feats = hooks[layer].features.view(hooks[layer].features.size(0), -1).cpu().numpy()
                all_features[layer].append(feats)

    for layer in layers:
        hook_handles[layer].remove()

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    for layer in layers:
        all_features[layer] = np.concatenate(all_features[layer], axis=0)

    # PCA reduction for inputs and features
    pca_input = PCA(n_components=min(pca_components, all_inputs.shape[1]))
    input_reduced = pca_input.fit_transform(all_inputs)

    mi_results = {}
    for layer in layers:
        pca_layer = PCA(n_components=min(pca_components, all_features[layer].shape[1]))
        layer_reduced = pca_layer.fit_transform(all_features[layer])

        # MI with Input (use first PCA component)
        input_bins = np.histogram_bin_edges(input_reduced[:, 0], bins=num_bins)
        layer_bins = np.histogram_bin_edges(layer_reduced[:, 0], bins=num_bins)
        input_discrete = np.digitize(input_reduced[:, 0], input_bins[:-1])
        layer_discrete = np.digitize(layer_reduced[:, 0], layer_bins[:-1])
        mi_input = mutual_info_score(input_discrete, layer_discrete)

        # MI with Output (labels are already discrete)
        label_bins = np.arange(all_labels.max() + 2)  # Discrete labels 0-9
        label_discrete = np.digitize(all_labels, label_bins[:-1])
        mi_output = mutual_info_score(layer_discrete, label_discrete)

        mi_results[layer] = {"mi_input": mi_input, "mi_output": mi_output}

    return mi_results


# Save Results to Text File
import os
from datetime import datetime


def save_results_to_file(student_results, mi_results, filename=".logs/MI_results.txt"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Distillation Results - Generated on {timestamp}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Teacher Layer':<15} {'Accuracy (%)':<15} {'MI Input':<15} {'MI Output':<15} {'Efficiency':<15}\n")
        f.write("-" * 80 + "\n")

        best_layer = None
        best_score = -float('inf')
        for layer in student_results:
            acc = student_results[layer]['accuracy']
            mi_in = mi_results[layer]['mi_input']
            mi_out = mi_results[layer]['mi_output']
            efficiency = mi_out / max(mi_in, 1e-6)  # Avoid division by zero
            f.write(f"{layer:<15} {acc:<15.2f} {mi_in:<15.4f} {mi_out:<15.4f} {efficiency:<15.4f}\n")

            # Weighted score: 70% accuracy, 30% efficiency
            score = 0.7 * (acc / 100) + 0.3 * efficiency
            if score > best_score:
                best_score = score
                best_layer = layer

        f.write("-" * 80 + "\n")
        f.write(f"Recommended Layer for Distillation: {best_layer} (Score: {best_score:.4f})\n")
    print(f"Results saved to {filename}")


# Test KD and Compute MI
teacher_layers = ["conv2", "conv4", "conv6"]
student_layer = "conv4"
student_results = {}
mi_results = {}

# Train students and get accuracy
for t_layer in teacher_layers:
    save_student_path = f"./logs/student_distilled_from_{t_layer}.pth"
    acc, teacher, test_loader, trained_student = main(t_layer, student_layer, epochs=20,
                                                      save_student_path=save_student_path)
    student_results[t_layer] = {"accuracy": acc}
    print(f"Layer {t_layer}: Accuracy = {acc:.2f}%")

# Compute MI for all teacher layers at once
mi_results = compute_layer_mi(teacher, teacher_layers, test_loader)

# Print Results
for layer in teacher_layers:
    acc = student_results[layer]['accuracy']
    mi_in = mi_results[layer]['mi_input']
    mi_out = mi_results[layer]['mi_output']
    print(f"Teacher {layer}: Acc = {acc:.2f}%, MI Input = {mi_in:.4f}, MI Output = {mi_out:.4f}")

# Save results
save_results_to_file(student_results, mi_results)