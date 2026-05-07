import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# Load MNIST Test Data
# -------------------------
transform = transforms.ToTensor()

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# -------------------------
# Model Definition (same architecture used in training)
# -------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# FGSM Attack Function
# -------------------------
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# -------------------------
# Evaluate CLEAN or DEFENDED MODEL
# -------------------------
def evaluate_clean(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return correct / len(loader.dataset)

# -------------------------
# Evaluate UNDER FGSM ATTACK
# -------------------------
def evaluate_fgsm(model, loader, epsilon):
    model.eval()
    correct = 0

    for data, target in loader:
        data.requires_grad = True

        output = model(data)
        loss = F.cross_entropy(output, target)

        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        adv_data = fgsm_attack(data, epsilon, data_grad)

        output = model(adv_data)
        pred = output.argmax(dim=1)

        correct += pred.eq(target).sum().item()

    return correct / len(loader.dataset)

# -------------------------
# Load Models
# -------------------------
def load_model(path):
    model = Net()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

print("\nLoading models...\n")

clean_model = load_model("mnist_model.pth")
defended_model = load_model("mnist_defended_model.pth")

# -------------------------
# Run Evaluations
# -------------------------
epsilon = 0.1

print("Evaluating CLEAN model...")
clean_acc = evaluate_clean(clean_model, test_loader)
clean_attack_acc = evaluate_fgsm(clean_model, test_loader, epsilon)

print("\nEvaluating DEFENDED model...")
defended_acc = evaluate_clean(defended_model, test_loader)
defended_attack_acc = evaluate_fgsm(defended_model, test_loader, epsilon)

# -------------------------
# Results Summary
# -------------------------
print("\n================ RESULTS ================\n")

print(f"Clean Model Accuracy:        {clean_acc * 100:.2f}%")
print(f"Clean Model (FGSM Attack):   {clean_attack_acc * 100:.2f}%")

print("\n--- After Defense ---\n")

print(f"Defended Model Accuracy:     {defended_acc * 100:.2f}%")
print(f"Defended Model (FGSM Attack):{defended_attack_acc * 100:.2f}%")

print("\n========================================\n")