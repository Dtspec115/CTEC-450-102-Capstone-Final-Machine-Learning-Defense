import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# Load MNIST Dataset
# -------------------------
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# -------------------------
# Model (same as before)
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

model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# FGSM Attack Function
# -------------------------
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# -------------------------
# Adversarial Training
# -------------------------
def train_adversarial(model, loader, epsilon):
    model.train()

    for epoch in range(8):
        total_loss = 0

        for data, target in loader:

            data.requires_grad = True

            # Forward pass (clean)
            output = model(data)
            loss = loss_fn(output, target)

            # Backprop to get gradients
            optimizer.zero_grad()
            loss.backward()

            data_grad = data.grad.data

            # Create adversarial examples
            adv_data = fgsm_attack(data, epsilon, data_grad)

            # Train on adversarial examples
            optimizer.zero_grad()
            output_adv = model(adv_data)
            loss_adv = loss_fn(output_adv, target)

            loss_adv.backward()
            optimizer.step()

            total_loss += loss_adv.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -------------------------
# Evaluation Function
# -------------------------
def test_model(model, loader):
    model.eval()

    correct = 0

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / len(loader.dataset)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# -------------------------
# Run Defense Training
# -------------------------
epsilon = 0.1  # attack strength to defend against

print("Starting Adversarial Training...\n")
train_adversarial(model, train_loader, epsilon)

print("\nTesting Model After Defense...\n")
test_model(model, test_loader)

# -------------------------
# Save defended model
# -------------------------
torch.save(model.state_dict(), "mnist_defended_model.pth")

print("\nDefended model saved as mnist_defended_model.pth")