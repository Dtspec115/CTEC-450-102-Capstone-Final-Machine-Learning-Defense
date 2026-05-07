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
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# -------------------------
# Build Neural Network
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
# Initialize Model
# -------------------------
model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

# -------------------------
# Train Model
# -------------------------
print("Starting Training...\n")

for epoch in range(5):

    running_loss = 0.0

    for data, target in train_loader:

        optimizer.zero_grad()

        output = model(data)

        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/5, Loss: {running_loss:.4f}")

# -------------------------
# Test Accuracy
# -------------------------
correct = 0

with torch.no_grad():

    for data, target in test_loader:

        output = model(data)

        pred = output.argmax(dim=1)

        correct += pred.eq(target).sum().item()

accuracy = correct / len(test_loader.dataset)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# -------------------------
# Save Model
# -------------------------
torch.save(model.state_dict(), "mnist_model.pth")

print("\nModel saved as mnist_model.pth")