import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# Load MNIST Test Dataset
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
# Define Neural Network
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
# Load Trained Model
# -------------------------
model = Net()

model.load_state_dict(torch.load("mnist_model.pth"))

model.eval()

print("Model loaded successfully.\n")

# -------------------------
# FGSM Attack Function
# -------------------------
def fgsm_attack(image, epsilon, data_grad):

    sign_data_grad = data_grad.sign()

    perturbed_image = image + epsilon * sign_data_grad

    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

# -------------------------
# Test Function
# -------------------------
def test(model, test_loader, epsilon):

    correct = 0

    adv_examples = []

    for data, target in test_loader:

        data.requires_grad = True

        output = model(data)

        init_pred = output.max(1, keepdim=True)[1]

        # Skip if already wrong
        if init_pred.item() != target.item():
            continue

        loss = F.cross_entropy(output, target)

        model.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        # Create adversarial image
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify perturbed image
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1

        # Save some examples for visualization
        if len(adv_examples) < 5:

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()

            adv_examples.append(
                (init_pred.item(), final_pred.item(), adv_ex)
            )

    final_acc = correct / float(len(test_loader))

    print(f"Epsilon: {epsilon} \t Test Accuracy = {final_acc * 100:.2f}%")

    return final_acc, adv_examples

# -------------------------
# Run FGSM Attacks
# -------------------------
epsilons = [0, 0.05, 0.1]

examples = []

accuracies = []

for eps in epsilons:

    acc, ex = test(model, test_loader, eps)

    accuracies.append(acc)

    examples.append(ex)

# -------------------------
# Plot Accuracy vs Epsilon
# -------------------------
plt.figure(figsize=(8, 5))

plt.plot(epsilons, accuracies, marker='o')

plt.title("Model Accuracy Under FGSM Attack")

plt.xlabel("Epsilon")

plt.ylabel("Accuracy")

plt.grid(True)

plt.show()

# -------------------------
# Show Adversarial Examples
# -------------------------
count = 0

plt.figure(figsize=(10, 8))

for i in range(len(epsilons)):

    for j in range(len(examples[i])):

        count += 1

        plt.subplot(len(epsilons), 5, count)

        plt.xticks([], [])

        plt.yticks([], [])

        orig, adv, ex = examples[i][j]

        plt.title(f"{orig} -> {adv}")

        plt.imshow(ex, cmap="gray")

plt.tight_layout()

plt.show()