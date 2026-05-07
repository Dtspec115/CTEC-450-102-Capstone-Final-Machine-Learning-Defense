# CTEC-450-102-Capstone-Final-Machine-Learning-Defense

## Overview

This capstone project explores how machine learning models can be attacked and defended using adversarial machine learning techniques. A neural network classifier was trained on the MNIST handwritten digit dataset, attacked using the Fast Gradient Sign Method (FGSM), and then strengthened using adversarial defense techniques.

The primary objective of this project is to demonstrate that highly accurate machine learning systems remain vulnerable to carefully crafted attacks and that defensive training methods can significantly improve robustness against adversarial manipulation.

---

## Project Objectives

This project includes three major components:

### 1. Build a Machine Learning Model

* Train a neural network classifier using the MNIST dataset
* Achieve high classification accuracy on handwritten digits

### 2. Perform an Adversarial Attack

* Implement the Fast Gradient Sign Method (FGSM)
* Generate adversarial examples that fool the classifier
* Measure the reduction in model accuracy

### 3. Defend the Model

* Implement adversarial training as a defense strategy
* Retrain the model using adversarial examples
* Evaluate improved robustness against attacks

---

## What is FGSM?

The Fast Gradient Sign Method is an adversarial attack technique that modifies input images using gradients from the neural network’s loss function. Even small perturbations can cause neural networks to misclassify images while remaining visually indistinguishable from human perception.

---

# Technologies Used

## Programming Language

* Python 3

## Libraries and Frameworks

* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

# Project Structure

```plaintext
CTEC-450-102-Capstone-Final-Machine-Learning-Defense/
│
├── DT_Finale_Train_Model.py
├── DT_Finale_fgsm_attack.py
├── DT_Finale_fgsm_Defense.py
├── DT_Finale_Evaluation_Script.py
├──Model Data (Previous Training) 
  ├── mnist_model.pth
  └── mnist_defended_model.pth
├── evaluation_graph.png
└── README.md
```

---

# File Descriptions

## `DT_Final_Train_Model.py`

Trains the original neural network classifier using the MNIST dataset.

### Features

* Loads MNIST data
* Builds the neural network
* Trains the classifier
* Saves trained model weights

### Output

```plaintext
mnist_model.pth
```

---

## `DT_Final_fgsm_attack.py`

Implements the FGSM adversarial attack against the trained model.

### Features

* Loads trained model
* Generates adversarial examples
* Tests model robustness
* Displays attack results and visualizations

---

## `DT_Final_fgsm_Defense.py`

Implements adversarial training to improve robustness against attacks.

### Features

* Generates adversarial training samples
* Retrains the model on attacked images
* Saves defended model

### Output

```plaintext
mnist_defended_model.pth
```

---

## `DT_Final_Evaluation_Script.py`

Compares the performance of:

* Original model
* Attacked model
* Defended model

### Features

* Clean accuracy testing
* FGSM attack evaluation
* Defense evaluation
* Final comparison results

---

# Installation Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CTEC-450-102-Capstone-Final-Machine-Learning-Defense.git
```

---

## 2. Navigate to the Project Folder

```bash
cd CTEC-450-102-Capstone-Final-Machine-Learning-Defense
```

---

## 3. Install Dependencies

```bash
pip install torch torchvision matplotlib numpy
```

---

# How to Run the Project

## Step 1 — Train the Model

Run:

```bash
python DT_Final_Train_Model.py
```

### Expected Output

<img width="1910" height="321" alt="Image" src="https://github.com/user-attachments/assets/0405aa07-b614-4a0c-a021-7dc9306fed54" />

---

## Step 2 — Run FGSM Attack

Run:

```bash
python DT_FInal_fgsm_attack.py
```

### Features

* Tests adversarial attacks at different epsilon values
* Displays adversarial examples
* Shows attack impact on model accuracy

---

## Step 3 — Run Defense Training

Run:

```bash
python DT_Finale_fgsm_Defense.py
```

### Expected Output

<img width="1919" height="400" alt="Image" src="https://github.com/user-attachments/assets/190cf056-65cd-483e-b691-9e6fb06d9fca" />

---

## Step 4 — Evaluate Results

Run:

```bash
python DT_Final_Evaluation_Script.py
```

### Example Output

<img width="1396" height="515" alt="Image" src="https://github.com/user-attachments/assets/7d87b4e5-4695-4a36-9715-04eb458f8cfb" />

---

# Results Summary

| Model          | Clean Accuracy | Accuracy Under FGSM |
| -------------- | -------------- | ------------------- |
| Original Model | 97.58%         | 10.35%               |
| Defended Model | 95.07%         | 81.72%              |

These results demonstrate that:

* Neural networks are highly vulnerable to adversarial attacks
* FGSM can dramatically reduce model performance
* Adversarial training significantly improves robustness

---

# Evaluation Graph

<img width="800" height="500" alt="Image" src="https://github.com/user-attachments/assets/ba564e93-e7a4-4bde-9714-070d912dee00" />

---

# Future Improvements

Potential future enhancements include:

* Using stronger attacks such as PGD
* Testing additional defense techniques
* Improving robustness against adaptive attacks
* Deploying models on GPU hardware for faster training

---

# Academic Relevance

This project demonstrates key cybersecurity and artificial intelligence concepts including:

* Adversarial machine learning
* Neural network vulnerabilities
* Defensive AI strategies
* Secure AI system development

The techniques explored in this project are relevant to:

* Autonomous vehicles
* Facial recognition systems
* Malware detection
* Medical AI systems
* Financial fraud detection

---

# References

Hojjat, H. (2018). MNIST dataset. Kaggle. https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Microsoft Research. (2020, June 30). A newly discovered principle reveals how adversarial training enables robust deep learning. Microsoft Research. https://www.microsoft.com/en-us/research/blog/newly-discovered-principle-reveals-how-adversarial-training-can-perform-robust-deep-learning/ (microsoft.com)

Practical DevSecOps. Fast gradient sign method (FGSM). https://www.practical-devsecops.com/glossary/fast-gradient-sign-method-fgsm/

PyTorch Documentation. (2026). PyTorch documentation. PyTorch Foundation. https://pytorch.org/docs/stable/index.html
