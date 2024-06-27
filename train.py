import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader import train_dataset, val_dataset, test_dataset

# Create the directory for saving models
os.makedirs('./SavedModels/', exist_ok=True)

# Hyperparameters
num_classes = 7
batch_size = 32
n_epochs = 50

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define and Train the Classifier with Transfer Learning
class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinLesionClassifier, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

classifier = SkinLesionClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(n_epochs):
    classifier.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item()}")

# Evaluate the classifier on validation set
classifier.eval()
all_preds = []
all_labels = []
all_logits = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = classifier(imgs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(outputs.cpu().numpy())

# Debugging statement: Print shapes
print(f"all_preds shape: {torch.tensor(all_preds).shape}")
print(f"all_labels shape: {torch.tensor(all_labels).shape}")

# Ensure the tensor is in the correct shape for softmax
all_preds_tensor = torch.tensor(all_preds, dtype=torch.float)
if all_preds_tensor.ndim == 1:
    all_preds_tensor = all_preds_tensor.unsqueeze(0)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
roc_auc = roc_auc_score(all_labels, all_logits, multi_class='ovo')

print(f'Validation Accuracy: {accuracy * 100}%')
print(f'Validation Precision: {precision}')
print(f'Validation Recall: {recall}')
print(f'Validation F1 Score: {f1}')
print(f'Validation ROC-AUC Score: {roc_auc}')

# Test the Classifier
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

all_test_preds = []
all_test_labels = []
all_test_logits = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = classifier(imgs)
        _, predicted = torch.max(outputs.data, 1)
        all_test_preds.extend(predicted.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())
        all_test_logits.extend(outputs.cpu().numpy())

# Ensure the tensor is in the correct shape for softmax
all_test_preds_tensor = torch.tensor(all_test_preds, dtype=torch.float)
if all_test_preds_tensor.ndim == 1:
    all_test_preds_tensor = all_test_preds_tensor.unsqueeze(0)

test_accuracy = accuracy_score(all_test_labels, all_test_preds)
test_precision = precision_score(all_test_labels, all_test_preds, average='macro')
test_recall = recall_score(all_test_labels, all_test_preds, average='macro')
test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
test_roc_auc = roc_auc_score(all_test_labels, all_test_logits, multi_class='ovo')

print(f'Test Accuracy: {test_accuracy * 100}%')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
print(f'Test F1 Score: {test_f1}')
print(f'Test ROC-AUC Score: {test_roc_auc}')
