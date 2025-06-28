import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm.notebook import tqdm, trange
import cv2
import numpy as np
from PIL import Image

# transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# loading datasets
data_dir = "/Users/harryhaghani/Desktop/data"

full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = full_dataset.classes
print("Classes:", class_names)  # rock, paper, scissors

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# defining CNN models
class RPS_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 256) # after passing through two pooling layers
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# training the model
model = RPS_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training model...")

for epoch in trange(3):
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        x = images
        y = model(x)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()

# test accuracy
correct = 0
total = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        x = images
        y = model(x)
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

accuracy = correct / total
print(f"Test accuracy: {accuracy:.2f}")

# WEBCAM prediction
model.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam is running — press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("read failed.")
        continue

    frame = cv2.flip(frame, 1)

    # Dynamically center ROI in frame
    height, width, _ = frame.shape
    roi_size = 200
    x = (width - roi_size) // 2
    y = (height - roi_size) // 2

    # Draw ROI box
    cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (255, 0, 0), 2)

    roi = frame[y:y + roi_size, x:x + roi_size]
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    input_tensor = transform(roi_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
        label = class_names[predicted]

    # Display prediction
    cv2.putText(frame, f'Prediction: {label}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Rock Paper Scissors Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

