import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

        # Dummy input to calculate the size of tensor after conv and pool layers so fc1 gets defined properly
        dummy_input = torch.randn(1, 3, 150, 150)
        dummy_output = self._forward_features(dummy_input)
        self.fc1 = nn.Linear(dummy_output.view(-1).size(0), 512)
        
    def _forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Prepare data
transform = transforms.Compose([
    transforms.Resize((150, 150)), # resize to 150 x 150
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor() # this converts images to pytorch tensors
])

train_data = datasets.ImageFolder(root='data/train', transform=transform) # load training data using the transform thing
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True) # chuck the train data here so you can btach/shuffle it

val_data = datasets.ImageFolder(root='data/val', transform=transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
]))
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False) # same goes for the validation stuff


# Initialising model, loss function and optimiser
model = SimpleCNN()
criterion = nn.BCELoss() # this is the binary cross entropy loss thing
optimizer = optim.Adam(model.parameters(), lr=0.001) # learning rate or 0.001

# Function to evaluate the model on the validation dataset
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            val_loss += loss.item()
            
            # For binary classification, apply threshold of 0.5
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.float().unsqueeze(1)).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = correct / total
    return val_loss, accuracy

# Train the model with validation

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # scheduler to adjust learning rate

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad() # reset gradient thing to zero
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    val_loss, val_accuracy = validate(model, val_loader, criterion)
    
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
    scheduler.step() # step the schedular

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Convert to ONNX format for later
model.eval()
dummy_input = torch.randn(1, 3, 150, 150)
onnx_path = "model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"])
print("Model converted to ONNX yay it workeddd.")
