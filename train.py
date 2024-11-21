import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# At the top of your script with other imports
from sklearn.metrics import accuracy_score
import argparse

# Add the calculate_accuracy function after imports and before training loop
def calculate_accuracy(model, data_loader, device):
    # ... function code as provided ...
# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Load data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

def get_device(force_cpu=False):
    """Setup device agnostic code"""
    if force_cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, test_loader, num_epochs, device, target_accuracy=96.0):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Calculate accuracy
        train_accuracy = calculate_accuracy(model, train_loader, device)
        test_accuracy = calculate_accuracy(model, test_loader, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Testing Accuracy: {test_accuracy:.2f}%')
        
        if test_accuracy >= target_accuracy:
            print(f'Reached target accuracy of {target_accuracy}%!')
            break

def main():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--target-accuracy', type=float, default=96.0, help='Target accuracy to achieve')
    args = parser.parse_args()

    # Setup device
    device = get_device(args.cpu)
    print(f'Using device: {device}')

    # Load your data
    train_loader, test_loader = load_data(batch_size=args.batch_size)
    
    # Initialize model
    model = SimpleCNN()  # Replace with your model class
    
    # Train
    train(model, train_loader, test_loader, args.epochs, device, args.target_accuracy)

if __name__ == '__main__':
    main()
    torch.save(model.state_dict(), "model.pth") 