"""
This script contains the code to train a neural network model for sentiment analysis.
"""
import torch
import pandas
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


class NeuralNetwork(nn.Module):
    # Define input size, hidden layer size, and number of output classes
    input_size = 300
    hidden_size1 = 128
    hidden_size2 = 64
    num_classes = 3
    model_path = "dataset/neural_network.pth"
    dropout_prob = 0.25

    def __init__(self, input_size=input_size, num_classes=num_classes, dropout_prob=dropout_prob):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # Fully connected layer 1
        self.fc1 = nn.Linear(16 * (input_size//2), self.hidden_size1)
        self.dropout1 = nn.Dropout(self.dropout_prob)
        # # Fully connected layer 2
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.dropout2 = nn.Dropout(dropout_prob)
        # Fully connected layer 3
        self.fc3 = nn.Linear(self.hidden_size2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))  # Add channel dimension
        # Apply pooling layer
        x = self.pool(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Using ReLU activation function
        x = F.relu(self.fc1(x))
        # Applying dropout to the first hidden layer
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        # Applying dropout to the second hidden layer
        x = self.dropout2(x)
        # No activation function in the last layer
        x = self.fc3(x)
        return x


class TrainDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pandas.read_csv(csv_file)
        self.data['vector'] = self.data['vector'].apply(lambda x: eval(re.sub(r'(?<!\[)\s+(?!\])', ',', x)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vector = self.data.vector.iloc[idx]
        label = self.data.label.iloc[idx]
        return torch.tensor(vector), label


def train():
    # Instantiate the neural network model
    model = NeuralNetwork()
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    # Training the model
    dataset = TrainDataset(f'dataset/market-aux-vectors{NeuralNetwork.input_size}_data.csv')
    train_data, test_data = train_test_split(dataset, test_size=0.9, random_state=42)

    batch_size = 64
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()  # Set the model to train mode
        correct = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the parameters
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()
        accuracy = correct / len(train_data_loader.dataset)
        epoch_loss = running_loss / len(train_data_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}")

    # test
    # Evaluating the model on test data
    test_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")
    torch.save(model.state_dict(), NeuralNetwork.model_path)


def load():
    model = NeuralNetwork()
    state_dict = torch.load(NeuralNetwork.model_path)
    model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    dataset = TrainDataset(f'dataset/market-aux-vectors{NeuralNetwork.input_size}_data.csv')
    dataset[1]
    train()
    print("model")
