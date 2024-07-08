import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import json
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from src.data.make_dataset import LMDataset

class MusicTranscriptionCNN(nn.Module):
    def __init__(self):
        super(MusicTranscriptionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 63 * 125, 128)
        self.fc2_onset = nn.Linear(128, 88)
        self.fc2_note = nn.Linear(128, 88)
        self.fc2_pitch = nn.Linear(128, 88)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 63 * 125)
        x = F.relu(self.fc1(x))
        onset_output = self.sigmoid(self.fc2_onset(x))
        note_output = self.sigmoid(self.fc2_note(x))
        pitch_output = self.sigmoid(self.fc2_pitch(x))
        return onset_output, note_output, pitch_output

def train_model():
    # Load scores.json
    with open(os.path.join('results', 'match_scores.json')) as f:
        scores = json.load(f)

    # Create dataset instance
    dataset = LMDataset(scores, 'data', 'results')

    # Split the dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = MusicTranscriptionCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            inputs = batch['audio'].float()
            labels = batch['midi'].float()

            optimizer.zero_grad()

            onset_output, note_output, pitch_output = model(inputs)
            loss_onset = criterion(onset_output, labels)
            loss_note = criterion(note_output, labels)
            loss_pitch = criterion(pitch_output, labels)
            loss = loss_onset + loss_note + loss_pitch

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs = batch['audio'].float()
                labels = batch['midi'].float()
                onset_output, note_output, pitch_output = model(inputs)
                loss_onset = criterion(onset_output, labels)
                loss_note = criterion(note_output, labels)
                loss_pitch = criterion(pitch_output, labels)
                loss = loss_onset + loss_note + loss_pitch
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Validation Loss after epoch [{epoch+1}/{num_epochs}]: {val_loss:.4f}')

    print('Finished Training')
    return model

if __name__ == "__main__":
    trained_model = train_model()
    torch.save(trained_model.state_dict(), 'results/mp3_to_midi_model.pth')
