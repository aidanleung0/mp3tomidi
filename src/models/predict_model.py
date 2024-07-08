import torch
import os
import json
from torch.utils.data import DataLoader
from src.models.train_model import MusicTranscriptionCNN
from src.data.make_dataset import LMDataset

def load_model(model_path):
    model = MusicTranscriptionCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, dataset):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['audio'].float()
            onset_output, note_output, pitch_output = model(inputs)
            predictions.append((onset_output, note_output, pitch_output))
    return predictions

if __name__ == "__main__":
    # Load the model
    model_path = 'results/mp3_to_midi_model.pth'
    model = load_model(model_path)

    # Load scores.json
    with open(os.path.join('results', 'match_scores.json')) as f:
        scores = json.load(f)

    # Create dataset instance
    dataset = LMDataset(scores, 'data', 'results')

    # Make predictions
    predictions = predict(model, dataset)
    print(predictions)
