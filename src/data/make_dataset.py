import os
import json
import numpy as np
import librosa
import pretty_midi

class LMDataset:
    def __init__(self, scores, data_path, results_path):
        self.scores = scores
        self.data_path = data_path
        self.results_path = results_path
        self.msd_ids = list(scores.keys())

    def __len__(self):
        return len(self.msd_ids)

    def __getitem__(self, idx):
        msd_id = self.msd_ids[idx]
        midi_md5, score = next(iter(self.scores[msd_id].items()))
        aligned_midi_path = self.get_midi_path(msd_id, midi_md5, 'aligned')
        audio_path = self.msd_id_to_mp3(msd_id)

        # Load and preprocess audio
        audio_cqt = self.preprocess_audio(audio_path)
        audio_cqt = np.expand_dims(audio_cqt, axis=0)  # Add channel dimension

        # Load and preprocess MIDI
        midi_piano_roll = self.preprocess_midi(aligned_midi_path)

        sample = {'audio': audio_cqt, 'midi': midi_piano_roll}
        return sample

    def msd_id_to_dirs(self, msd_id):
        """Given an MSD ID, generate the path prefix. E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
        return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

    def msd_id_to_mp3(self, msd_id):
        """Given an MSD ID, return the path to the corresponding mp3"""
        return os.path.join(self.data_path, 'msd', 'mp3', self.msd_id_to_dirs(msd_id) + '.mp3')

    def get_midi_path(self, msd_id, midi_md5, kind):
        """Given an MSD ID and MIDI MD5, return path to a MIDI file. kind should be one of 'matched' or 'aligned'."""
        return os.path.join(self.results_path, 'lmd_{}'.format(kind), self.msd_id_to_dirs(msd_id), midi_md5 + '.mid')

    def preprocess_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=22050)
        cqt = librosa.cqt(y, sr=sr, bins_per_octave=36, n_bins=252)
        cqt = np.abs(cqt)
        return cqt

    def preprocess_midi(self, midi_path):
        pm = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = pm.get_piano_roll(fs=22050)
        piano_roll = piano_roll[12:96]  # Use 7 octaves starting from C1
        return piano_roll

# Load scores.json
with open(os.path.join('results', 'match_scores.json')) as f:
    scores = json.load(f)

# Create dataset instance
dataset = LMDataset(scores, 'data', 'results')
