import matplotlib.pyplot as plt
import librosa.display

def visualize_predictions(audio, pred_piano_roll):
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    librosa.display.specshow(pred_piano_roll, y_axis='cqt_note', cmap=plt.cm.hot)
    plt.title('Predicted MIDI piano roll')
    plt.subplot(212)
    cqt = librosa.amplitude_to_db(librosa.cqt(audio), ref=np.max)
    librosa.display.specshow(cqt, y_axis='cqt_note', x_axis='time', cmap=plt.cm.hot)
    plt.title('Audio CQT')
    plt.show()
