import click
import joblib

import numpy as np

import torch

from transformers import Wav2Vec2Processor, Wav2Vec2Model

import matplotlib.pyplot as plt


class AudioClassifier:
    def __init__(self, model_id, classifier_path, sampling_rate=16000):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2Model.from_pretrained(model_id)
        self.clf = joblib.load(classifier_path)
        self.sampling_rate = sampling_rate


    def classify_audio(self, audio_data):
        inputs = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors='pt', padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        features = outputs['last_hidden_state']
        features = np.mean(features.detach().cpu().numpy(), axis=1)[0]
        features = features.reshape(1, -1)
        label = self.clf.predict(features)[0]

        return label


    def visualize_audio(self, audio_data):
        plt.figure(figsize=(10, 4))
        plt.plot(audio_data)
        plt.title('Audio Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.show()


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--model_id', default='voidful/wav2vec2-xlsr-multilingual-56', help='Pretrained model ID')
@click.option('--classifier_path', default='audio_classifier.joblib', help='Path to the trained classifier')
@click.option('--visualize', is_flag=True, help='Visualize the audio waveform')
def main(audio_file, model_id, classifier_path, visualize):
    audio_data = np.load(audio_file)

    audio_classifier = AudioClassifier(model_id, classifier_path)

    label = audio_classifier.classify_audio(audio_data)
    click.echo(f'Predicted label: {label}')

    if visualize:
        audio_classifier.visualize_audio(audio_data)


if __name__ == '__main__':
    main()
