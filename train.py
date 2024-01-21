import click

import numpy as np

import torch

from datasets import load_dataset, Audio

from joblib import dump
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from transformers import Wav2Vec2Processor, Wav2Vec2Model


class AudioClassifier:
    def __init__(self, model_id, data_dir, labels_allowed, sampling_rate=16000):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2Model.from_pretrained(model_id)
        self.sampling_rate = sampling_rate
        self.dataset = load_dataset('audiofolder', data_dir=data_dir, split='train')
        self.dataset = self.dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))
        self.labels_allowed = labels_allowed


    def preprocess_function(self, dataset):
        audio_array = dataset['audio']['array']
        inputs = self.processor(audio_array, sampling_rate=self.sampling_rate, return_tensors='pt', padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        features = outputs['last_hidden_state']

        return {'features': np.mean(features.detach().cpu().numpy(), axis=1)[0]}


    def encode_audio(self):
        return self.dataset.map(self.preprocess_function, batched=False)


    def prepare_data(self, audios_encoded):
        x = np.array([element['features'] for element in audios_encoded])
        y = [Path(element['audio']['path']).stem.split('-')[0] for element in self.dataset]

        return train_test_split(x, y, test_size=0.2)


    def train_model(self, x_train, y_train, classifier=RandomForestClassifier()):
        classifier.fit(x_train, y_train)

        return classifier


    def evaluate_model(self, classifier, x_test, y_test):
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=self.labels_allowed)

        return accuracy, cm


    def save_model(self, classifier, filename):
        dump(classifier, filename)


@click.command()
@click.option('--data-dir', default='data/', help='Directory containing the audio data.')
@click.option('--model-id', default='voidful/wav2vec2-xlsr-multilingual-56', help='Pretrained model ID.')
@click.option('--labels', default='circle,square,triangle,delete,color,flip', help='Comma-separated list of labels.')
def main(data_dir, model_id, labels):
    labels_allowed = labels.split(',')
    
    audio_classifier = AudioClassifier(model_id, data_dir, labels_allowed)
    audios_encoded = audio_classifier.encode_audio()
    
    x_train, x_test, y_train, y_test = audio_classifier.prepare_data(audios_encoded)
    
    classifier = audio_classifier.train_model(x_train, y_train)
    accuracy, cm = audio_classifier.evaluate_model(classifier, x_test, y_test)
    
    print(f'Accuracy: {accuracy}')
    
    ConfusionMatrixDisplay(cm, display_labels=labels_allowed).plot()
    audio_classifier.save_model(classifier, 'audio_classifier.joblib')


if __name__ == '__main__':
    main()
