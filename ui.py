import tkinter as tk
from tkinter import Canvas

import sounddevice as sd
import numpy as np

import threading
import random
import os

import soundfile as sf
import librosa

from joblib import load
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import torch


class AudioClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Fidia')
        self.root.resizable(False, False)
        
        self.canvas = Canvas(root, width=512, height=512)
        self.canvas.pack()
        
        self.record_button = tk.Button(root, text='Record')
        self.record_button.pack()

        self.record_button.bind('<ButtonPress>', self.start_recording)
        self.record_button.bind('<ButtonRelease>', self.stop_recording)
        
        self.current_color = self.random_color()
        self.current_shape = None
        self.recording = False
        self.audio_data = np.array([], dtype=np.float32)
        self.actual_samplerate = None
        
        self.clf = load('audio_classifier.joblib')
        model_id = 'voidful/wav2vec2-xlsr-multilingual-56'
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2Model.from_pretrained(model_id)
        self.sampling_rate = 16000
        
        self.save_path = 'recordings/'


    def start_recording(self, event=None):
        if not self.recording:
            self.recording = True
            self.audio_data = np.array([], dtype=np.float32)
            threading.Thread(target=self.record_audio).start()


    def stop_recording(self, event=None):
        self.recording = False


    def random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))


    def record_audio(self):
        stream = sd.InputStream(callback=self.audio_callback)
        
        with stream:
            self.actual_samplerate = stream.samplerate
            
            while self.recording:
                sd.sleep(100)
        
        self.classify_and_update()


    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.recording:
            self.audio_data = np.concatenate((self.audio_data, indata[:frames, 0]))
        else:
            raise sd.CallbackStop


    def classify_and_update(self):
        if self.audio_data.size > 0:
            category = self.classify_audio(self.audio_data)
            self.update_canvas(category)
            self.audio_data = np.array([], dtype=np.float32)


    def classify_audio(self, audio_data):
        audio_data = librosa.resample(audio_data, orig_sr=self.actual_samplerate, target_sr=self.sampling_rate)

        os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.join(self.save_path, 'latest_recording.wav')
        sf.write(file_path, audio_data, samplerate=self.sampling_rate)
        
        inputs = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors='pt', padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        features = outputs['last_hidden_state']
        features = np.mean(features.detach().cpu().numpy(), axis=1)[0]
        features = features.reshape(1, -1)
        
        label = self.clf.predict(features)[0]
        print(label)
        
        return label

    def update_canvas(self, category):
        if category not in ['color', 'flip']:
            self.canvas.delete('shape')
        
        width, height = 200, 200
        x, y = self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2
        
        if category == 'circle':
            self.current_shape = 'circle'
            self.canvas.create_oval(x - width//2, y - height//2, x + width//2, y + height//2, fill=self.current_color, tags='shape')
        elif category == 'square':
            self.current_shape = 'square'
            self.canvas.create_rectangle(x - width//2, y - height//2, x + width//2, y + height//2, fill=self.current_color, tags='shape')
        elif category == 'triangle':
            self.current_shape = 'triangle'
            points = [x, y - height//2, x - width//2, y + height//2, x + width//2, y + height//2]
            self.canvas.create_polygon(points, fill=self.current_color, tags='shape')
        elif category == 'color':
            self.current_color = self.random_color()
            item = self.canvas.find_withtag('shape')
            if item:
                self.canvas.itemconfig(item[0], fill=self.current_color)
        elif category == 'flip':
            item = self.canvas.find_withtag('shape')
            
            if item:
                if self.current_shape == 'triangle':
                    points = self.canvas.coords(item[0])
                    mid_y = self.canvas.winfo_height() / 2
                    new_points = [points[0], 2 * mid_y - points[1],
                                points[2], 2 * mid_y - points[3],
                                points[4], 2 * mid_y - points[5]]
                    self.canvas.coords(item[0], new_points)
                elif self.current_shape in ['circle', 'square']:
                    bbox = self.canvas.bbox(item[0])
                    mid_y = self.canvas.winfo_height() / 2
                    dist_to_mid = mid_y - (bbox[1] + bbox[3]) / 2
                    self.canvas.move(item[0], 0, 2 * dist_to_mid)
        elif category == 'delete':
            self.canvas.delete('shape')

if __name__ == '__main__':
    root = tk.Tk()

    app = AudioClassifierApp(root)
    
    root.mainloop()
