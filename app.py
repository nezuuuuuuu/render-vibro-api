from flask import Flask
from IPython import display
import tensorflow as tf
import kagglehub
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
from flask import render_template
from flask import request, jsonify
import os
app = Flask(__name__)
try:
    # Use the handle that includes the framework and variant
    model_path = "model"
    print("Path to model files:", model_path)
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Please ensure you have kagglehub installed and configured (e.g., logged in via kaggle CLI).")
    exit() # Exit if download fails


print("Loading YAMNet model from disk...")
try:

    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure TensorFlow is installed correctly and the downloaded path is valid.")
    exit() 


class_map_path = os.path.join(model_path, "assets\\yamnet_class_map.csv")
try:
    class_names_df = pd.read_csv(class_map_path)
    class_names = class_names_df.sort_values('index')['display_name'].tolist()
    print(f"Loaded {len(class_names)} class names.")
except FileNotFoundError:
    print(f"Error: Class map file not found at {class_map_path}")
    print("Please check the contents of the downloaded model directory.")
    exit()
except Exception as e:
    print(f"Error reading class map file: {e}")
    exit()



SAMPLE_RATE = 16000  # Match your model's expected sample rate
DURATION = 1  # Seconds per recording
FRAME_LENGTH = SAMPLE_RATE * DURATION  # Samples per frame

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    audio_file =  request.files['audio']
    print()

    if not audio_file:
        return "No audio file uploaded", 400

    # Load and resample audio
    y, sr = librosa.load(audio_file, sr=None)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)

    # Run inference
    output = infer(waveform)
    print(output)
    scores = output['output_0']    
    embeddings = output['output_1'] 
    spectrogram = output['output_2'] 
    class_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(class_scores)
    inferred_class = class_names[int(top_class)]
    top_score = float(class_scores[top_class])

    print(f'[YAMNet] The main sound is: {inferred_class} ({top_score})')

    return jsonify({
       
        "inferred_class": inferred_class,
        "confidence": top_score
    })

@app.route("/hehe")
def hehe():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run()
