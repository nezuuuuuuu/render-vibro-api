import tensorflow as tf
import kagglehub
import numpy as np
import pandas as pd
import os


print("Downloading YAMNet model...")
try:
    # Use the handle that includes the framework and variant
    model_path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")
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



target_sample_rate = 16000

# Option A: Generate a simple sine wave for demonstration
duration_s = 4.0 # 2 seconds long
frequency = 16000 # A4 note
amplitude = 0.5
t = np.linspace(0., duration_s, int(target_sample_rate * duration_s), endpoint=False)
waveform_np = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
print(f"Generated a {duration_s}s sine wave at {frequency} Hz.")


print(f"Waveform shape: {waveform_np.shape}, dtype: {waveform_np.dtype}, Sample rate: {target_sample_rate}")

# Convert the NumPy waveform to a TensorFlow Tensor
waveform_tf = tf.constant(waveform_np, dtype=tf.float32)

# --- 5. Run Inference ---
print("Running inference...")
# The YAMNet model signature takes the waveform tensor directly
try:
    # The output is a dictionary containing scores, embeddings, and log_mel_spectrogram
    output = infer(waveform_tf)
    print(output)
    scores = output['output_0']      # Shape: (N, 521), N = number of frames
    embeddings = output['output_1'] # Shape: (N, 1024)
    spectrogram = output['output_2'] # Shape: (N, 64) - Log mel spectrogram
    print("Inference complete.")
    print(f"Output scores shape: {scores.shape}") # Scores for each class for each time frame
except Exception as e:
    print(f"Error during inference: {e}")
    exit()

mean_scores = tf.reduce_mean(scores, axis=0).numpy() # Average scores over frames
top_n = 10 # Number of top results to display
top_class_indices = np.argsort(mean_scores)[-top_n:][::-1] # Get indices of top N scores

print(f"\n--- Top {top_n} Predicted Audio Events ---")
for i in top_class_indices:
    class_name = class_names[i]
    score = mean_scores[i]
    print(f"- {class_name}: {score:.3f}")
