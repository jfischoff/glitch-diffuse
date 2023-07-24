import torch
import torchaudio
from PIL import Image
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import ffmpeg     
from tqdm import tqdm
# Load the .m4a audio file
waveform, sample_rate = torchaudio.load('separated/htdemucs/in-the-morning/other.wav')

print("Waveform shape:", waveform.shape)
print("Sample rate:", sample_rate)

# Set the window size for the STFT to 1/30 of a second (assuming the audio is sampled at 44.1kHz)
hop_length = int(sample_rate / 30) 
window_size = hop_length*4
print("Window size:", window_size)
print("Hop length:", hop_length)

# Perform the STFT
# stft_transform = torchaudio.transforms.Spectrogram(n_fft=window_size, hop_length=hop_length, power=None)
# spectrogram = stft_transform(waveform)
# 
# print("Spectrogram shape:", spectrogram.shape)
# 
# # Take the magnitude of the spectrogram (convert from complex to real)
# spectrogram_abs = torch.abs(spectrogram)
# 
# # Convert the spectrogram to dB scale
# db_transform = torchaudio.transforms.AmplitudeToDB()
# spectrogram_db = db_transform(spectrogram_abs)
# 
# # Normalize the spectrogram to the range 0-255
# spectrogram_norm = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
# spectrogram_norm = (spectrogram_norm * 255).byte()

# Create MelSpectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_length, n_mels=800)

# Perform the MelSpectrogram transformation
mel_spectrogram = mel_transform(waveform)

print("Mel Spectrogram shape:", mel_spectrogram.shape)

# Convert the Mel spectrogram to dB scale
db_transform = torchaudio.transforms.AmplitudeToDB()
mel_spectrogram_db = db_transform(mel_spectrogram)
# mel_spectrogram_db = mel_spectrogram

# Normalize the Mel spectrogram to the range 0-255
mel_spectrogram_norm = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
mel_spectrogram_norm = (mel_spectrogram_norm)

# Create a directory for the images if it doesn't exist
output_dir = 'spectrum-images'
os.makedirs(output_dir, exist_ok=True)
# Create a black image with height as frequency bins and width as 1920
img_array = np.zeros((mel_spectrogram_norm.shape[1], 1920))

# TODO: 
# for each spectrogram slice
# add it to a black image
# when the image is 1920 pixels wide
# drop the first pixel and add it at the end
# save the image

# reorder the img_array so the high frequencies are at the top
img_array = np.flip(img_array, axis=0)

# The pixel column index for the black image
img_idx = 0
slice_pixel_width = 5

writer = imageio.get_writer('spectrogram-mel-in-the-morning.mp4', fps=30)

# Iterate over the spectrogram slices
for i in tqdm(range(mel_spectrogram_norm.shape[2])):
    # Get the spectrogram slice
    spec_slice = mel_spectrogram_norm[0, :, i:i+1]

    if i*slice_pixel_width+slice_pixel_width > 1920:
        # Drop the first column and add the new slice at the end
        img_array = np.roll(img_array, -slice_pixel_width, axis=1)
        img_array[:, -slice_pixel_width:] = np.repeat(spec_slice, slice_pixel_width, axis=1)
    else:
        # Add the new slice at the current index
        img_array[:, i*slice_pixel_width:i*slice_pixel_width+slice_pixel_width] = np.repeat(spec_slice, slice_pixel_width, axis=1)

    # Convert the numpy array to an image
    img = plt.get_cmap('jet')(img_array)  # Convert grayscale to RGB using 'jet' color map

    # Convert the floating point image in the range [0,1] to uint8 in the range [0,255] and ignore the alpha channel
    img_uint8 = (img[..., :3] * 255).astype(np.uint8)

    # Write the image to the video
    writer.append_data(img_uint8)

# Close the writer
writer.close()

input_video = ffmpeg.input('spectrogram-mel-in-the-morning.mp4')
input_audio = ffmpeg.input('in-the-morning.m4a')

# Get the duration of the video in seconds
video_info = ffmpeg.probe('spectrogram-mel-in-the-morning.mp4')
video_duration = float(video_info['streams'][0]['duration'])

# Trim the audio to match the video duration and add it to the video
output = ffmpeg.output(input_video.video, input_audio.audio.filter('atrim', duration=video_duration), 'spectrogram-mel-in-the-morning-audio.mp4')
ffmpeg.run(output, overwrite_output=True)