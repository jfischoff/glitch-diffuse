import torch
import torchaudio
import numpy as np
from PIL import Image
import glob
import os
from tqdm import tqdm

# List of wave files and image directories
wave_files = ['separated/htdemucs/lawyers-in-love/bass.wav', 
              'separated/htdemucs/lawyers-in-love/drums.wav', 
              'separated/htdemucs/lawyers-in-love/other.wav', 
              'separated/htdemucs/lawyers-in-love/vocals.wav']
image_dirs = ['images-3', 'images-4', 'images-5', 'images-6']


output_dir = 'blended-images'

os.makedirs(output_dir, exist_ok=True)
# Find data of wave files
audio_data = []
for wave_file in wave_files:
    waveform, sample_rate = torchaudio.load(wave_file)
    # combine the two channels
    waveform = torch.mean(waveform, dim=0)
    audio_data.append(waveform.squeeze())

# Assume that all audio files and image sequences are of the same length
# Find the length of one sequence (or audio file) by counting files in the first directory
sequence_length = len(glob.glob(os.path.join(image_dirs[0], '*.png')))

scales = [6.0, 4.0, 7.0, 6.0]

# Iterate over every frame
for frame_index in tqdm(range(sequence_length)):
    # Load one image from each directory
    frame = f"{frame_index:08d}.png"
    images = [Image.open(os.path.join(dir, f"{frame}")) for dir in image_dirs]

    # Compute the average absolute volume over the 1/30 second interval
    start = (frame_index * sample_rate) // 30
    end = ((frame_index + 1) * sample_rate) // 30
    
    volumes = []
    for data in audio_data:
      vabs = torch.abs(data[start:end])
      vmean = torch.mean(vabs)
      volumes.append(vmean)


    

    # Multiply this value by their respective image and blend using additive alpha
    size = images[0].size
    blended_image = np.zeros((*images[0].size[::-1], 3), dtype=np.float32)

    # Add each image to the blended image, scaling by the volume
    for scale, image, volume in zip(scales, images, volumes):
        image_array = (np.array(image) / 255.0) * volume.item() * scale
        blended_image += image_array

    # Convert the blended image back to an Image object, scaling values to the range [0, 255]
    blended_image = (np.clip(blended_image, a_min=0, a_max=1) * 255).astype(np.uint8)
    blended_image = Image.fromarray(blended_image)

    # Save the composited image
    blended_image.save(f"{output_dir}/{frame_index:08d}.png")
