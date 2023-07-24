from pathlib import Path
import torch
import torchaudio
from PIL import Image
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import ffmpeg     
from tqdm import tqdm
from diffusers import (StableDiffusionPipeline, 
                       StableDiffusionImg2ImgPipeline, 
                       DDIMScheduler)
from PIL import Image
import tomesd
from BatchEulerAncestralDiscreteScheduler import EulerAncestralDiscreteScheduler

def embed_prompt(pipe, prompt, device='cpu'):
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = pipe.tokenizer.batch_decode(
            untruncated_ids[:, pipe.tokenizer.model_max_length - 1 : -1]
        )

    attention_mask = None

    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

   

# Load the seperated audio file
waveform, sample_rate = torchaudio.load('in-the-morning-trim.m4a')

print("Waveform shape:", waveform.shape)
print("Sample rate:", sample_rate)

# Set the window size for the STFT to 1/30 of a second (assuming the audio is sampled at 44.1kHz)
hop_length = int(sample_rate / 30) 
window_size = hop_length * 1
print("Window size:", window_size)
print("Hop length:", hop_length)

dtype = torch.float16
if dtype == torch.float16:
   batch_count = 50
else:
   batch_count = 32


# Create MelSpectrogram transform
# mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_length, n_mels=800)

# Perform the MelSpectrogram transformation
# mel_spectrogram = mel_transform(waveform)

# print("Mel Spectrogram shape:", mel_spectrogram.shape)

# Convert the Mel spectrogram to dB scale
# db_transform = torchaudio.transforms.AmplitudeToDB()
# mel_spectrogram_db = db_transform(mel_spectrogram)
# mel_spectrogram_db = mel_spectrogram

stft_transform = torchaudio.transforms.Spectrogram(n_fft=window_size, hop_length=hop_length, power=None)
spectrogram = stft_transform(waveform)
# 
# print("Spectrogram shape:", spectrogram.shape)
# 
# # Take the magnitude of the spectrogram (convert from complex to real)
spectrogram_abs = torch.abs(spectrogram)

# Normalize the Mel spectrogram to the range 0-255
# mel_spectrogram_norm = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
mel_spectrogram_norm = torch.pow(spectrogram_abs, 3.0)
# clip anything less than 0.001 to zero
mel_spectrogram_norm = torch.clamp(mel_spectrogram_norm, min=0.001)
print("mel_spectrogram_norm", mel_spectrogram_norm.shape)

total_frames = int((waveform.shape[1] / sample_rate) * 30)
print("total_frames", total_frames)

spectrograms_per_frame = int(mel_spectrogram_norm.shape[2] / total_frames)
print("spectrograms_per_frame", spectrograms_per_frame)

device = torch.device("cuda")
repo_id = Path("/home/jonathan/models/v1-5")
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
pipe = StableDiffusionPipeline.from_pretrained(repo_id, 
                                               torch_dtype=dtype,
                                               safety_checker=None,
                                               )
pipe.to(device)
tomesd.apply_patch(pipe, ratio=0.5)
pipe.enable_xformers_memory_efficient_attention()

if batch_count > 1:
  pipe.enable_vae_slicing()
  # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
else:
  pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


prompt="glowing lights in the dark"

prompt_embeds = embed_prompt(pipe, prompt, device=device).to(dtype=dtype)

last_image = Image.open("initial_image.png")

output_dir = "images-3"

os.makedirs(output_dir, exist_ok=True)

# I need to duplicated the prompt_embeds batch_count times
embed_means = prompt_embeds.mean()
print("embed_means", embed_means.shape)
print("embed_means", embed_means)
prompt_embeds = prompt_embeds.repeat(batch_count, 1, 1)
print("prompt_embeds", prompt_embeds.shape)

spectrum_multipler = 0.0000000001

height = 512
width = 512

print("mel_spectrogram_norm", mel_spectrogram_norm.shape)

generator = torch.Generator(device=device).manual_seed(1024)


initial_latents = torch.randn(1, 
                              4, 
                              height//8, 
                              width//8, 
                              generator=generator, 
                              dtype=dtype,
                              device=device).repeat(batch_count, 1, 1, 1)

for f in tqdm(range(0, total_frames, batch_count)): 
    generator = torch.Generator(device=device).manual_seed(1024)

    all_exist = all(os.path.exists(f"{output_dir}/{f+i:08d}.png") for i in range(batch_count))

    if all_exist:
        print(f"All images for batch starting at {f:08d} exist")
        continue
    
    # convert a frame index to a spectrogram index
    start_spectrogram_index = f 
    end_spectrogram_index = start_spectrogram_index + batch_count 

    # get the spectrogram slice at the index
    spectrogram_slice = mel_spectrogram_norm[:, :, start_spectrogram_index:end_spectrogram_index]

    average_spectrogram_slice = torch.mean(spectrogram_slice, dim=0).permute(1, 0)

    # average_spectrogram_slice = torch.flip(average_spectrogram_slice, [1])

    # drop the extra frequencies
    trimmed_spectrogram_slice = average_spectrogram_slice[:, :768]

    # pad with zeros to be 768
    trimmed_spectrogram_slice = torch.cat((trimmed_spectrogram_slice, 
                                           torch.zeros(trimmed_spectrogram_slice.shape[0], 
                                                       (768 - trimmed_spectrogram_slice.shape[1]))), dim=1)

    expanded_spectrogram_slice = trimmed_spectrogram_slice.unsqueeze(1).expand(-1, 77, -1) 

    # expanded_spectrogram_slice[:, 11:, :] = 0.0

    new_prompt_embeds = prompt_embeds + spectrum_multipler * expanded_spectrogram_slice.to(device=device, dtype=dtype)

    # new_prompt_embeds_means = new_prompt_embeds.mean(dim=0)
    # print("expanded_spectrogram_slice_means", new_prompt_embeds.shape)

    # new_prompt_embeds = (new_prompt_embeds / new_prompt_embeds_means) * embed_means

    # get the spectrogram slices before and after the index
    with torch.inference_mode():
      image = pipe(prompt_embeds=new_prompt_embeds,
                 guidance_scale=7.5, 
                 num_inference_steps=20,
                 height=height,
                 width=width,
                 generator=generator,
                 latents=initial_latents, 
                 )[0]

      # save the image to a file using the frame index as the filename
      # and padding by 8 zeros
      for i in range(batch_count):
        image[i].save(f"{output_dir}/{f+i:08d}.png")


