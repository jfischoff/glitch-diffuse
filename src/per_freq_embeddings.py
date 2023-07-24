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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import tomesd
from utils import embed_prompt

# TODO not padding is breaking things!


def generate_interpolated_embeddings(pipe, prompts, bin_count, device='cpu'):
    # Generate embeddings for each prompt
    embeddings = [embed_prompt(pipe, prompt, device) for prompt in prompts]

    # pad with zeros based on the longest embedding
    max_length = max([x.shape[1] for x in embeddings])
    embeddings = [torch.cat([x, 
                             torch.zeros(x.shape[0], max_length - x.shape[1], x.shape[2]).to(device)], dim=1) for x in embeddings]

    for e in embeddings:
      print("embeddings", e.shape)

    # The step size for each bin
    bin_size = bin_count / (len(prompts) - 1)

    # Create tensor to store interpolated embeddings
    interpolated_embeddings = torch.zeros(bin_count, embeddings[0].size(0), embeddings[0].size(1), embeddings[0].size(2))
    print("interpolated_embeddings", interpolated_embeddings.shape)

    for bin_index in range(bin_count):
        # Identify the prompts to interpolate between
        first_prompt = int(bin_index / bin_size)
        second_prompt = min(first_prompt + 1, len(prompts) - 1)

        # Calculate interpolation weights
        first_weight = 1 - ((bin_index % bin_size) / bin_size)
        second_weight = 1 - first_weight

        # Generate interpolated embedding
        interpolated = first_weight * embeddings[first_prompt] + second_weight * embeddings[second_prompt]


        # Store the interpolated embedding
        interpolated_embeddings[bin_index] = interpolated

    return interpolated_embeddings

def generate_spectrogram(audio_file, window_duration=1/30):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)

    print("Waveform shape:", waveform.shape)
    print("Sample rate:", sample_rate)

    # Set the window size for the STFT
    hop_length = int(sample_rate * window_duration) 
    window_size = hop_length * 1
    print("Window size:", window_size)
    print("Hop length:", hop_length)

    # Generate spectrogram
    stft_transform = torchaudio.transforms.Spectrogram(n_fft=window_size, hop_length=hop_length, power=None)
    spectrogram = stft_transform(waveform)

    spectrogram_abs = torch.abs(spectrogram)

    # Normalize spectrogram
    mel_spectrogram_norm = torch.pow(spectrogram_abs, 2.0)
    print("mel_spectrogram_norm", mel_spectrogram_norm.shape)

    total_frames = int((waveform.shape[1] / sample_rate) / window_duration)
    print("total_frames", total_frames)

    return mel_spectrogram_norm, waveform, sample_rate, total_frames

def concat_and_resize(tensors):
    # Concatenate tensors along the third dimension
    result = torch.cat(tensors, dim=1)

    # Check the size of the third dimension
    if result.shape[1] > 77:
        # If it's greater than 77, truncate
        result = result[:, :, :77, :]
    elif result.shape[1] < 77:
        # If it's less than 77, pad with zeros
        padding = torch.zeros(result.shape[0], 77 - result.shape[1], result.shape[2]).to(result.device)
        result = torch.cat([result, padding], dim=1)

    return result

[(bass_spectrogram, bass_waveform, sample_rate, total_frames), 
 (drums_spectrogram, drums_waveform, _, _), 
 (other_spectrogram, other_waveform, _, _),
 (vocals_spectrogram, vocals_waveform, _, _)] = [generate_spectrogram(f"separated/htdemucs/in-the-morning-trim/{x}.wav") for x in ["bass", "drums", "other", "vocals"]]

all_spectrograms = [bass_spectrogram, drums_spectrogram, other_spectrogram, vocals_spectrogram]
all_waveforms = [bass_waveform, drums_waveform, other_waveform, vocals_waveform]

spectrograms_per_frame = int(bass_spectrogram.shape[2] / total_frames)

device = torch.device("cuda")
repo_id = Path("/home/jonathan/models/dreamshaper-6")
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
pipe = StableDiffusionPipeline.from_pretrained(repo_id)
pipe.to(device)
# tomesd.apply_patch(pipe, ratio=0.5)
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


other_colors = ["green", "light blue", "blue", "purple", "red", "light red", "orange", "yellow"] + (["white"] * 100)
other_prompts=[f"hundreds of little glowing {x} lights in the dark" for x in other_colors]
other_prompt_embeddings = generate_interpolated_embeddings(pipe, 
                                                           other_prompts, 
                                                           other_spectrogram.shape[1], device)

bass_colors = ["blue", "purple", "violet"]
bass_prompts = [f"vibranting, glowing {x} bars in the dark" for x in bass_colors]
bass_prompt_embeddings = generate_interpolated_embeddings(pipe,
                                                          bass_prompts,
                                                          bass_spectrogram.shape[1], device)

drum_prompts = ["vibrating glowing mound of dirt in the dark", 
                "vibrating disc at night, in the dark",
                "millions of fireflies in the dark, glowing, vibrating"] 
drum_prompt_embeddings = generate_interpolated_embeddings(pipe,
                                                          drum_prompts,
                                                          drums_spectrogram.shape[1], device)

voice_colors = ["green", "yellow", "orange", "red"]
voice_prompts = [f"swirling, thick clouds of {x} smoke in the dark" for x in voice_colors]
voice_prompt_embeddings = generate_interpolated_embeddings(pipe,
                                                            voice_prompts,
                                                            vocals_spectrogram.shape[1], device)

all_embeddings = [bass_prompt_embeddings, 
                  drum_prompt_embeddings, 
                  other_prompt_embeddings, 
                  voice_prompt_embeddings]

background_prompt = "black, in the dark"
background_prompt_embedding = embed_prompt(pipe, background_prompt, device).to("cpu") 

def average_waveform_slice(frame_index, waveform, sample_rate=44100, fps=30):
    # Calculate the number of samples per frame
    samples_per_frame = sample_rate / fps

    # Calculate the start and end indices for the waveform slice
    start_index = int(frame_index * samples_per_frame)
    end_index = int((frame_index + 1) * samples_per_frame)
    print("start_index", start_index)
    print("end_index", end_index)

    # Extract the waveform slice and average across the two channels and all samples
    waveform_slice = waveform[:, start_index:end_index]
    average_waveform_slice = torch.mean(waveform_slice.abs())

    print("average_waveform_slice", average_waveform_slice)

    return average_waveform_slice

def generate_spectrogram_slice(frame_index, spectrograms_per_frame, spectrogram, waveform, sample_rate=44100):
    # Convert a frame index to a spectrogram index
    spectrogram_index = frame_index * spectrograms_per_frame
    print("spectrogram_index", spectrogram_index)

    # Get the spectrogram slice at the index
    spectrogram_slice = spectrogram[:, :, spectrogram_index]
    print("spectrogram_slice", spectrogram_slice.shape)

    # Calculate the average spectrogram slice
    average_spectrogram_slice = torch.mean(spectrogram_slice, dim=0)
    print("average_spectrogram_slice", average_spectrogram_slice.shape)

    # Normalize the average spectrogram slice so that the sum of all values equals 1
    the_sum = average_spectrogram_slice.sum()
    print("the_sum", the_sum)
    average_spectrogram_slice = average_spectrogram_slice / the_sum

    # Adjust dimensions for broadcasting
    average_spectrogram_slice = average_spectrogram_slice.view(-1, 1, 1, 1)

    # average the waveform over this time slice
    average_amplitude = average_waveform_slice(frame_index, waveform, sample_rate)

    return average_spectrogram_slice, average_amplitude


last_image = Image.open("images-1/00000000.png")

output_dir = "images-2"

os.makedirs(output_dir, exist_ok=True)

height=512
width=512

these_embeddings = [other_prompt_embeddings]
these_spectrograms = [other_spectrogram]
these_waveforms = [other_waveform]

for f in tqdm(range(total_frames)): 
    generator = torch.Generator(device=device).manual_seed(1024)
    # for each frame get 33 spectrogram samples in front of the frame
    # and 33 spectrogram samples after the frame
    # and the spectrogram sample at the frame

    #check if the output image exists
    if os.path.exists(f"{output_dir}/{f:08d}.png"):
      print(f"images/{f:08d}.png exists")
      continue


    all_scaled_embeddings = []
    for embedding, spectrogram, waveform in zip(these_embeddings, these_spectrograms, these_waveforms):
      spectrogram_slice, average_amplitude = generate_spectrogram_slice(f, spectrograms_per_frame, spectrogram, waveform, sample_rate)
      result = embedding * spectrogram_slice

      actual_scale = min(1, average_amplitude * 100.0)
      print("actual_scale", actual_scale)
      actual_scale = 1.0

      all_scaled_embeddings.append(result.sum(dim=0, keepdim=True).squeeze(0) * actual_scale)


    # Multiply
    prompt_embeddings = concat_and_resize(all_scaled_embeddings)


    # get the spectrogram slices before and after the index
    image = pipe(prompt_embeds=prompt_embeddings,
                 guidance_scale=7.5, 
                 generator=generator,
                 num_inference_steps=20
                 # image=last_image,
                 # strength=0.98,
                 )[0]

    # save the image to a file using the frame index as the filename
    # and padding by 8 zeros
    image[0].save(f"{output_dir}/{f:08d}.png")

    last_image = image[0]

