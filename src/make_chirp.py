import torch
import soundfile as sf

def generate_chirp(sample_rate=44100, duration=5.0, f_start=20, f_end=22050):
    t = torch.linspace(0, duration, steps=int(sample_rate*duration))  # Time array
    freq_sweep = torch.linspace(f_start, f_end, steps=int(sample_rate*duration))  # Frequency array
    chirp = torch.sin(2 * torch.pi * freq_sweep * t)  # Generate the chirp
    return chirp



def save_as_wav(tensor, sample_rate, filename):
    # Convert tensor to numpy array
    data = tensor.detach().numpy()
    
    # Normalize data to range -1 to 1
    data = data / torch.max(torch.abs(tensor))
    
    # Save as .wav file
    sf.write(filename, data, sample_rate)



if __name__ == "__main__":
  chirp = generate_chirp()
  save_as_wav(chirp, 44100, 'chirp.wav')