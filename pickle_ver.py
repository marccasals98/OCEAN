import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img

# Load the pickle file
with open('/home/usuaris/veussd/DATABASES/Ocean/Spectrograms_AcousticTrends/23_08_04_13_56_33_267b8tlx_astral-elevator-83/BallenyIslands2015_20150115-170000_Blue_Bm-D_20150115_2652_7652_1000Hz.pickle', 'rb') as f:
    spectrogram = pickle.load(f)

# Plot the spectrogram
plt.imshow(spectrogram['features'])
plt.xlabel('Time')
plt.ylabel('Frequency')
img.imsave('spectrogram_noAxis.png', spectrogram['features'])

# Save the spectrogram as a PNG file
plt.savefig('spectrogram.png', dpi=300, bbox_inches='tight')