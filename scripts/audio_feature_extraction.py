import librosa

def feature_extraction(file_path):

    # Load the wave file
    x, sr = librosa.load(file_path)

    # Extract the mean of the signal
    mean = librosa.feature.mean(x)

    # Extract the standard deviation of the signal
    std = librosa.feature.std(x)

    # Extract the amplitude of the signal
    amplitude = librosa.feature.rmse(x)

    # Extract the kurtosis of the signal
    kurtosis = librosa.feature.kurtosis(x)

    # Extract the phase of the signal
    phase = librosa.feature.phase(x)

    # Extract the spectral energy density of the signal
    spectral_energy_density = librosa.feature.spectral_energy_density(x, sr)

    # Extract the power spectrum of the signal
    power_spectrum = librosa.feature.power_spectral_density(x, sr)

    # Extract the frequency content of the signal
    spectral_centroid = librosa.feature.spectral_centroid(x, sr)

    # Extract the spectral spread of the signal
    spectral_spread = librosa.feature.spectral_spread(x, sr)

    # Extract the time domain features Mel-frequency cepstral coefficients (MFCCs)
    time_domain_features = librosa.feature.mfcc(x, sr)

    # Print the extracted features
    print("Mean:", mean)
    print("Standard deviation:", std)
    print("Kurtosis:", kurtosis)
    print("Spectral energy density:", spectral_energy_density)
    print("Frequency content:", spectral_centroid)
    print("Time domain features:", time_domain_features)
    print("Amplitude:", amplitude)
    print("Phase:", phase)
    print("Power spectrum:", power_spectrum)
    print("Spectral spread:", spectral_spread)