import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import librosa
import csv
from create_header import Header


header_string='filename chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmonics_mean harmonics_var perceptual_mean perceptual_var spectral_contrast1_mean spectral_contrast1_var spectral_constrast2_mean spectral_contrast2_var spectral_constrast3_mean spectral_contrast3_var  spectral_constrast4_mean spectral_contrast4_var spectral_constrast5_mean spectral_contrast5_var spectral_contrast6_mean spectral_contrast6_var spectral_contrast7_mean spectral_contrast7_var spectral_flatness_mean spectral_flatness_var tonnetz_1_mean tonnetz_1_var tonnetz_2_mean tonnetz_2_var tonnetz_3_mean tonnetz_3_var tonnetz_4_mean tonnetz_4_var tonnetz_5_mean tonnetz_5_var tonnetz_6_mean tonnetz_6_var '
header=Header(header_string)
header=header.form_header()


file = open('music.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'C:/Users/PROMIT/Desktop/Music_Genre/Data/genres_original/{g}'):
        songname = f'C:/Users/PROMIT/Desktop/Music_Genre/Data/genres_original/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        harmonics, perceptual = librosa.effects.hpss(y)
        tempo=librosa.beat.tempo(y=y,sr=sr)
        spectral_contrast=librosa.feature.spectral_contrast(y=y,sr=sr)
        #Spectral flatness
        spectral_flatness=librosa.feature.spectral_flatness(y=y)
        mel_spectrogram=librosa.feature.melspectrogram(y=y,sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.mean(rms)} {np.var(rms)} {np.mean(spec_cent)} {np.var(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)} {np.mean(harmonics)} {np.var(harmonics)} {np.mean(perceptual)} {np.var(perceptual)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
            to_append += f' {np.var(e)}'
        for j in range(0,7):
            to_append += f' {np.mean(spectral_contrast[j])}'
            to_append += f' {np.var(spectral_contrast[j])}'
            j=0
    
        to_append += f' {np.mean(spectral_flatness)}'
        to_append += f' {np.var(spectral_flatness)}'
        y_harmonic=librosa.effects.harmonic(y=y)
        tonnetz=librosa.feature.tonnetz(y=y_harmonic,sr=sr)
        for k in range(0,6):
            to_append += f' {np.mean(tonnetz[j])}'
            to_append += f' {np.var(tonnetz[j])}'
            k=0
        for l in range(0,128):
            to_append += f' {np.mean(mel_spectrogram[j])}'
            to_append += f' {np.var(mel_spectrogram[j])}'
        to_append += f' {float(tempo)}'
        to_append += f' {g}'
        file = open('music.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
