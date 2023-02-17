import math
import os

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment


def split_audio_into_files(filepath, folder, sec_per_split):
    filename = os.path.basename(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # get the length of the audio file
    audio = AudioSegment.from_wav(filepath)
    total_seconds = math.ceil(audio.duration_seconds)
    for i in range(0, total_seconds, sec_per_split):
        split_filename = str(i) + "_" + filename
        from_sec = i
        to_sec = i + sec_per_split
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = audio[t1:t2]
        split_audio.export(os.path.join(folder, split_filename), format="wav")
        print(str(i) + " Done")
        if i == total_seconds - sec_per_split:
            print("All splited successfully")


def features_extractor(filepath):
    # load the file (audio)
    audio, sample_rate = librosa.load(filepath, res_type="kaiser_fast")
    # we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


def create_df_from_audio_files(folder_audio_to_add):
    """
    Takes the folder of little audio files, extracts the features and returns a df with them.
    :param folder_audio_to_add: The folder that contains all of the little audio files
    :return: new dataframe with the audio features and class
    !!! NOTE: The small audio files need to be '.wav' files !!!
    """
    extracted_features = []
    class_ = os.path.split(folder_audio_to_add)[-1]
    for fname in folder_audio_to_add:
        if not fname.endswith(".wav"):
            continue
        filepath = os.path.join(folder_audio_to_add, fname)
        #         print(filepath)
        data = features_extractor(filepath)
        extracted_features.append([data, class_])
    # converting extracted_features to Pandas dataframe
    new_df = pd.DataFrame(extracted_features, columns=["feature", "class"])
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    return new_df
