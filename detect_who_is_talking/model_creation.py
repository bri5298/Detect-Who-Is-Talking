import os
import pickle
from datetime import datetime

import librosa
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def encode_and_train_test_split(df, save_encoder=True):
    assert list(df.columns) == ["feature", "class"]
    # Split the dataset into independent and dependent dataset
    X = np.array(df["feature"].tolist())
    y = np.array(df["class"].tolist())
    # Label Encoding -> Label Encoder
    labelencoder = LabelEncoder()
    y = to_categorical(labelencoder.fit_transform(y))
    if save_encoder:
        output = open(os.path.join("models", "labelencoder.pkl"), "wb")
        pickle.dump(labelencoder, output)
        output.close()
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test, labelencoder


def create_model(y):
    # No of classes
    num_labels = y.shape[1]
    model = Sequential()
    # first layer
    model.add(Dense(100, input_shape=(40,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # second layer
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # third layer
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # final layer
    model.add(Dense(num_labels))
    model.add(Activation("softmax"))
    return model


def compile_and_train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs=100,
    num_batch_size=32,
    save_model=True,
):
    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )
    # Training the model
    if save_model:
        checkpointer = ModelCheckpoint(
            filepath=os.path.join("models", "audio_classification.hdf5"),
            verbose=1,
            save_best_only=True,
        )
        callbacks_ = [checkpointer]
    else:
        callbacks_ = None
    start = datetime.now()
    model.fit(
        X_train,
        y_train,
        batch_size=num_batch_size,
        epochs=num_epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks_,
        verbose=1,
    )
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    return model


def get_model_accuracy(model, X_test, y_test):
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return test_accuracy[1]


def predict_single_audio_file(filename, model, labelencoder):
    # preprocess the audio file
    audio, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    # Reshape MFCC feature to 2-D array
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    x_predict = model.predict(mfccs_scaled_features, verbose=0)
    predicted_label = np.argmax(x_predict, axis=1)
    # print(predicted_label)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    # print(f'\nPrediction is: {prediction_class[0]}')
    return prediction_class[0]


def get_labelencoder_from_file(le_filepath):
    pkl_file = open(le_filepath, "rb")
    labelencoder = pickle.load(pkl_file)
    pkl_file.close()
    return labelencoder


def load_model_from_file(model_filepath):
    model = load_model(model_filepath)
    return model
