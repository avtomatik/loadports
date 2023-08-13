#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 15:56:02 2023

@author: green-machine
"""


from pathlib import Path

import tensorflow as tf
from data.make_dataset import get_data_frame, get_X_y
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from visualization.visualize import plot_model_train_val_losses

if __name__ == '__main__':

    X_raw, y_raw = get_data_frame().pipe(get_X_y)

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X_raw[:, 0])
    X = X_vec.toarray()

    le = LabelEncoder()
    y = le.fit_transform(y_raw.ravel())

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)


# =============================================================================
# TensorFlow: Deep Learning
# =============================================================================
    model = Sequential()
    model.add(
        Dense(
            64,
            input_dim=X.shape[1],
            kernel_initializer='he_uniform',
            activation='sigmoid'
        )
    )
    model.add(Dropout(.05))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mae')
# =============================================================================
# Architecture
# =============================================================================
    model.summary()

    model_history = model.fit(
        X_train,
        y_train,
        verbose=1,
        epochs=100,
        validation_split=.2
    )

    model.evaluate(X_test, y_test, verbose=0)

# =============================================================================
# Plot the Results
# =============================================================================
    plot_model_train_val_losses(model_history.history)

# =============================================================================
# Save Model
# =============================================================================
    PATH_MODEL = Path(__file__).parent.parent.joinpath('models')
    PATH_MODEL.mkdir(exist_ok=True)
    model.save(PATH_MODEL)

# =============================================================================
# Load Model
# =============================================================================
    new_model = tf.keras.models.load_model(PATH_MODEL)
# =============================================================================
# Architecture
# =============================================================================
    new_model.summary()
