#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:37:29 2023

@author: green-machine
"""


import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def get_model_trained(X) -> keras.engine.sequential.Sequential:
    model = Sequential()
    model.add(
        Dense(
            64,
            input_dim=X.shape[1],
            kernel_initializer='he_uniform',
            activation='sigmoid'
        )
    )
    model.add(Dropout(.2))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mae')
    return model
