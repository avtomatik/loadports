#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 23:07:16 2023

@author: green-machine
"""


from matplotlib import pyplot as plt


def plot_model_train_val_losses(history_dict: dict[str, list[float]]) -> None:
    """
    Plots Train & Validation Losses per Epoch
    """
    plt.figure(figsize=(8, 5))
    for array in history_dict.values():
        plt.plot(array)

    plt.title("Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend(history_dict.keys())
    plt.grid()
    plt.show()
