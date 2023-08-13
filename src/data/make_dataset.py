#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 15:38:50 2023

@author: green-machine
"""


from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd


@cache
def get_data_frame() -> pd.DataFrame:
    return pd.read_csv(
        Path(__file__).parent.parent.parent.joinpath('data').joinpath('dataset.csv')
    )


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray]:
    col_num = 2
    return df.iloc[:, :col_num].values, df.iloc[:, col_num:].values
