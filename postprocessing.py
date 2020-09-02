
import numpy as np


def postprocess(y_data, y_pred, block=125, inc=125):
    print(type(y_data), type(y_pred))
    if type(y_data) is np.ndarray:
        y_data = y_data.flatten()

    if type(y_pred) is np.ndarray:
        y_pred = y_pred.flatten()

    processed_data = []
    processed_pred = []
    start = 0

    while start + block < len(y_pred):
        processed_pred.append(round(sum(y_pred[start:start + block]) / block))
        start += inc

    start = 0
    while start + block < len(y_data):
        processed_data.append(round(sum(y_data[start:start + block]) / block))
        start += inc

    return processed_data, processed_pred
