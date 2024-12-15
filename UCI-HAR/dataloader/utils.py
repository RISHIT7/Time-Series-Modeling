import pandas as pd
import numpy as np

import torch

import random

# Set seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


def _file_load(filepath, separator="\s+") -> np.ndarray:
    """
    Load a file as a NumPy array.

    Parameters:
    -----------
    filepath : str
        The path to the file to load.
    separator : str, optional
        Separator used in the file (default is '\s+').

    Returns:
    --------
    np.ndarray
        Loaded data as a NumPy array.
    """
    try:
        df = pd.read_csv(filepath, header=None, sep=separator)
        return df.values
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filepath}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file: {filepath}") from e


def _train_test_append(filenames, append_before=""):
    datalist = list()
    for name in filenames:
        data = _file_load(append_before + name)
        datalist.append(data)
    datalist = np.dstack(datalist)
    return datalist


def _inertial_signals_load(group, append_before=""):
    filepath = append_before + group + "/Inertial Signals/"
    filenames = list()
    filenames += [
        "total_acc_x_" + group + ".txt",
        "total_acc_y_" + group + ".txt",
        "total_acc_z_" + group + ".txt",
    ]
    filenames += [
        "body_acc_x_" + group + ".txt",
        "body_acc_y_" + group + ".txt",
        "body_acc_z_" + group + ".txt",
    ]
    filenames += [
        "body_gyro_x_" + group + ".txt",
        "body_gyro_y_" + group + ".txt",
        "body_gyro_z_" + group + ".txt",
    ]

    X = _train_test_append(filenames, filepath)
    y = _file_load(append_before + group + "/y_" + group + ".txt")

    return X, y


def load_dataset(
    filepath: str, append_before=""
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the UCI HAR dataset.

    Parameters:
    -----------

    filepath : str
        The path to the dataset files.
    append_before : str, optional
        A string to append before the filepath (default is '').

    Returns:
    --------

    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - trainX: np.ndarray
            The training data features.
        - trainy: np.ndarray
            The one-hot encoded training data labels.
        - testX: np.ndarray
            The testing data features.
        - testy: np.ndarray
            The one-hot encoded testing data labels.
    """
    trainX, trainy = _inertial_signals_load("train", append_before + filepath)
    testX, testy = _inertial_signals_load("test", append_before + filepath)

    trainy -= 1
    testy -= 1

    trainy = pd.get_dummies(trainy[:, 0], dtype=int).values
    testy = pd.get_dummies(testy[:, 0], dtype=int).values

    return trainX, trainy, testX, testy


if __name__ == "__main__":
    # Example usage
    trainX, trainy, testX, testy = load_dataset("UCI HAR Dataset/")
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
