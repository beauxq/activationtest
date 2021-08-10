import os
import numpy as np

def render_mnist_image(data: np.ndarray, high: float = 1.0) -> str:
    """ ascii art """
    width = 28
    height = 28
    colors = " -+@"  # ᐧ █
    assert len(data) == width * height
    assert high > 0

    tr = ""

    for val in data:
        color = min(int(val * len(colors) / high), len(colors) - 1)
        tr += colors[color]
        
        if len(tr) % (width + 1) == width:
            tr += "\n"

    return tr[:-1]

def _read_big_endian_uint32(data: bytes) -> int:
    assert len(data) == 4
    return (data[0] << 24) + (data[1] << 16) + (data[2] << 8) + data[3]

def _get_image_data(filename: str) -> np.ndarray:
    try:
        with open(os.path.join("mnist_data", filename), 'rb') as file:
            data = file.read()
            assert _read_big_endian_uint32(data[:4]) == 2051  # images
            length = _read_big_endian_uint32(data[4:8])
            row_count = _read_big_endian_uint32(data[8:12])
            col_count = _read_big_endian_uint32(data[12:16])
            pixels = np.frombuffer(data, dtype = np.uint8, offset = 16)
            tr = pixels / 255  # change from [0, 255] to [0, 1]
            return tr.reshape(length, row_count * col_count)
    except IOError as error:
        print("need mnist data in directory `mnist_data`")
        print("")
        raise error

_classification_labels = [
    np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
]

def _get_labels(filename: str, as_value: bool) -> np.ndarray:
    """
    if not `as_value`, returns the classification array
    """
    try:
        with open(os.path.join("mnist_data", filename), 'rb') as file:
            data = file.read()
            assert _read_big_endian_uint32(data[:4]) == 2049  # labels
            length = _read_big_endian_uint32(data[4:8])
            values = np.frombuffer(data, dtype = np.uint8, offset = 8)
            assert len(values) == length
            if as_value:
                return values
            tr = [_classification_labels[val] for val in values]
            return np.array(tr)
    except IOError as error:
        print("need mnist data in directory `mnist_data`")
        print("")
        raise error

def get_data():
    """
    returns tuple of numpy arrays:
     - training images
     - training labels - as classification array - ex. for 2 `[0.0, 0.0, 1.0, 0.0, ...]`
     - test images
     - test labels - as values
    """
    return (
        _get_image_data("train-images.idx3-ubyte"),
        _get_labels("train-labels.idx1-ubyte", False),
        _get_image_data("t10k-images.idx3-ubyte"),
        _get_labels("t10k-labels.idx1-ubyte", True)
    )
