"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import gzip
import os
import struct

# Third-party libraries
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "emnist-digits-train-images-idx3-ubyte.gz")
TRAIN_LABELS = os.path.join(DATASET_DIR, "emnist-digits-train-labels-idx1-ubyte.gz")
TEST_IMAGES = os.path.join(DATASET_DIR, "emnist-digits-test-images-idx3-ubyte.gz")
TEST_LABELS = os.path.join(DATASET_DIR, "emnist-digits-test-labels-idx1-ubyte.gz")

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    train_images = _load_idx_images(TRAIN_IMAGES)
    train_labels = _load_idx_labels(TRAIN_LABELS)
    test_images = _load_idx_images(TEST_IMAGES)
    test_labels = _load_idx_labels(TEST_LABELS)

    validation_size = min(10000, len(train_images) // 5)
    validation_data = (
        train_images[:validation_size],
        train_labels[:validation_size],
    )
    training_data = (
        train_images[validation_size:],
        train_labels[validation_size:],
    )
    test_data = (test_images, test_labels)

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def digit_from_vector(v):
    """Convert a 10-dim unit vector back to the digit index."""
    v = np.asarray(v).reshape(-1)
    if v.size != 10:
        raise ValueError("digit_from_vector expects a length-10 vector")
    return int(np.argmax(v))


def _load_idx_images(path):
    """Load EMNIST image file as flattened, normalized vectors."""
    with gzip.open(path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num_images, rows * cols).astype(np.float32) / 255.0
    return images


def _load_idx_labels(path):
    """Load EMNIST label file as integer targets."""
    with gzip.open(path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels