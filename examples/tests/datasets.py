import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

_current_dataset = {
  'name': None,
  'x_train': None,
  'y_train': None,
  'x_test': None,
  'y_test': None,
}

def get_dataset(ds_name = None):
  if _current_dataset['name'] is not None and _current_dataset['name'] == ds_name:
    pass
  elif ds_name == 'mnist':
    load_mnist()
  return _current_dataset

def load_mnist():
  path = 'mnist.npz'
  origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
  path = get_file(
    path,
    origin=origin_folder + 'mnist.npz',
    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
  with np.load(path) as f:
    global _current_dataset
    _current_dataset = {}
    _current_dataset['name'] = 'mnist'
    _current_dataset['x_train'] = f['x_train'] / 255.0
    _current_dataset['y_train'] = f['y_train']
    _current_dataset['x_test'] = f['x_test'] / 255.0
    _current_dataset['y_test'] = f['y_test']
