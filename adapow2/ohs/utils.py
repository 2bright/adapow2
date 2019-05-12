import numpy as np
import matplotlib.pyplot as plt
import os
import json

def mkdir_p(path, error_if_exists = False):
  try:
    os.makedirs(path)
  except OSError as ex: 
    if error_if_exists:
      print('Error: path "%s" already exits.'% path)
      raise

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if callable(obj):
            return ''
        return json.JSONEncoder.default(self, obj)

def slice_1d_name(slice_1d):
  step_start_loss = slice_1d['step_losses'][0]
  step_end_loss = slice_1d['step_losses'][1]
  step_loss_direction = 'less' if step_end_loss < step_start_loss else ('greater' if step_end_loss > step_start_loss else 'equal')
  return 'e{:0>5d}-b{:0>7d}-i{}-{}'.format(slice_1d['epoch'], slice_1d['batch'], slice_1d['iterations'], step_loss_direction)

def save_slice_1d(slice_1d, path):
  path = path + '/'
  mkdir_p(path)
  with open(path + slice_1d_name(slice_1d) + '.json', 'w') as json_file:
    json_file.write(json.dumps(slice_1d, indent=2, cls=NumpyEncoder))

def plot_slice_1d(slice_1d, path):
  path = path + '/'
  mkdir_p(path)

  plt.rcParams['figure.figsize'] = (14.0, 8.0)

  fig, ax1 = plt.subplots()
  ax1.plot(slice_1d['points'], slice_1d['losses'], 'b-')
  ax1.plot(slice_1d['step_points'], slice_1d['step_losses'], 'r-')
  ax1.set_xlabel('ratio to tiny_norm (tiny_norm:' + str(slice_1d['tiny_norm']) + ', step_norm:' + str(slice_1d['step_norm']) + ')', color='b')
  ax1.set_ylabel('loss', color='b')
  
  fig.tight_layout()
  plt.title('epoch:' + str(slice_1d['epoch']) + ', batch:' + str(slice_1d['batch']) + ', iterations:' + str(slice_1d['iterations']))
  plt.savefig(path + slice_1d_name(slice_1d) + '.png')
  plt.close()

def sampling_config(config):
  """generate sampling points

  Arguments:
      config: a tuple, each value in the tuple is a pair. for a pair, the first value define a range, the second value is sampling gap within this range. e.g., ((-8, 2), (-2, 1), (2, 0.5), (8, 2)) means using 2 as gap within [-8, -2], using 1 as gap within [-2, 0], using 0.5 as gap within [0, 2], using 2 as gap within [2, 8]
  """

  forward = []
  backward = []

  for c in config:
    if c[0] > 0:
      forward.append(c)
    elif c[0] < 0:
      backward.append(c)

  points = [0.]

  prev_r = (0, 0)
  for r in forward:
    step_points = np.arange(r[0], prev_r[0], -r[1])
    points = points + list(reversed(step_points))
    prev_r = r

  prev_r = (0, 0)
  for r in reversed(backward):
    step_points = np.arange(r[0], prev_r[0], r[1])
    points = list(step_points) + points
    prev_r = r

  return tuple(points)
