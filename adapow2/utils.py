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
        if isinstance(obj, np.bool_) or isinstance(obj, bool):
            return int(obj)
        if callable(obj):
            return ''
        print(type(obj))
        return json.JSONEncoder.default(self, obj)

def store_history_state(history_state, path):
  if len(history_state) == 0:
    return

  plt.rcParams['figure.figsize'] = (14.0, 8.0)

  with open(path + '.history_state.json', 'w') as his_file:
    his_file.write(json.dumps(history_state, cls=NumpyEncoder, indent=2))

  for k, vs in history_state.items():
    fig, ax1 = plt.subplots()

    segs = []
    seg_i = []
    seg_v = []

    seq_start = True
    first_non_zero = 0.

    for vi in range(len(vs)):
      v = vs[vi]
      if v != 0.:
        first_non_zero = v
        break

    for vi in range(len(vs)):
      v = vs[vi]

      if seq_start and v == 0.:
        v = first_non_zero
      else:
        seq_start = False

      seg_v.append(v if v != np.inf else -1)
      seg_i.append(vi)

      if len(seg_v) > 1 and ((v <= 0. and seg_v[0] > 0.) or (v > 0. and seg_v[0] <= 0.)):
        segs.append([seg_i, seg_v, 'g-' if seg_v[0] > 0. else 'r-'])
        seg_i = []
        seg_v = []
        seg_v.append(v if v != np.inf else -1)
        seg_i.append(vi)

    segs.append([seg_i, seg_v, 'g-' if seg_v[0] > 0. else 'r-'])
    
    for seg in segs:
      ax1.plot(*seg)

    ax1.set_ylabel(k, color='b')

    ax1.set_xlabel('iterations: ' + str(segs[-1][1][-5:-1]))
    
    fig.tight_layout()
    plt.savefig(path + '.' + k + '.png')
    plt.close()
