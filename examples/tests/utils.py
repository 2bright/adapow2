import numpy as np
import matplotlib.pyplot as plt
import os
import json

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

def plot_dict(history_state = {}):
  if len(history_state) == 0:
    return

  with open(path + '.history_state.json', 'w') as his_file:
    his_file.write(json.dumps(history_state, cls=NumpyEncoder, indent=2))

  for k, vs in history_state.items():
    fig, ax1 = plt.subplots()

    segs = []
    seg_i = []
    seg_v = []

    for vi in range(len(vs)):
      v = vs[vi]

      seg_v.append(v if v != np.inf else -1)
      seg_i.append(vi)

      if len(seg_v) > 1 and ((v == 0. and seg_v[0] != 0.) or (v != 0. and seg_v[0] == 0.)):
        segs.append([seg_i, seg_v, 'g-' if seg_v[0] != 0. else 'r-'])
        seg_i = []
        seg_v = []
        seg_v.append(v if v != np.inf else -1)
        seg_i.append(vi)

    segs.append([seg_i, seg_v, 'g-' if seg_v[0] != 0. else 'r-'])
    
    for seg in segs:
      ax1.plot(*seg)

    ax1.set_ylabel(k, color='b')

    ax1.set_xlabel('iterations: ' + str(segs[-1][1][-5:-1]))
    
    fig.tight_layout()
    plt.savefig(path + '.' + k + '.png')
    plt.close()

def save_metrics(metrics, path, hp = None, history_state = {}):
  if hp is not None:
    with open(path + '.hp.json', 'w') as hp_file:
      hp_file.write(hp)

  with open(path + '.metrics.json', 'w') as metrics_file:
    metrics_file.write(json.dumps(metrics, indent=2, cls=NumpyEncoder))

  plt.rcParams['figure.figsize'] = (14.0, 8.0)

  fig, ax1 = plt.subplots()
  
  ax1.plot(metrics['loss'], 'b-')
  if 'val_loss' in metrics:
    ax1.plot(metrics['val_loss'], 'b:')
  ax1.set_ylabel('loss', color='b')
  
  ax1.set_xlabel('epochs')
  ax2 = ax1.twinx()
  
  if 'acc' in metrics:
    ax2.plot(metrics['acc'], 'r-')
  if 'val_acc' in metrics:
    ax2.plot(metrics['val_acc'], 'r:')
  ax2.set_ylabel('acc', color='r')
  
  fig.tight_layout()
  plt.savefig(path + '.loss-acc.png')
  plt.close()

  plot_dict(history_state)

def plot_vae_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
