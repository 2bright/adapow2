import numpy as np
import tensorflow as tf
from models import *
from datasets import *
from utils import *
import os
import time
import glob
import hashlib
import json
from adapow2.x import NoLossUp
from adapow2.ohs import OptimizerHyperspaceSlice as OHS
import kv_prod_union as kv

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/'

try:
  os.makedirs(data_dir)
except:
  pass

data_len = None

def test_model(hp, md):
  optimizer = NoLossUp(hp)
  # optimizer = OHS(NoLossUp(hp), save_path=hp['save_path'])

  ds = get_dataset(md['dataset'])
  time1 = time.time()
  print('shape:', ds['x_train'].shape)
  if md['model'] == 'mnist_vae':
    model, encoder, decoder = create_model(md['model'], optimizer=optimizer)
    metrics = model.fit(ds['x_train'][:data_len], epochs=md['epochs'])
    # plot_vae_results([encoder, decoder], [ds['x_test'], ds['y_test']], batch_size=md['batch_size'], model_name='data/mnist_vae')
  else:
    model = create_model(md['model'], optimizer=optimizer)
    metrics = model.fit(ds['x_train'][:data_len], ds['y_train'][:data_len], epochs=md['epochs'])
  time2 = time.time()
  return time2 - time1, metrics.history, {} #model.optimizer.history_state_cache

model_dataset_samples = kv.compile({
  ('model', 'dataset', 'epochs', 'batch_size', 'regularization'):
  [
    (['mnist_2_layers'], 'mnist', 10, 128, 'l2'),
    # (['mnist_logreg', 'mnist_tf_demo'], 'mnist', 30, 128, 'l2'),
    # (['mnist_logreg', 'mnist_tf_demo', 'mnist_2_layers', 'mnist_2_layers_l2', 'mnist_vae', 'mnist_mlp', 'mnist_2c2d'], 'mnist', 50, 128, 'l2'),
  ]
})

adapow2_config = {
  ('pow2_increase_delta', 'pow2_decrease_delta', 'store_history_state'):
  [
    (0.1, 1.0, True),
  ]
}

adapow2_hyperparameters = kv.compile(adapow2_config)
print('adapow2_hyperparameters number:', len(adapow2_hyperparameters))

index = 0
for md in model_dataset_samples:
  print(md)
  for hp in adapow2_hyperparameters:
    hp['save_path'] = 'data.ohs.' + md['model']
    hp['history_state_path'] = 'data.NoLossUp.' + md['model']
    print(hp)
    index += 1
    hp_str = json.dumps(hp, indent=2, cls=NumpyEncoder)
    hp_md5 = hashlib.md5(hp_str.encode('utf-8')).hexdigest()
    result_path_pattern = '{}{:04d}.*.{}.{}.{}.{}.*'.format(data_dir, index, md['model'], md['dataset'], md['epochs'], hp_md5)
    if len(glob.glob(result_path_pattern)) >= 3:
      print('already exists, skip.')
    else:
      time_elapsed, metrics, history_state = test_model(hp, md)
      result_path = '{}{:04d}.{}.{}.{}.{}.{}'.format(data_dir, index, int(time_elapsed), md['model'], md['dataset'], md['epochs'], hp_md5)
      save_metrics(metrics, result_path, hp_str, history_state)
