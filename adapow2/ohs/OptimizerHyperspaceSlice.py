import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
import math
import numpy as np
from . import utils

tf.logging.set_verbosity(tf.logging.ERROR)

class OptimizerHyperspaceSlice(Optimizer):
  """OptimizerHyperspaceSlice a tool for creating low dimensional slice for hyperspace on specified point and direction, implemented for keras optimizers.

  Arguments:
      optimizer: keras optimizer instance, the optimizer you want to analyze.
      tiny_step: float > 0, the x axis of a slice is a sequence of ratio to this tiny_step.
      relative_sampling: tuple, the points you want to sample, using the current step as unit.
      absolute_sampling: tuple, the points you want to sample, using the tiny step as unit.
      save_path: string, directory for storing plot image and json file.
      plot: bool, storing plot image or not.
  """

  def __init__(self,
      optimizer,
      tiny_step = 1e-6,
      relative_sampling = utils.sampling_config(((1.0, 0.1), (2, 0.1))),
      absolute_sampling = utils.sampling_config(((20, 2), (120, 10))),
      save_path = 'data.ohs',
      plot = True,
      ):
    self.optimizer = optimizer
    self.tiny_step = tiny_step
    self.relative_sampling = relative_sampling
    self.absolute_sampling = absolute_sampling
    self.save_path = save_path
    self.plot = plot

  def get_gradients(self, loss, params):
    return self.optimizer.get_gradients(loss, params)

  def set_weights(self, weights):
    return self.optimizer.set_weights(weights)

  def get_weights(self):
    return self.optimizer.get_weights()

  def get_config(self):
    return self.optimizer.get_config()

  @classmethod
  def from_config(cls, config):
    return OptimizerHyperspaceSlice(cls(**config), self.tiny_step, self.relative_sampling, self.absolute_sampling, self.save_path, self.plot)

  def on_epoch_begin(self, epoch):
    if hasattr(self.optimizer, 'on_epoch_begin'):
      return self.optimizer.on_epoch_begin(epoch)

  def on_epoch_end(self, epoch, epoch_logs):
    if hasattr(self.optimizer, 'on_epoch_end'):
      return self.optimizer.on_epoch_end(epoch, epoch_logs)

  def on_iteration_begin(self, batch_logs):
    if hasattr(self.optimizer, 'on_iteration_begin'):
      return self.optimizer.on_iteration_begin(batch_logs)
    else:
      return True

  def on_iteration_end(self, batch_logs):
    self.batch_logs = batch_logs

    if hasattr(self.optimizer, 'on_iteration_end'):
      stop_training_batch = self.optimizer.on_iteration_end(batch_logs)
    else:
      stop_training_batch = True

    if stop_training_batch:
      self._create_slice_1d()

    return stop_training_batch

  def _create_slice_1d(self):
    slice_1d = self.eval_slice_1d()
    for _ in range(2, len(slice_1d['points']) + 1):
      self.batch_logs['train_function'](self.batch_logs['inputs'])
    slice_1d = self.eval_slice_1d()
    if self.save_path and len(slice_1d['points']) > 0:
      utils.save_slice_1d(slice_1d, self.save_path)
      if self.plot:
        utils.plot_slice_1d(slice_1d, self.save_path)

  def eval_slice_1d(self):
    slice_1d = K.get_session().run(self._slice_1d)
    n = slice_1d['num_of_points']
    indices_sorted = list(sorted(range(n), key=lambda i: slice_1d['points'][i]))
    return {
      'epoch': self.batch_logs['epoch'],
      'batch': self.batch_logs['batch'],
      'iterations': slice_1d['iterations'],
      'tiny_norm': self.tiny_norm,
      'step_norm': slice_1d['step_norm'],
      'points': slice_1d['points'][:n][indices_sorted],
      'losses': slice_1d['losses'][:n][indices_sorted],
      'step_points': slice_1d['points'][:2],
      'step_losses': slice_1d['losses'][:2],
    }

  def get_updates(self, loss, params):
    # points[0] is step start point, points[1] is step end point, others are sampling points
    max_num_of_points = 2 + len(self.relative_sampling) + len(self.absolute_sampling)

    points = K.zeros(max_num_of_points, dtype='float32')
    losses = K.zeros(max_num_of_points, dtype='float32')
    num_of_points = K.variable(0, dtype='int32')
    point_index = K.variable(2, dtype='int32')
    step_norm = K.variable(0., dtype='float32')
    iterations = K.variable(0, dtype='int32')

    self._slice_1d = {
      'points': points,
      'losses': losses,
      'num_of_points': num_of_points,
      'point_index': point_index,
      'step_norm': step_norm,
      'iterations': iterations,
    }

    P = params
    self.tiny_norm = np.float32(self.tiny_step * np.sqrt(np.sum([np.prod(K.int_shape(p)) for p in P])))
    P_prev = [K.zeros(K.int_shape(p), dtype='float32') for p in P]
    P_cache = [K.zeros(K.int_shape(p), dtype='float32') for p in P]
    S = [K.zeros(K.int_shape(p), dtype='float32') for p in P]

    def store_initial_params():
      def _store_initial_params():
        with ops.control_dependencies([p_prev.assign(p) for p_prev, p in zip(P_prev, P)]):
          return K.constant(True)
      return control_flow_ops.cond(
          math_ops.equal(iterations, 0),
          _store_initial_params,
          lambda: K.constant(True))

    def compute_slice_1d_points():
      def compute_points():
        step_end_point = step_norm / self.tiny_norm
        assign_relative_sampling_points = [
          points[0].assign(0.),
          points[1].assign(step_end_point),
        ]
        pi = 2
        for i in range(len(self.relative_sampling)):
          if self.relative_sampling[i] == 0 or self.relative_sampling[i] == 1:
            continue
          assign_relative_sampling_points.append(points[pi].assign(step_end_point * self.relative_sampling[i]))
          pi += 1

        with ops.control_dependencies(assign_relative_sampling_points + [point_index.assign(pi)]):
          min_relative_sampling_point = 0. if pi == 2 else (step_end_point * self.relative_sampling[0])
          max_relative_sampling_point = 0. if pi == 2 else (step_end_point * self.relative_sampling[-1])
          k_absolute_sampling = K.variable(self.absolute_sampling, dtype='float32')
          def check_absolute_sampling_point(i):
            def assign_absolute_sampling_point():
              with ops.control_dependencies([
                  points[point_index].assign(k_absolute_sampling[i]),
                  point_index.assign_add(1),
                ]):
                return i + 1
            return control_flow_ops.cond(math_ops.logical_or(
                math_ops.less(k_absolute_sampling[i], min_relative_sampling_point),
                math_ops.greater(k_absolute_sampling[i], max_relative_sampling_point)
                ),
                lambda: assign_absolute_sampling_point(),
                lambda: i + 1)
          assign_absolute_sampling_points = control_flow_ops.while_loop(
              lambda i: math_ops.less(i, len(self.absolute_sampling)),
              check_absolute_sampling_point,
              [K.constant(0, dtype='int32')])
          with ops.control_dependencies([assign_absolute_sampling_points]):
            return num_of_points.assign(point_index)

      def on_zero_step():
        with ops.control_dependencies([num_of_points.assign(0)]):
          return K.constant(0)

      def on_none_zero_step():
        with ops.control_dependencies([compute_points()]):
          return K.constant(0)

      assign_S = [s.assign(p_prev - p) for s, p_prev, p in zip(S, P_prev, P)]
      with ops.control_dependencies(assign_S):
        assign_P_cache = [p_cache.assign(p) for p_cache, p in zip(P_cache, P)]
        with ops.control_dependencies(assign_P_cache):
          assign_step_norm = step_norm.assign(math_ops.sqrt(math_ops.reduce_sum([math_ops.reduce_sum(math_ops.square(s)) for s in S])))
          with ops.control_dependencies([assign_step_norm]):
            return control_flow_ops.cond(math_ops.equal(step_norm, 0.), on_zero_step, on_none_zero_step)

    def compute_slice_1d_loss():
      def on_last_point():
        with ops.control_dependencies([p_prev.assign(p_cache) for p_prev, p_cache in zip(P_prev, P_cache)]):
          return [p.assign(p_cache) for p, p_cache in zip(P, P_cache)]
      save_prev_loss = losses[point_index - 1].assign(loss)
      with ops.control_dependencies([save_prev_loss]):
        update_P = control_flow_ops.cond(
            math_ops.less(point_index, num_of_points),
            lambda: [p.assign(p_prev - s * ((self.tiny_norm / step_norm) * points[point_index])) for p, p_prev, s in zip(P, P_prev, S)],
            on_last_point)
        with ops.control_dependencies(update_P):
          return point_index.assign_add(1)

    def train_and_slice():
      if hasattr(self.optimizer, 'stop_training_batch'):
        optimizer_stop_training_batch = self.optimizer.stop_training_batch
      else:
        optimizer_stop_training_batch = K.constant(True, dtype='bool')
      def on_start_trianing_batch():
        with ops.control_dependencies([num_of_points.assign(0), losses[0].assign(loss), iterations.assign_add(1)]):
          return K.constant(0)
      init_batch = control_flow_ops.cond(optimizer_stop_training_batch, on_start_trianing_batch, lambda: K.constant(0))
      with ops.control_dependencies([init_batch]):
        with ops.control_dependencies(self.optimizer.get_updates(loss, P)):
          prepare_slice = control_flow_ops.cond(optimizer_stop_training_batch, compute_slice_1d_points, lambda: K.constant(0))
          with ops.control_dependencies([prepare_slice]):
            return point_index.assign(2)

    with ops.control_dependencies([store_initial_params()]):
      return [control_flow_ops.cond(
          math_ops.less_equal(point_index, num_of_points),
          compute_slice_1d_loss,
          train_and_slice)]
