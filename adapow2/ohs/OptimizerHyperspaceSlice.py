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

  def on_train_begin(self, train_logs):
    if hasattr(self.optimizer, 'on_train_begin'):
      self.optimizer.on_train_begin(train_logs)

  def on_train_end(self, train_logs):
    if hasattr(self.optimizer, 'on_train_end'):
      self.optimizer.on_train_end(train_logs)

  def on_epoch_begin(self, epoch_logs):
    if hasattr(self.optimizer, 'on_epoch_begin'):
      self.optimizer.on_epoch_begin(epoch_logs)

  def on_epoch_end(self, epoch_logs):
    if hasattr(self.optimizer, 'on_epoch_end'):
      self.optimizer.on_epoch_end(epoch_logs)

  def on_iteration_begin(self, batch_logs):
    if hasattr(self.optimizer, 'on_iteration_begin'):
      self.optimizer.on_iteration_begin(batch_logs)

  def on_iteration_end(self, batch_logs):
    self.batch_logs = batch_logs

    self._create_slice_1d()

    if hasattr(self.optimizer, 'on_iteration_end'):
      return self.optimizer.on_iteration_end(batch_logs)
    else:
      return True

  def _create_slice_1d(self):
    slice_1d = self.eval_slice_1d()
    while (slice_1d['point_index'] <= slice_1d['num_of_points']):
      self.batch_logs['train_function'](self.batch_logs['inputs'])
      slice_1d = self.eval_slice_1d()
    
    if self.save_path:
      utils.save_slice_1d(slice_1d, self.save_path)
      if self.plot:
        utils.plot_slice_1d(slice_1d, self.save_path)

  def eval_slice_1d(self):
    slice_1d = K.get_session().run(self._slice_1d)
    n = slice_1d['num_of_points']
    indices_sorted = list(sorted(range(n), key=lambda i: slice_1d['points'][i]))
    return {
      'point_index': slice_1d['point_index'],
      'num_of_points': n,
      'epoch': self.batch_logs['epoch'],
      'batch': self.batch_logs['batch'],
      'iterations': slice_1d['iterations'],
      'tiny_norm': self.tiny_norm,
      'step_norm': slice_1d['step_norm'],
      'grad_norm': slice_1d['grad_norm'],
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
    grad_norm = K.variable(0., dtype='float32')

    self._slice_1d = {
      'points': points,
      'losses': losses,
      'num_of_points': num_of_points,
      'point_index': point_index,
      'step_norm': step_norm,
      'grad_norm': grad_norm,
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

    def compute_losses_of_all_points():
      def on_all_points_computed():
        with ops.control_dependencies(
          [p_prev.assign(p_cache) for p_prev, p_cache in zip(P_prev, P_cache)] +
          [p.assign(p_cache) for p, p_cache in zip(P, P_cache)]
        ):
          return K.constant(1)

      def move_to_next_point():
        alpha_S = (self.tiny_norm / step_norm) * points[point_index]
        with ops.control_dependencies(
          [p.assign(p_prev - s * alpha_S) for p, p_prev, s in zip(P, P_prev, S)]
        ):
          return K.constant(0)

      with ops.control_dependencies([
        losses[point_index - 1].assign(loss) # this loss is loss of the previous point
      ]):
        with ops.control_dependencies([
          control_flow_ops.cond(
            math_ops.less(point_index, num_of_points),
            move_to_next_point,
            on_all_points_computed)
        ]):
          return point_index.assign(point_index + 1)

    def train_and_slice():
      def do_slice():
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

        with ops.control_dependencies(
          [s.assign(p_prev - p) for s, p_prev, p in zip(S, P_prev, P)]
        ):
          with ops.control_dependencies(
            [p_cache.assign(p) for p_cache, p in zip(P_cache, P)]
          ):
            with ops.control_dependencies([
              step_norm.assign(math_ops.sqrt(math_ops.reduce_sum([math_ops.reduce_sum(math_ops.square(s)) for s in S])))
            ]):
              return control_flow_ops.cond(
                math_ops.equal(step_norm, 0.),
                lambda: num_of_points.assign(0),
                compute_points)

      def do_train():
        grads = self.optimizer.get_gradients(loss, params)
        grad_norm_new = math_ops.sqrt(math_ops.reduce_sum([math_ops.reduce_sum(math_ops.square(g)) for g in grads]))

        with ops.control_dependencies([
          grad_norm.assign(grad_norm_new),
        ]):
          return self.optimizer.get_updates(loss, P)

      with ops.control_dependencies(
        do_train() +
        [
          num_of_points.assign(0),
          losses[0].assign(loss),
          iterations.assign_add(1),
        ]
      ):
        with ops.control_dependencies([
          do_slice(),
        ]):
          # loss of point 0 is already computed.
          # loss of point 1 will be computed at next iteration.
          # point_index = 2 means params will be moved to point 2 at next iteration after loss of point 1 is computed.
          return point_index.assign(2)

    with ops.control_dependencies([
      store_initial_params() # this function is called only once.
    ]):
      return [
        control_flow_ops.cond(
          math_ops.less_equal(point_index, num_of_points),
          compute_losses_of_all_points,
          train_and_slice) # (initially point_index = 2 and num_of_points = 0) or (point_index = num_of_points + 1)
      ]
