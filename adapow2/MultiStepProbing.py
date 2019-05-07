import numpy as np
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from . import utils

class MultiStepProbing(Optimizer):
  """MultiStepProbing is an adaptive gradient descent optimizer, which adjusting the power of 2 of a tiny step size using multiple steps probing.

    MultiStepProbing periodically adjusts the step size, once per several batches. When adjusting, the optimizer starts from a step size smaller than the current one, applies gradient descent (gd) multiple times using a fixed step size, records loss, then doubles step size and applies gd multiple times and records loss again, compares the losses of different step size and chooses the better one.

    Arguments:
      config['probing_steps_num']: int > 0. number of gd steps to get a loss value when adjusting step size.
      config['adjustment_interval']: int > 0. number of batches per adjustment.
      config['tiny_step_size']: float > 0. the tiny unit step size on one parameter.
      config['static_pow2']: list of float. a list of pow2 for early epochs, the last pow2 for the remaining epochs. If static_pow2 is not None, step size will not be adjusted automatically. e.g. config['static_pow2'] = list(numpy.linspace(max_pow2, min_pow2, epoch_number))
      config['loss_ema_beta']: float >= 0. beta parameter for ema average of loss.
      config['store_history_state']: bool. If True, storing 'pow2, loss, loss_ema' of each batch in config['history_state_path'] directory as json file and plot image.
      config['history_state_path']: string. directory path for storing history state.
  """

  def __init__(self, config = None, **kwargs):
    super(MultiStepProbing, self).__init__(**kwargs)
    default_config = {
      'probing_steps_num': 10,
      'adjustment_interval': 300,
      'tiny_step_size': 1e-6,
      'static_pow2': None,
      'loss_ema_beta': 0.99,
      'store_history_state': False,
      'history_state_path': 'data.MultiStepProbing',
    }
    if config is None:
      self._config = default_config
    else:
      self._config = dict(list(default_config.items()) + list(config.items()))

    self.stop_training_batch = K.variable(True, dtype='bool')
    self.pow2 = K.variable(5., dtype='float32')

  def get_config(self):
    base_config = super(MultiStepProbing, self).get_config()
    return dict(list(base_config.items()) + list(self._config.items()))

  def on_epoch_begin(self, epoch):
    self._history_state = None

    if self._config['static_pow2'] is not None:
      static_pow2 = self._config['static_pow2'] if isinstance(self._config['static_pow2'], list) else [self._config['static_pow2']]
      K.get_session().run([
        self.pow2.assign(static_pow2[min(epoch, len(static_pow2) - 1)]),
      ])

  def on_epoch_end(self, epoch, epoch_logs):
    if self._config['store_history_state']:
      utils.mkdir_p(self._config['history_state_path'])
      utils.store_history_state(self._history_state, self._config['history_state_path'] + '/epoch-' + str(epoch + 1))

  def on_iteration_end(self, batch_logs = None):
    if self._config['store_history_state']:
      if self._history_state is None:
        self._history_state = {}
        for k, _ in self.state.items():
          self._history_state[k] = []
      state = K.get_session().run(self.state)
      for k, v in state.items():
        self._history_state[k].append(v)

    return K.get_session().run(self.stop_training_batch)

  def get_updates(self, loss, params):
    prev_params = [K.zeros(K.int_shape(p), dtype='float32') for p in params]
    cached_grads = [K.ones(K.int_shape(p), dtype='float32') for p in params]

    grads = self.get_gradients(loss, params)
    grad_norm = math_ops.sqrt(math_ops.reduce_sum([math_ops.reduce_sum(math_ops.square(g)) for g in grads]))

    alpha_cached_grads = K.variable(0., dtype='float32')
    tiny_step_norm = np.float32(self._config['tiny_step_size'] * np.sqrt(np.sum([np.prod(K.int_shape(p)) for p in params])))

    prev_pow2 = K.variable(0., dtype='float32')
    loss_now = K.variable(np.inf, dtype='float32')
    loss_cmp = K.variable(np.inf, dtype='float32')

    loss_max = K.variable(-np.inf, dtype='float32')
    loss_min = K.variable(np.inf, dtype='float32')

    loss_ema_i = K.variable(0., dtype='float32')
    loss_ema_unfixed = K.variable(0., dtype='float32')
    loss_ema = K.variable(0., dtype='float32')

    adjustment_timer = K.variable(self._config['adjustment_interval'], dtype='int32')
    adjustment_try_count = K.variable(0, dtype='int32')
    probing_step_i = K.variable(0, dtype='int32')
    code = K.variable(0, dtype='float32')

    self.state = {
      'pow2': self.pow2,
      'loss_now': loss_now,
      'loss_ema': loss_ema,
    }

    def cache_grads():
      with ops.control_dependencies(
        [g.assign(new_g) for g, new_g in zip(cached_grads, grads)] + 
        [alpha_cached_grads.assign(math_ops.pow(2., self.pow2) * tiny_step_norm / grad_norm)]
      ):
        return K.constant(0)

    def on_zero_grads():
      return K.constant(0)

    def on_descent():
      with ops.control_dependencies([
        control_flow_ops.cond(math_ops.equal(grad_norm, 0.), on_zero_grads, cache_grads),
        self.stop_training_batch.assign(True),
        adjustment_timer.assign_add(1),
        loss_now.assign(loss),
        loss_max.assign(math_ops.maximum(loss_max, loss)),
        loss_min.assign(math_ops.minimum(loss_min, loss)),
        loss_ema_unfixed.assign(loss * (1 - self._config['loss_ema_beta']) + loss_ema_unfixed * self._config['loss_ema_beta']),
        loss_ema_i.assign_add(1.),
      ]):
        with ops.control_dependencies(
          [p.assign_sub(g * alpha_cached_grads) for p, g in zip(params, cached_grads)] +
          [
            loss_ema.assign(loss_ema_unfixed / (1 - math_ops.pow(self._config['loss_ema_beta'], loss_ema_i))),
          ]
        ):
          return K.constant(0)

    def on_adjust():
      def on_best_step():
        with ops.control_dependencies([
            self.pow2.assign(math_ops.log((math_ops.pow(2., self.pow2 - 1.) + math_ops.pow(2., prev_pow2)) / 2) / math_ops.log(2.)),
            adjustment_try_count.assign(0),
            adjustment_timer.assign(0),
            loss_max.assign(-np.inf),
            loss_min.assign(np.inf),
            ]):
          return K.constant(5)

      def on_better_step():
        with ops.control_dependencies([
          self.pow2.assign_add(1.),
          loss_cmp.assign(loss_now),
          adjustment_try_count.assign_add(1),
        ]):
          return K.constant(3)

      def on_probing_done():
        with ops.control_dependencies(
          [p.assign(p_prev) for p, p_prev in zip(params, prev_params)] +
          [probing_step_i.assign(0)]
        ):
          return control_flow_ops.cond(
              math_ops.less(loss_now, loss_cmp),
              on_better_step,
              on_best_step)

      def on_probing_doing():
        with ops.control_dependencies([
          control_flow_ops.cond(
            math_ops.equal(grad_norm, 0.),
            on_zero_grads,
            cache_grads)
        ]):
          with ops.control_dependencies(
            [p.assign_sub(g * alpha_cached_grads) for p, g in zip(params, cached_grads)] +
            [probing_step_i.assign_add(1)]
          ):
            return K.constant(2)

      def init_adjustment():
        def do_init_adjustment():
          with ops.control_dependencies(
            [p_prev.assign(p) for p_prev, p in zip(prev_params, params)] +
            [
              adjustment_try_count.assign(1),
              self.stop_training_batch.assign(False),
              probing_step_i.assign(0),
              loss_cmp.assign(loss_now),
              prev_pow2.assign(self.pow2),
            ]
          ):
            with ops.control_dependencies([
              self.pow2.assign(math_ops.maximum(0., self.pow2 - 1)),
            ]):
              return K.constant(0)

        def try_next_batch():
          with ops.control_dependencies([
            self.stop_training_batch.assign(False),
            adjustment_timer.assign_sub(1),
          ]):
            return K.constant(0)

        loss_avg, loss_nth_up, loss_nth_down = control_flow_ops.cond(
          math_ops.equal(loss_ema_i, 0),
          lambda: (loss_now, 0., 0.),
          lambda: (loss_ema, (loss_max - loss_ema) / 3., (loss_ema - loss_min) / 3.))

        return control_flow_ops.cond(
          math_ops.logical_and(math_ops.greater_equal(loss_now, loss_avg), math_ops.less_equal(loss_now, loss_avg + loss_nth_up)),
          do_init_adjustment,
          try_next_batch)

      with ops.control_dependencies([
        loss_now.assign(loss),
        control_flow_ops.cond(
          math_ops.equal(adjustment_try_count, 0),
          init_adjustment,
          lambda: K.constant(0)),
      ]):
        return control_flow_ops.cond(
          math_ops.equal(adjustment_try_count, 0),
          lambda: K.constant(0),
          lambda: control_flow_ops.cond(
            math_ops.equal(probing_step_i, self._config['probing_steps_num']),
            on_probing_done,
            on_probing_doing))

    if self._config['static_pow2'] is None:
      new_code = control_flow_ops.cond(
        math_ops.less(adjustment_timer, self._config['adjustment_interval']),
        on_descent,
        on_adjust)
    else:
      new_code = on_descent()

    self.updates = [code.assign(new_code)]
    return self.updates
