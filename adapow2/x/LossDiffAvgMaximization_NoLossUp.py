import numpy as np
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from .. import utils

class LossDiffAvgMaximization_NoLossUp(Optimizer):
  """adapow2.LossDiffAvgMaximization_NoLossUp is an adaptive gradient descent optimizer, which adjusting the power of 2 of a tiny step size by checking loss difference average up or down after two gradient descent steps on each batch.

    Arguments:
      config['tiny_step_size']: float > 0. the tiny unit step size on one parameter.
      config['pow2_delta']: float > 0. pow2 of step size increase or decrease delta if loss down.
      config['store_history_state']: bool. If True, storing 'pow2, loss, loss_ema' of each batch in config['history_state_path'] directory as json file and plot image.
      config['history_state_path']: string. directory path for storing history state.
  """
  def __init__(self, config = None, **kwargs):
    super(LossDiffAvgMaximization_NoLossUp, self).__init__(**kwargs)
    default_config = {
      'tiny_step_size': 1e-6,
      'pow2_delta': 0.1,
      'loss_diff_ema_beta': 0.99,
      'store_history_state': False,
      'history_state_path': 'data.LossDiffAvgMaximization_NoLossUp',
    }
    if config is None:
      self._config = default_config
    else:
      self._config = dict(list(default_config.items()) + list(config.items()))

    self.stop_training_batch = K.variable(True, dtype='bool')
    self.pow2 = K.variable(5., dtype='float32')

  def get_config(self):
    base_config = super(LossDiffAvgMaximization_NoLossUp, self).get_config()
    return dict(list(base_config.items()) + list(self._config.items()))

  def on_epoch_begin(self, epoch_logs):
    self._history_state = None

  def on_epoch_end(self, epoch_logs):
    if self._config['store_history_state']:
      utils.mkdir_p(self._config['history_state_path'])
      utils.store_history_state(self._history_state, self._config['history_state_path'] + '/epoch-' + str(epoch_logs['epoch'] + 1))

  def on_iteration_end(self, batch_logs = None):
    stop = K.get_session().run(self.stop_training_batch)

    if self._config['store_history_state'] and stop:
      if self._history_state is None:
        self._history_state = {}
        for k, _ in self.state.items():
          self._history_state[k] = []
      state = K.get_session().run(self.state)
      for k, v in state.items():
        self._history_state[k].append(v)

    return stop

  def get_updates(self, loss, params):
    cached_grads = [K.zeros(K.int_shape(p), dtype='float32') for p in params]

    grads = self.get_gradients(loss, params)
    grad_norm = math_ops.sqrt(math_ops.reduce_sum([math_ops.reduce_sum(math_ops.square(g)) for g in grads]))

    alpha_cached_grads = K.variable(0., dtype='float32')
    tiny_step_norm = np.float32(self._config['tiny_step_size'] * np.sqrt(np.sum([np.prod(K.int_shape(p)) for p in params])))

    loss_now = K.variable(np.inf, dtype='float32')
    loss_cmp = K.variable(np.inf, dtype='float32')

    loss_diff = K.variable(np.inf, dtype='float32')
    loss_diff_ema_i = K.variable(0., dtype='float32')
    loss_diff_ema_unfixed = K.variable(0., dtype='float32')
    loss_diff_ema = K.variable(0., dtype='float32')
    loss_direction = K.variable(1., dtype='float32')

    code = K.variable(0, dtype='float32')

    self.state = {
      'pow2': self.pow2,
      'loss_now': loss_now,
      'loss_diff': loss_diff,
      'loss_diff_ema': loss_diff_ema,
      'code': code,
    }

    def on_loss_down():
      stop_curr = math_ops.logical_not(self.stop_training_batch)
      with ops.control_dependencies([
        loss_diff.assign(control_flow_ops.cond(
          stop_curr,
          lambda: loss_cmp - loss_now,
          lambda: loss_diff)),
        loss_diff_ema_i.assign_add(control_flow_ops.cond(
          stop_curr,
          lambda: 1.,
          lambda: 0.)),
        loss_diff_ema_unfixed.assign(control_flow_ops.cond(
          stop_curr,
          lambda: (loss_cmp - loss_now) * (1 - self._config['loss_diff_ema_beta']) +
            loss_diff_ema_unfixed * self._config['loss_diff_ema_beta'],
          lambda: loss_diff_ema_unfixed)),
      ]):
        new_loss_diff_ema = loss_diff_ema_unfixed / (1 - math_ops.pow(self._config['loss_diff_ema_beta'], loss_diff_ema_i))
        new_loss_direction = control_flow_ops.cond(
          math_ops.less_equal(new_loss_diff_ema, loss_diff_ema),
          lambda: -loss_direction,
          lambda: loss_direction)

        with ops.control_dependencies([
          self.pow2.assign_add(control_flow_ops.cond(
            stop_curr,
            lambda: (0. + new_loss_direction) * self._config['pow2_delta'],
            lambda: 0.)),
        ]):
          alpha_weaken = control_flow_ops.cond(stop_curr, lambda: 0.2, lambda: 1.)
          alpha_grads = math_ops.pow(2., self.pow2) * tiny_step_norm / grad_norm

          with ops.control_dependencies(
            [p.assign_sub(g * (alpha_grads * alpha_weaken)) for p, g in zip(params, grads)] +
            [g.assign(new_g) for g, new_g in zip(cached_grads, grads)] + 
            [
              alpha_cached_grads.assign(alpha_grads),
              self.stop_training_batch.assign(stop_curr),
              loss_cmp.assign(control_flow_ops.cond(stop_curr, lambda: np.inf, lambda: loss_now)),
              loss_diff_ema.assign(control_flow_ops.cond(stop_curr, lambda: new_loss_diff_ema, lambda: loss_diff_ema)),
              loss_direction.assign(control_flow_ops.cond(stop_curr, lambda: new_loss_direction, lambda: loss_direction)),
            ]
          ):
            return K.constant(0)

    def on_loss_up():
      with ops.control_dependencies([
        loss_diff.assign(loss_cmp - loss_now),
        loss_diff_ema_i.assign_add(1),
        loss_diff_ema_unfixed.assign(
          (loss_cmp - loss_now) * (1 - self._config['loss_diff_ema_beta']) +
          loss_diff_ema_unfixed * self._config['loss_diff_ema_beta'],
        ),
      ]):
        new_loss_diff_ema = loss_diff_ema_unfixed / (1 - math_ops.pow(self._config['loss_diff_ema_beta'], loss_diff_ema_i))
        new_loss_direction = control_flow_ops.cond(
          math_ops.less_equal(new_loss_diff_ema, loss_diff_ema),
          lambda: -loss_direction,
          lambda: loss_direction)

        with ops.control_dependencies(
          [p.assign_add(g * (alpha_cached_grads * 0.9)) for p, g in zip(params, cached_grads)] +
          [
            self.stop_training_batch.assign(True),
            loss_cmp.assign(np.inf),
            self.pow2.assign_add((0. + new_loss_direction) * self._config['pow2_delta']),
          ]
        ):
          with ops.control_dependencies([
            loss_diff_ema.assign(new_loss_diff_ema),
            loss_direction.assign(new_loss_direction),
          ]):
            return K.constant(1)

    def on_tiny_loss():
      with ops.control_dependencies([
        self.stop_training_batch.assign(True),
        loss_cmp.assign(np.inf),
      ]):
        return K.constant(2)

    def on_zero_grads():
      with ops.control_dependencies([
        self.stop_training_batch.assign(True),
        loss_cmp.assign(np.inf),
      ]):
        return K.constant(3)

    with ops.control_dependencies([
      loss_now.assign(loss),
      self.pow2.assign(math_ops.maximum(0., self.pow2)),
    ]):
      new_code = control_flow_ops.cond(
        math_ops.less_equal(loss_now, 1e-5),
        on_tiny_loss,
        lambda: control_flow_ops.cond(
          math_ops.greater(loss_now, loss_cmp),
          on_loss_up,
          lambda: control_flow_ops.cond(
            math_ops.equal(grad_norm, 0.),
            on_zero_grads,
            on_loss_down)))

      self.updates = [code.assign(new_code)]
      return self.updates
