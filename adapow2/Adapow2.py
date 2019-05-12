from .x import MultiStepProbing

class Adapow2(MultiStepProbing):
  def __init__(self):
    super(Adapow2, self).__init__({'history_state_path': 'data.Adapow2'})
