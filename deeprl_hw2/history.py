import numpy as np
from collections import deque

class History:
  def __init__(self, history_length=1):
        self.history = deque(maxlen=history_length)
        self.history_length = history_length;
    self.cnn_format = config.cnn_format

    batch_size, history_length, screen_height, screen_width = \
        config.batch_size, config.history_length, config.screen_height, config.screen_width

    self.history = np.zeros(
        [history_length, screen_height, screen_width], dtype=np.float32)

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose(self.history, (1, 2, 0))
    else:
      return self.history
