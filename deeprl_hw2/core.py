"""Core classes."""

import random
import numpy as np

class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, window_length, input_shape, batch_size):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.window_length = window_length
        self.max_size = max_size
        self.batch_size = batch_size
        self.states = np.empty((max_size,) + input_shape, dtype = np.uint8)
        self.actions = np.empty(max_size, dtype = np.uint8)
        self.rewards = np.empty(max_size, dtype = np.integer)
        self.terminals = np.empty(max_size, dtype = np.bool)
        self.current = 0
        self.size = 0

        self.prestates = np.empty((batch_size, window_length) + input_shape, dtype = np.uint8)
        self.poststates = np.empty((batch_size, window_length) + input_shape, dtype = np.uint8)

    def append(self, state, action, reward, terminal):
        self.states[self.current, ...] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.current += 1
        if self.size < self.max_size:
            self.size += 1
        if (self.current >= self.max_size):
            self.current = 0

    def getState(self, index):
        # if is not in the beginning of matrix
        if index >= self.window_length - 1:
            # use faster slicing
            return self.states[(index - (self.window_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [index - i for i in reversed(range(self.window_length))]
            return self.states[indexes, ...]

    def sample(self):
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index 
            while True:
              # sample one index (ignore 
              index = random.randint(0, self.size - 1)
              # if states wrap over, the memory must be filled up
              if index < self.window_length and self.size < self.max_size:
                  continue
              # if wraps over current pointer, then get new one
              if index >= self.current and index - self.window_length < self.current:
                  continue
              # if wraps over episode end, then get new one
              # NB! poststate (last screen) can be terminal state!
              if self.terminals[(index - self.window_length):index].any():
                  continue
              # otherwise use this index
              break
            
            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates.astype(np.float32), actions, rewards, \
                self.poststates.astype(np.float32), terminals


    def clear(self):
        self.memory = [None] * self.max_size
        self.current = 0
        self.size = 0
