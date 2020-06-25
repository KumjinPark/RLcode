import numpy as np
from collections import deque
import random


class SumTree:
    def __init__(self, capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "Capacity must be positive and a power of 2."
        self.tree = np.zeros((2*capacity-1, ))
        self.capacity = capacity
        self.cursor = 0

    def add(self, p):
        idx = self.capacity - 1 + self.cursor
        self.update(idx, p)
        self.cursor = (self.cursor + 1) % self.capacity

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self.update_parent(idx, change)

    def update_parent(self, idx, change):
        if idx == 0:
            return
        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self.update_parent(parent_idx, change)


class MinTree:
    def __init__(self, capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "Capacity must be positive and a power of 2."
        self.tree = np.zeros((2*capacity-1, ))
        self.capacity = capacity
        self.cursor = 0

    def add(self, p):
        idx = self.cursor + self.capacity - 1
        self.update(idx, p)
        self.cursor = (self.cursor + 1) % self.capacity

    def update(self, idx, p):
        self.tree[idx] = p
        self.update_parent(idx, p)

    def update_parent(self, idx, p):
        if idx == 0:
            return
        parent_idx = (idx - 1) // 2
        parent_p = self.tree[parent_idx]
        if p < parent_p:
            self.tree[parent_idx] = p
            if parent_idx != 0:
                self.update_parent(parent_idx, p)

    def total_min(self):
        return self.tree[0]


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, e):
        self.buffer.append(e)

    def sample(self, batch_size):
        return random.sample(self.buffer, k=batch_size)

    def __len__(self):
        return len(self.buffer)


class PrioritizedBuffer:
    def __init__(self, buffer_size, alpha, beta_0, beta):
        self.tree = SumTree(buffer_size)
        self.buffer = []
        self.cursor = 0
        self.buffer_size = buffer_size
        self.tree_size = 2*buffer_size - 1
        self.alpha = 0
        self.beta_0 = beta_0
        self.beta = beta

    def push(self, e, p):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(e)
        else:
            self.buffer[self.cursor] = e
        self.cursor = (self.cursor + 1) % self.buffer_size
        p = p ** self.alpha
        self.tree.add(p)

    def retrieve(self, s, idx):
        left_idx = 2*idx + 1
        right_idx = left_idx + 1
        if left_idx >= self.tree_size:
            return idx

        left = self.tree[left_idx]
        right = self.tree[right_idx]
        if left > s:
            self.retrieve(s, left_idx)
        else:
            self.retrieve(s-right, right_idx)

    def sample(self, batch_size):
        len_seg = int(self.buffer_size / batch_size)
        indices = []
        mini_batch = []
        weights = []
        for i in range(batch_size):
            s = random.random*len_seg + i*len_seg
            idx = self.retrieve(s, 0)
            indices.append(idx)
            p = self.tree[idx]
            weight = (1/(p*batch_size)) ** self.beta
            weights.append(weight)
            idx -= (self.buffer_size - 1)
            mini_batch.append(self.buffer[idx])

        return mini_batch, weights, indices

    def priority_update(self, indices, new_p):
        


#%%
