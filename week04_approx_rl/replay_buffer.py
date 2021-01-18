# This code is shamelessly stolen from
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones)
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)


class SumTree:
    """A binary tree to efficiently sample experiences based on
    curmulative priority
    

    """
    def __init__(self, capacity):
        """The capacity needs to be 2**n 
        in order to make a complete binary tree
        
        A complete binary tree would make the testing
        a bit easier 
        """
        self.capacity = capacity

        # use an array to represent the binary tree
        # leaf-nodes are the priorities of the 
        # experiences
        self.tree =  np.zeros(2*capacity - 1)

        # pointer to the nodes in self.tree
        self.tree_ptr = 0 
        
        # experiences
        self.data = np.zeros(capacity, dtype=object)
        
        # pointer to the experience
        self.data_ptr = 0
        
        # maximum priority
        self._max_priority = 0.0

    def add(self, priority, experience):
        """Add a an experience to the buffer"""
        # compute the tree pointer based on the data pointer
        tree_ptr = self.data_ptr + self.capacity - 1
        
        # put the experience into the slot
        self.data[self.data_ptr] = experience

        # update the values of nodes in the tree
        self.update(tree_ptr, priority)
        
        # move data pointer to the next position
        self.data_ptr += 1

        # remove the earliest expr when the storage is full
        # idea is earlier experience comes from weaker policy
        # we wont lose that much by throwing it away
        if self.data_ptr >= self.capacity:
            # if the expr to be removed happens to 
            # be the one with max priority
            # then do a linear scan to find the next 
            # max priority
            if self.tree[-self.capacity] == self._max_priority:
                self._max_priority = np.max(self.tree[-self.capacity+1:])
            self.data_ptr = 0
        return

    def update(self, tree_ptr, priority):
        """Update the value (sum of child nodes)
        when node at $tree_ptr is given the new priority
        $priority
        """
        # update the max priority
        if priority > self._max_priority:
            self._max_priority = priority

        change = priority -  self.tree[tree_ptr]
        self.tree[tree_ptr] = priority
        while tree_ptr != 0:
            tree_ptr = (tree_ptr - 1) // 2
            self.tree[tree_ptr] += change
        return 

    @property
    def total_priority(self):
        return self.tree[0]
    
    @property
    def max_priority(self):
        return self._max_priority

    def get_leaf(self, cp):
        """Get a leaf node in the tree such that 
        the cumulative priority (from left)
        is close to $cp

        Iterative implementation of dfs
        
        Return:
        (tree index of the expr, priority, sampled expr)
        
        """
        parent_ptr = 0
        
        while True:
            left_ptr = 2*parent_ptr + 1
            right_ptr = left_ptr + 1
            
            if left_ptr >= len(self.tree):
                left_ptr = parent_ptr
                break
            else:
                if cp <= self.tree[left_ptr]:
                    parent_ptr = left_ptr
                else:
                    cp-=self.tree[left_ptr]
                    parent_ptr = right_ptr
        
        data_ptr = left_ptr - self.capacity + 1
        return left_ptr, self.tree[left_ptr], self.data[data_ptr]



class PrioritizedReplayBuffer:
    # Hyperparameter used to avoid some expr having
    # 0 probability of being sampled
    # Think of it as the smoothing factor
    PER_e = 0.01 

    # Hyperparamter used to make a trade off between
    # sampling high priority expr vs low priority expr
    PER_a = 0.6

    # Importance sampling exponent
    # increase from the initial value to 1
    # linearly with respect to steps
    PER_b = 0.4

    # Hyperparameter used to clip the TD error
    absolute_error_upper = 1.0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, experience):
        """Push one experience to the buffer 
        give it the maximum priority to make 
        sure it is sampled
        """
        max_priority = self.tree.max_priority

        # if max_priority is 0, then we cannot
        # set the current expr to that
        # because then the expr will never be sampled.
        # use the smooth factor instead
        if max_priority == 0.0:
            max_priority = self.PER_e
        self.tree.add(max_priority, experience)
        return

    def sample(self, batch_size):
        """Sample a batch of experiences according to
        their priorities
        """
        # sampled experiences
        minibatch = []
        
        # indices of sampled experience 
        # in self.tree.data
        b_idx = [0]*batch_size

        priority_segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a, b = priority_segment*i, priority_segment*(i+1)
            # cumulative priority
            cp = np.random.uniform(a, b)
            ix, priority, expr = self.tree.get_leaf(cp)
            b_idx[i] = ix
            minibatch.append(expr)
        return b_idx, minibatch

    def batch_update(self, b_idx, abs_errors):
        """Update the priorities of the batch of 
        experiences the agent is trained on for one step
        """
        abs_errors += self.PER_e
        clipped_errors = np.minimum(
            abs_errors, self.absolute_error_upper
            )
        ps = np.power(clipped_error, self.PER_a)
        for ti, p in zip(b_idx, ps):
            self.tree.update(ti, p)
        return

