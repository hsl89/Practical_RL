# Test the priority replay buffer
from replay_buffer import SumTree
from replay_buffer import PrioritizedReplayBuffer

import numpy as np
from numpy.testing import assert_equal
import random


# What should be tested for SumTree
# Check the comments in 
# test_max_and_total_priority and test_sampling


capacity = 1024
sumtree = SumTree(capacity)

class Experience:
    def __init__(self):
        pass

def test_max_and_total_priority(capacity):
    # Total priority should be computed correctly when
    # new exprs are inserted

    # Max priority should be computed corrected when 
    # new exprs are inserted
    sumtree = SumTree(capacity)
    total_priority = 0
    for i in range(capacity):
        priority = i
        data = Experience()
        sumtree.add(priority, data)

        total_priority += priority
        assert sumtree.max_priority == priority
        assert sumtree.total_priority == total_priority


    for i in range(capacity, capacity + capacity):
        replace = i - capacity 
        priority = i
        data = Experience()
        sumtree.add(priority, data)

        total_priority += (i - replace)
        assert sumtree.max_priority == priority
        assert sumtree.total_priority == total_priority
    return

def test_sampling(capacity):
    # If experiences are sorted by (in ascending order) by
    # their priority. Then, if we sample a leaf node 
    # by total priority, I should get the 


    # If I sample n exprs according to the right boundary
    # of then I should get roughly n exprs with top priorties
    # to make things simple, I should insert experience
    # in accending order according to their priorities
    
    sumtree = SumTree(capacity)
    for i in range(capacity):
        priority = i
        sumtree.add(priority, Experience())
    
    # sample 10 experieces
    n = 10
    seg = sumtree.total_priority / n
    
    batch = []
    for i in range(n):
        L, R = seg*i, seg*(i+1)
        ix, priority, expr = sumtree.get_leaf(R)
        print(R, priority)
        batch.append(priority)
    
    print(batch)
    print(sumtree.tree[-n:])
    return


def test_update(capacity):
    # When priorities of a batch of experiences 
    # the sum at each node in the tree should have 
    # the correct value
    # the maximum priority should also be updated
    sumtree = SumTree(capacity)
    
    for i in range(capacity):
        priority = i
        sumtree.add(priority, Experience())
    
    # sample n leaf nodes at random
    # tree indices in [capacity - 1, 2*capacity - 2]
    n = 10
    idx = random.sample([x for x in range(capacity-1, 2*capacity-1)], n)
    
    new_priority = [random.randint(0, capacity*2)]
    
    for i, p in zip(idx, new_priority):
        sumtree.update(i, p)
    
    # test the maximum priority
    assert sumtree.max_priority == np.max(sumtree.tree[-capacity:])
    
    # test the node value 
    # by building a new tree from the updated priorities
    ps = [i for i in range(capacity)]
    for i, p in zip(idx, new_priority):
        # i is tree index
        # tree index to data index
        data_idx = i - (capacity - 1)
        ps[data_idx] = p
    
    new_tree = SumTree(capacity)
    for p in ps:
        new_tree.add(p, Experience())
        
    # sumtree == new_tree
    assert_equal(sumtree.tree, new_tree.tree)
        
    

if __name__ == '__main__':
    print('======== Testing ===========')
    test_max_and_total_priority(capacity)
    test_sampling(capacity)
    
    test_update(capacity)