#!/usr/bin/env python
# coding: utf-8

# # DQN With Prioritized Replay Buffer
# Use prioritized replay buffer to train a DQN agent.

# In[11]:


import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

import gym
import numpy as np

from gym.core import ObservationWrapper
from gym.spaces import Box
import cv2
import os

import atari_wrappers # adjust env
from framebuffer import FrameBuffer # stack 4 consec images 


# In[12]:


ENV_NAME = "BreakoutNoFrameskip-v4"

# create break-out env
env = gym.make(ENV_NAME)
env.reset()


# ## Preprocessing
# Crop the important part of the image, then resize to 64 x 64

# In[13]:


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.image_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.image_size)

    def observation(self, img):
        """what happens to each observation"""

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize from any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type

        # crop the image 
        # remove the top part
        img = img[50:]

        # resize the image
        img = cv2.resize(img, dsize=(self.image_size[1], self.image_size[2]))

        # gray scale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # normalize to (0, 1)
        img = img.astype(np.float32) / 255.0

        # add channel dimension
        return img[None]

# adjust the env by some wrappers
def PrimaryAtariWrap(env, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id

    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env
    
def make_env(clip_rewards=True, seed=None):
    env = gym.make(ENV_NAME)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env

env = make_env()
env.reset()
n_actions = env.action_space.n
state_shape = env.observation_space.shape
print("adjusted env with 4 consec images stacked can be created")


# ## Model

# In[14]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv2d_size_out(size, kernel_size, stride):
    """
    common use case:
    cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
    cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
    to understand the shape for dense layer's input
    """
    return (size - (kernel_size - 1) - 1) // stride  + 1

class DuelingDQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        kernel_size = 3
        stride = 2
        self.conv1 = nn.Conv2d(4, 16, kernel_size, stride)
        out_size = conv2d_size_out(state_shape[1], kernel_size, stride)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride)
        out_size = conv2d_size_out(out_size, kernel_size, stride)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, stride)
        out_size = conv2d_size_out(out_size, kernel_size, stride)

        # size of the output tensor after convolution batch_size x 64 x out_size x out_size
        self.linear = nn.Linear(64*out_size*out_size, 256)
        
        # advantage
        self.advantage = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )
        
        # state value
        self.value = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        # qvalues = <YOUR CODE>
        t = self.conv1(state_t)
        t = F.relu(t)
        t = self.conv2(t)
        t = F.relu(t)
        t = self.conv3(t)
        t = F.relu(t)

        t = t.view(state_t.shape[0], -1)
        t = self.linear(t)
        t = F.relu(t)
        
        # compute advantage and state value as different heads
        advantage = self.advantage(t)
        value = self.value(t)
        
        qvalues = value + advantage - advantage.mean(dim=1, keepdim=True)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    
# Evaluate the agent
def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for _ in range(n_games):
        reward = 0.0
        s = env.reset()
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(
                qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break
                
        rewards.append(reward)
    return np.mean(rewards)


# ## Compute TD loss

# In[48]:


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network,
                    gamma=0.99,
                    device=device, check_shapes=False):
    """ Compute td loss using torch operations only. Use the formulae above. '''
    
    objective of agent is 
    \hat Q(s_t, a_t) = r_t + \gamma Target(s_{t+1}, argmax_{a} Q(s_{t+1}, a))    
    """
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    
    is_done = torch.tensor(
        is_done,
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    
    is_not_done = 1 - is_done
    
    # get q-values for all actions in current states
    predicted_qvalues = agent(states)
   
    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)
   
    # best action in next state
    next_best_actions = torch.argmax(agent(states), dim=1)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]
    
        # compute the objective of the agent
    next_state_values = predicted_next_qvalues[range(
        len(actions)), next_best_actions]                          
                                                     
    # assert next_state_values.dim(
    # == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    # target_qvalues_for_actions = <YOUR CODE>

    target_qvalues_for_actions = rewards + next_state_values * is_not_done

    # mean squared error loss adjusted by importance sampling weights to minimize
    #loss = torch.mean(
    #        weights * torch.pow(predicted_qvalues_for_actions - target_qvalues_for_actions.detach(), 2)
    #)
    
    # return the TD-loss
    
    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"
    
    return target_qvalues_for_actions - predicted_qvalues_for_actions


# ## Test the memory need of the replay buffer

# Init DQN agent and play a total 10^4 time steps

# In[19]:


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    sum_rewards = 0.0 
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0] 
        next_s, r, done, _  = env.step(action)

        exp_replay.add((s, action, r, next_s, done))
        sum_rewards += r
        if done:
            s = env.reset()
        else:
            s = next_s

    return sum_rewards, s


# In[39]:

from replay_buffer import PrioritizedReplayBuffer


#n_actions = env.action_space.n
#state_shape = env.observation_space.shape

agent = DuelingDQNAgent(state_shape=state_shape, n_actions=n_actions)
exp_replay = PrioritizedReplayBuffer(10**4)

'''
for i in range(100):
    state = env.reset()
    if not utils.is_enough_ram(min_available_gb=0.1):
        print("""
            Less than 100 Mb RAM available. 
            Make sure the buffer size in not too huge.
            Also check, maybe other processes consume RAM heavily.
            """
             )
        break
    play_and_record(state, agent, env, exp_replay, n_steps=10**2)
    if len(exp_replay) == 10**4:
        break
print(len(exp_replay))

del exp_replay
'''


# In[76]:


seed = 42

# env
n_lives = 5


# training params
T = 1 # number of experiences to get from env before each update
batch_size = 16
total_steps = 3 * 10**1 # total steps to train the agent
decay_steps = 10**1 # steps to decay the epsilon, 
                    # after the decay_steps, epsilon stops decaying
                    # and the agent explores with a fixed probability
max_grad_norm = 50 

            
refresh_target_network_freq = 5000 # freqency to update the target network
learning_rate = 1e-4


# agent 
gamma = 0.99 # discount factor
init_epsilon = 1.0
final_epsilon = 0.1

# buffer
buffer_size = 10**4

# eval
loss_freq = 50 
eval_freq = 5000

# logs 
ckpt_dir = 'logs'
ckpt_file = 'prioritized_experience_replay_ckpt.pth'
metrics_file = 'prioritized_experience_replay_metrics.pth'
ckpt_freq = 10*5000 # Debug param


# In[75]:


# main loop

env = make_env(seed)

state_shape = env.observation_space.shape
n_actions = env.action_space.n
state = env.reset()

agent = DuelingDQNAgent(state_shape, n_actions, epsilon=1).to(device)
target_network = DuelingDQNAgent(state_shape, n_actions).to(device)
target_network.load_state_dict(agent.state_dict())

exp_replay = PrioritizedReplayBuffer(buffer_size)



opt = torch.optim.Adam(agent.parameters(), lr=learning_rate)

mean_rw_history = []
td_loss_history = []
grad_norm_history = []
initial_state_v_history = []

print("Starts training on {}".format(next(agent.parameters()).device))

# populate the buffer with 128 samples
init_size = 128
play_and_record(state, agent, env, exp_replay, init_size)

for step in range(total_steps):
    agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)
    
    # play for $T time steps and cache the exprs to the buffer
    _, state = play_and_record(state, agent, env, exp_replay, T)
    
    b_idx, obses_t, actions, rewards, obses_tp1, dones, weights = exp_replay.sample(
        batch_size)
        
    # td loss for each sample
    td_loss = compute_td_loss(
        states=obses_t, 
        actions=actions, 
        rewards=rewards, 
        next_states=obses_tp1, 
        is_done=dones,
        agent=agent,
        target_network=target_network,
        gamma=gamma,
        device=device,
        check_shapes=True)
                             
    '''
    A batch of samples from prioritized replay looks like:
    (states, actions, rewards, next_states, weights, is_done)
    weights here are importance sampling weights

    Basically:
        Loss = weights * MSE
    
    '''
    
    # compute MSE adjusted by importance sampling weights
    # and backprop
    weights = torch.tensor(weights, dtype=torch.float32)
    #print(weights, torch.pow(td_loss, 2))
    loss = torch.mean(weights * torch.pow(td_loss, 2))
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()
    
    # update the priorities of sampled exprs
    exp_replay.batch_update(b_idx, np.abs(td_loss.detach().cpu().numpy()))
    
    # increase the importance sampling hyperparameter b gradually to 1
    exp_replay.increment_b()
    
    if step % loss_freq == 0:
        # save MSE without importance sampling
        loss = torch.mean(torch.pow(td_loss, 2))
        td_loss_history.append(loss.cpu().item())
    
    if step % refresh_target_network_freq == 0:
        target_network.load_state_dict(agent.state_dict())
    
    if step % eval_freq == 0:
        mean_rw_history.append(evaluate(
            make_env(clip_rewards=True, seed=step),
            agent, n_games=3*n_lives, greedy=True
        ))
        
        initial_state_q_values = agent.get_qvalues(
            [make_env(seed=step).reset()]
        )
    
        initial_state_v_history.append(np.max(initial_state_q_values))
        
        print("buffer size = %i, epsilon: %.5f" % 
             (len(exp_replay), agent.epsilon))
        
    
    # TODO 
    # checkpointing
    if step % ckpt_freq == 0:
        print("checkpointing ...")
        
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            
        # check point model and optimizer
        checkpoint = {
            "step": step,
            "agent": agent.state_dict(),
            "epsilon": agent.epsilon,
            "target_network": target_network.state_dict(),
            "optimizer": opt.state_dict(),
            "replay_buffer": exp_replay
        }
        
        torch.save(checkpoint, os.path.join(ckpt_dir, ckpt_file))
    
        # save the performance metric   
        metrics = {
            "mean_rw_history": mean_rw_history,
            "td_loss_history": td_loss_history,
            "grad_norm_history": grad_norm_history,
            "initial_state_v_history": initial_state_v_history
        }
        
        torch.save(metrics, os.path.join(ckpt_dir, metrics_file))
        
        
# check point model and optimizer
checkpoint = {
    "step": step,
    "agent": agent.state_dict(),
    "epsilon": agent.epsilon,
    "target_network": target_network.state_dict(),
    "optimizer": opt.state_dict(),
    "replay_buffer": exp_replay
}

torch.save(checkpoint, os.path.join(ckpt_dir, ckpt_file))

# save the performance metric   
metrics = {
    "mean_rw_history": mean_rw_history,
    "td_loss_history": td_loss_history,
    "grad_norm_history": grad_norm_history,
    "initial_state_v_history": initial_state_v_history
}

torch.save(metrics, os.path.join(ckpt_dir, metrics_file))

