"""
We're writing a DQN agent to solve the following game:

state is an 8x8 grid  -- the grid has 0s  -- squares where the agent can move, 1s -- blocked squares  and a single 2 --  representing the AI, and a single 3 representing the "goal" 
the actions that the AI can take are "try to move in any of the directions"
the action fails if the AI tries to move into a wall or off the board 

the AI gets a reward of -1 for every step where it's not at the 3 
once it reaches the 3 the episode ends

this is tricky-- be careful please

You've already given a great implementation, but it needs better documentation
-- Please give some doc strings that describe what the functions do.

"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import matplotlib.pyplot as plt
from deps import *

# Initialize a list to store loss values
losses = []
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Episode")
ax.set_ylabel("Loss")

# --- Training Loop ---
if __name__ == '__main__':
    """
    Main training loop for the DQN agent.

    Runs a specified number of episodes. In each episode, the agent interacts with the environment,
    collects transitions, and learns from them. After training, the agent's performance is tested
    by rendering the environment.
    """
    num_episodes = 500
    save_every = 100
    checkpoint_dir = "./checkpoints"
    
    # Create a directory to save checkpoints if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = GridWorldEnv(max_steps=100)
    agent = DQNAgent()
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Save transition in the replay buffer.
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # Update the agent (learn from random mini-batch)
            loss = agent.update()
            if loss:
                losses.append(loss.detach().numpy())
        
        agent.decay_epsilon()
        print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

        if (episode + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode+1}.pth")
            torch.save({
                'episode': episode + 1,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon
            }, checkpoint_path)
            print(f"Checkpoint saved at episode {episode+1}")

        if (episode + 1) % 5 == 0:
            chunk_size = 100
            n_chunks = len(losses) // chunk_size
            averaged_losses = [np.mean(losses[i*chunk_size:(i+1)*chunk_size]) for i in range(n_chunks)]
            ax.clear()  # Clear the current plot
            ax.plot(averaged_losses, marker='o', color='b')  # Plot up to the current episode
            ax.set_xlabel("Episode")
            ax.set_ylabel("Loss")
            plt.draw()  # Update the figure
            plt.pause(0.25)  # Pause to allow for the plot to update
    
    # Optionally, test the trained agent
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        env.render()


plt.show()

"""
Hypothesis about why this is not working:

It's super slow. 
There are very few weight updates over the whole course of my training. 

1. Figure out how many weight updates happen.
2. Figure out why episodes are so slow.
Intuitively, I'd guess you should be able to do thousands of episodes per second

ok nvm it's actually working fine after reward shaping

"""

