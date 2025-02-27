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

# Initialize plotting at the beginning of your main() function
def setup_plotting():
    """
    Set up interactive plotting for real-time training visualization.
    
    Returns:
        tuple: Figure, axes, and data lists for real-time plotting
    """
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Set up the plots
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    reward_line, = axes[0].plot([], [], 'b-')
    eval_scatter = axes[0].scatter([], [], color='r', s=30, label='Evaluation')
    axes[0].legend()
    
    axes[1].set_title('Episode Lengths')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    length_line, = axes[1].plot([], [], 'g-')
    
    axes[2].set_title('Training Loss')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Loss')
    loss_line, = axes[2].plot([], [], 'r-')
    
    axes[3].set_title('Exploration Rate (ε)')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Epsilon')
    eps_line, = axes[3].plot([], [], 'k-')
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Create lists to store data
    data = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'losses': [],
        'epsilons': [],
        'eval_episodes': [],
        'eval_rewards': []
    }
    
    # Pack everything into a dict to return
    plot_elements = {
        'fig': fig,
        'axes': axes,
        'lines': {
            'reward': reward_line,
            'length': length_line,
            'loss': loss_line,
            'epsilon': eps_line,
            'eval_scatter': eval_scatter
        },
        'data': data
    }
    
    return plot_elements

def update_plots(plot_elements, episode, reward, length, loss, epsilon, is_eval=False, eval_reward=None):
    """
    Update the interactive plots with new data.
    
    Args:
        plot_elements (dict): Dictionary containing plot elements
        episode (int): Current episode number
        reward (float): Episode reward
        length (int): Episode length
        loss (float): Episode loss
        epsilon (float): Current epsilon value
        is_eval (bool): Whether this is an evaluation episode
        eval_reward (float): Evaluation episode reward
    """
    data = plot_elements['data']
    lines = plot_elements['lines']
    axes = plot_elements['axes']
    fig = plot_elements['fig']
    
    # Update data lists for regular training episodes
    if not is_eval:
        data['episodes'].append(episode)
        data['rewards'].append(reward)
        data['lengths'].append(length)
        if loss is not None:
            data['losses'].append(loss)
        data['epsilons'].append(epsilon)
        
        # Update line plots
        lines['reward'].set_data(data['episodes'], data['rewards'])
        lines['length'].set_data(data['episodes'], data['lengths'])
        
        if data['losses']:
            lines['loss'].set_data(data['episodes'][:len(data['losses'])], data['losses'])
        
        lines['epsilon'].set_data(data['episodes'], data['epsilons'])
    
    # Update evaluation scatter plot
    else:
        data['eval_episodes'].append(episode)
        data['eval_rewards'].append(eval_reward)
        lines['eval_scatter'].set_offsets(np.column_stack((data['eval_episodes'], data['eval_rewards'])))
    
    # Rescale axes if needed
    for i, ax in enumerate(axes):
        ax.relim()
        ax.autoscale_view()
    
    # Draw and flush events
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

plot_elements = setup_plotting()

"""
Optimized training loop for DQN agent with performance enhancements.
"""
# Training parameters
num_episodes = 1000
save_every = 100
eval_every = 20
checkpoint_dir = "./checkpoints"

# Create checkpoints directory
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize environment and agent
env = GridWorldEnv(max_steps=100)
agent = DQNAgent(lr=1e-4, gamma=0.99, buffer_capacity=50000, batch_size=128)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
agent.policy_net.to(device)
agent.target_net.to(device)

# Metrics tracking
all_rewards = []
episode_lengths = []
all_losses = []

# Warmup phase - collect initial experiences with random actions
print("Warming up replay buffer...")
state = env.reset()
for _ in range(min(1000, agent.replay_buffer.capacity // 10)):
    action = random.randint(0, 3)  # Random action
    next_state, reward, done = env.step(action)
    agent.replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    if done:
        state = env.reset()

print(f"Starting training for {num_episodes} episodes")

# Main training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    episode_loss = 0
    steps = 0
    losses = []
    
    # Episode loop
    done = False
    while not done:
        # Select and execute action
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        episode_reward += reward
        steps += 1
        
        # Store transition
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        
        # Learn from experience
        loss = agent.update()
        if loss is not None:
            losses.append(loss)
    
    # Update exploration rate
    agent.set_epsilon(episode, num_episodes)
    
    # Calculate metrics
    episode_lengths.append(steps)
    all_rewards.append(episode_reward)
    if losses:
        episode_loss = sum(losses) / len(losses)
        all_losses.append(episode_loss)
    
    # Print progress
    if (episode + 1) % 10 == 0:
        avg_reward = sum(all_rewards[-10:]) / 10
        avg_length = sum(episode_lengths[-10:]) / 10
        avg_loss = sum(all_losses[-10:]) / 10 if all_losses else 0
        print(f"Episode {episode+1}/{num_episodes} | " 
              f"Avg Reward: {avg_reward:.2f} | "
              f"Avg Length: {avg_length:.1f} | "
              f"Avg Loss: {avg_loss:.6f} | "
              f"Epsilon: {agent.epsilon:.3f}")
    
        # Update the plots with latest data
        update_plots(
            plot_elements, 
            episode + 1,
            episode_reward,  # Current episode reward
            steps,           # Current episode length
            episode_loss,    # Current episode loss
            agent.epsilon    # Current epsilon
        )

    # Save checkpoint
    if (episode + 1) % save_every == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode+1}.pth")
        torch.save({
            'model_state_dict': agent.policy_net.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at episode {episode+1}")
    
    # Evaluation phase
    if (episode + 1) % eval_every == 0:
        eval_rewards = []
        agent.epsilon = 0  # No exploration during evaluation
        
        for _ in range(5):  # Run 5 evaluation episodes
            eval_state = env.reset()
            eval_reward = 0
            eval_done = False
            
            while not eval_done:
                eval_action = agent.select_action(eval_state)
                eval_state, reward, eval_done = env.step(eval_action)
                eval_reward += reward
            
            eval_rewards.append(eval_reward)
        
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        print(f"Evaluation: Avg Reward = {avg_eval_reward:.2f}")
        
        # Reset epsilon for training
        agent.set_epsilon(episode, num_episodes)
        
        # After running evaluation episodes and calculating avg_eval_reward
        update_plots(
            plot_elements,
            episode + 1,
            None,
            None,
            None,
            None,
            is_eval=True,
            eval_reward=avg_eval_reward
        )

# Plot training metrics
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(all_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(2, 2, 2)
plt.plot(episode_lengths)
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps')

if all_losses:
    plt.subplot(2, 2, 3)
    plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

print("Training completed!")

