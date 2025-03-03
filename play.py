import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from deps import *

"""
TODO: there is some horrible bug in this code.
"""


#  def create_movie(env, agent, ax, movie_filename='agent_journey.mp4'):
#      frames = []
#      for i in range(4):
#          state = env.reset()
#          while True:
#              frames.append(state.copy())
#              action = agent.select_action(state)
#              next_state, reward, done = env.step(action)
#              state = next_state
#              if done:
#                  break
    
#      # Initialize the image artist with the first frame
#      img = ax.imshow(frames[0], cmap='hot', interpolation='nearest')
    
#      def update(frame):
#          img.set_data(frame)
#          return [img]
    
#      # Create animation using the frames list
#      FPS = 20
#      ani = FuncAnimation(fig, update, frames=frames, interval=1000/FPS, repeat=False, blit=True)
    
#      # Save the animation
#      ani.save(movie_filename, writer='ffmpeg', fps=FPS)
    
#      print(f"Movie saved as {movie_filename}")

def tensor_to_numpy_grid(tensor):
    """
    Convert a tensor of shape [nchannels, size, size] into a numpy array of shape [size, size]
    where:
    - Value is 1 if the first channel (index 0) has a 1
    - Value is 2 if the second channel (index 1) has a 1
    - Value is 0 otherwise
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [nchannels, size, size]
    
    Returns:
        numpy.ndarray: A size x size numpy array with encoded values
    """
    # Convert tensor to numpy array and get dimensions
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    tensor_np = tensor.detach().numpy()
    _, size1, size2 = tensor_np.shape
    
    # Initialize result array with zeros
    result = np.zeros((size1, size2), dtype=np.int8)
    
    # Set cells to 1 where first channel has 1
    result[tensor_np[0] > 0.5] = 1
    
    # Set cells to 2 where second channel has 1
    # This will override any 1 values if both channels have 1s
    result[tensor_np[1] > 0.5] = 2
    
    return result


def play_agent(env, agent, ax):
    state = env.reset()
    for i in range(50):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        if done:
            break
        ax.clear()
        ax.imshow(tensor_to_numpy_grid(state))
        ax.set_title(f"Step {i}")
        plt.pause(.5)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldEnv(device, max_steps=100)
    agent = DQNAgent(device, lr=1e-4, gamma=0.99, buffer_capacity=50000, batch_size=64, update_target_every=200)
    checkpoint_path = 'checkpoints/checkpoint_10000.pth'
    checkpoint = torch.load(checkpoint_path, weights_only=True) # weights_only=False if it's broken
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0.1
    print(f"Checkpoint loaded from {checkpoint_path}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(env.grid_size))
    ax.set_yticks(np.arange(env.grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(5):
        print("episode ", i)
        play_agent(env, agent, ax)

