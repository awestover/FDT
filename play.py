import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from deps import *

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


def play_agent(env, agent, ax):
    state = env.reset()
    for i in range(50):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        if done:
            break
        ax.clear()
        ax.imshow(state)
        ax.set_title(f"Step {i}")
        plt.pause(0.01)

if __name__ == '__main__':
    env = GridWorldEnv(max_steps=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = init_agent(device)
    checkpoint_path = 'checkpoints/checkpoint_100.pth'
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

    #  create_movie(env, agent, ax, movie_filename='agent_journey.mp4')

