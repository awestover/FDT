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
    axes[0].set_title("Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    (reward_line,) = axes[0].plot([], [], "b-")
    eval_scatter = axes[0].scatter([], [], color="r", s=30, label="Evaluation")
    axes[0].legend()

    axes[1].set_title("Episode Lengths")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    (length_line,) = axes[1].plot([], [], "g-")

    axes[2].set_title("Training Loss")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Loss")
    (loss_line,) = axes[2].plot([], [], "r-")

    axes[3].set_title("Exploration Rate (ε)")
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Epsilon")
    (eps_line,) = axes[3].plot([], [], "k-")

    plt.tight_layout()
    plt.show(block=False)

    # Create lists to store data
    data = {
        "episodes": [],
        "rewards": [],
        "lengths": [],
        "losses": [],
        "epsilons": [],
        "eval_episodes": [],
        "eval_rewards": [],
    }

    # Pack everything into a dict to return
    plot_elements = {
        "fig": fig,
        "axes": axes,
        "lines": {
            "reward": reward_line,
            "length": length_line,
            "loss": loss_line,
            "epsilon": eps_line,
            "eval_scatter": eval_scatter,
        },
        "data": data,
    }

    return plot_elements


def update_plots(
    plot_elements,
    episode,
    reward,
    length,
    loss,
    epsilon,
    is_eval=False,
    eval_reward=None,
):
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
    data = plot_elements["data"]
    lines = plot_elements["lines"]
    axes = plot_elements["axes"]
    fig = plot_elements["fig"]

    # Update data lists for regular training episodes
    if not is_eval:
        data["episodes"].append(episode)
        data["rewards"].append(reward)
        data["lengths"].append(length)
        if loss is not None:
            data["losses"].append(loss)
        data["epsilons"].append(epsilon)

        # Update line plots
        lines["reward"].set_data(data["episodes"], data["rewards"])
        lines["length"].set_data(data["episodes"], data["lengths"])

        if data["losses"]:
            lines["loss"].set_data(
                data["episodes"][: len(data["losses"])], data["losses"]
            )

        lines["epsilon"].set_data(data["episodes"], data["epsilons"])

    # Update evaluation scatter plot
    else:
        data["eval_episodes"].append(episode)
        data["eval_rewards"].append(eval_reward)
        lines["eval_scatter"].set_offsets(
            np.column_stack((data["eval_episodes"], data["eval_rewards"]))
        )

    # Rescale axes if needed
    for i, ax in enumerate(axes):
        ax.relim()
        ax.autoscale_view()

    # Draw and flush events
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)


#  plot_elements = setup_plotting()


def optimized_main():
    """
    Optimized training loop for DQN agent with performance enhancements
    especially focused on speeding up the one-hot encoding process.
    Now with curriculum learning to improve training.
    """
    # Training parameters
    num_episodes = 1000  # More episodes for better results
    save_every = 100
    eval_every = 50
    checkpoint_dir = "./checkpoints"

    # Curriculum learning parameters
    initial_difficulty = 0.1  # Start with very easy mazes
    final_difficulty = 0.8  # End with challenging mazes
    initial_start_distance = 0.2  # Start close to goal (20% of max distance)
    final_start_distance = 1.0  # End with starting at the beginning

    # Enable CUDA benchmarking to optimize CUDA operations
    torch.backends.cudnn.benchmark = True

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    GAMMA = 0.99

    # Create checkpoints directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize optimized environment and agent
    env = CurriculumGridWorldEnv(max_steps=100)  # Using our new curriculum environment

    # Choose the appropriate agent implementation
    # Uncomment the one you want to use:
    agent = init_optimized_agent(device)  # Most optimized
    # agent = init_fast_agent(device)      # Fast with simplified caching
    # agent = init_agent(device)           # Original implementation

    # Metrics tracking
    all_rewards = []
    episode_lengths = []
    all_losses = []

    # Setup plotting if desired
    plot_elements = setup_plotting()

    # Warmup phase with vectorized operations
    print("Warming up replay buffer...")
    state = env.reset(
        difficulty=initial_difficulty, start_distance=initial_start_distance
    )
    for _ in range(min(1000, agent.replay_buffer.capacity // 10)):
        action = random.randint(0, 3)  # Random action
        next_state, reward, done = env.step(action, GAMMA)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset(
                difficulty=initial_difficulty, start_distance=initial_start_distance
            )

    print(f"Starting training for {num_episodes} episodes")

    # Main training loop
    for episode in range(num_episodes):
        # Calculate current curriculum parameters based on training progress
        progress = episode / num_episodes
        current_difficulty = initial_difficulty + progress * (
            final_difficulty - initial_difficulty
        )
        current_start_distance = initial_start_distance + progress * (
            final_start_distance - initial_start_distance
        )

        # Reset environment with current curriculum parameters
        state = env.reset(
            difficulty=current_difficulty, start_distance=current_start_distance
        )
        episode_reward = 0
        episode_loss = 0
        steps = 0
        losses = []

        # Episode loop - collect full episode before updating
        done = False
        transitions = []

        while not done:
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, done = env.step(action, GAMMA)

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            transitions.append((state, action, reward, next_state, done))

            # Update state and tracking variables
            state = next_state
            episode_reward += reward
            steps += 1

        # Batch learning after episode completion
        for _ in range(min(4, len(transitions))):
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        # Print cache statistics occasionally
        if hasattr(agent, "cache_stats") and (episode + 1) % 10 == 0:
            stats = agent.cache_stats()
            hit_rate = stats["hit_rate"] * 100
            print(
                f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, {hit_rate:.2f}% hit rate"
            )

            # Limit cache size if needed
            if hasattr(agent, "limit_cache_size"):
                agent.limit_cache_size(max_size=10000)

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
            print(
                f"Episode {episode+1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.1f} | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Difficulty: {current_difficulty:.2f} | "
                f"Start Distance: {current_start_distance:.2f}"
            )

            # Update the plots with latest data
            update_plots(
                plot_elements,
                episode + 1,
                episode_reward,
                steps,
                episode_loss,
                agent.epsilon,
            )

        # Save checkpoint
        if (episode + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_{episode+1}.pth"
            )
            torch.save(
                {"model_state_dict": agent.policy_net.state_dict()}, checkpoint_path
            )
            print(f"Checkpoint saved at episode {episode+1}")

            # Clear cache to avoid memory issues with long training runs
            if hasattr(agent, "clear_cache"):
                agent.clear_cache()

        # Evaluation phase
        if (episode + 1) % eval_every == 0:
            eval_rewards = []
            agent.zero_epsilon()  # No exploration during evaluation

            # For evaluation, use the final difficulty but vary start distances
            for i in range(5):  # Run 5 evaluation episodes
                # Use different start distances for evaluation
                eval_start_distance = 0.2 * (i + 1)  # 0.2, 0.4, 0.6, 0.8, 1.0
                eval_state = env.reset(
                    difficulty=final_difficulty, start_distance=eval_start_distance
                )
                eval_reward = 0
                eval_done = False

                while not eval_done:
                    eval_action = agent.select_action(eval_state)
                    eval_state, reward, eval_done = env.step(eval_action, GAMMA)
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
                eval_reward=avg_eval_reward,
            )

    print("Training completed!")

    # Print final cache statistics
    if hasattr(agent, "cache_stats"):
        stats = agent.cache_stats()
        hit_rate = stats["hit_rate"] * 100
        print(
            f"Final cache stats: {stats['hits']} hits, {stats['misses']} misses, {hit_rate:.2f}% hit rate"
        )


# You can run this with profiling to verify the improvements
if __name__ == "__main__":
    import cProfile

    cProfile.run("optimized_main()", "optimized_stats")

    import pstats

    p = pstats.Stats("optimized_stats")
    p.sort_stats("cumulative").print_stats(20)
