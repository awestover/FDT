from deps import *
import random
import torch
import os

PLOTTING = False
if PLOTTING:
    from plotting import setup_plotting, update_plots

def main():
    if PLOTTING:
        plot_elements = setup_plotting()

    """
    Optimized training loop for DQN agent with performance enhancements
    especially focused on speeding up the one-hot encoding process.
    Now with curriculum learning to improve training.
    """
    # Training parameters
    num_episodes = 5000  # More episodes for better results
    save_every = num_episodes//10
    eval_every = num_episodes//100
    checkpoint_dir = "./checkpoints"

    # Curriculum learning parameters
    initial_difficulty = 0.5  # Start with very easy mazes
    final_difficulty = 0.8  # End with challenging mazes
    init_dist_to_end = 1.0  # Start with short distance to end
    final_dist_to_end = 1.0 # End with long distance to end

    # Enable CUDA benchmarking to optimize CUDA operations
    torch.backends.cudnn.benchmark = True

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoints directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize optimized environment and agent
    env = GridWorldEnv(device, max_steps=100)  # Using our new curriculum environment

    # TODO: increase batch size to as big as the GPU can handle
    agent = DQNAgent(device, lr=1e-4, gamma=0.99, buffer_capacity=50000, batch_size=1024, update_target_every=500)

    # Tracking variables
    episode_lengths = []
    all_rewards = []
    all_losses = []

    # Warmup phase with vectorized operations
    print("Warming up replay buffer...")
    state = env.reset(initial_difficulty, init_dist_to_end)
    for _ in range(min(1000, agent.replay_buffer.capacity // 10)):
        action = random.randint(0, 3)  # Random action
        next_state, reward, done = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset(initial_difficulty, init_dist_to_end)

    print(f"Starting training for {num_episodes} episodes")

    # Main training loop
    for episode in range(num_episodes):
        agent.set_epsilon(episode, num_episodes)

        # Calculate current curriculum parameters based on training progress
        progress = episode / num_episodes
        cur_difficulty = initial_difficulty + progress * (final_difficulty - initial_difficulty)
        cur_dist_to_end = init_dist_to_end + progress * (final_dist_to_end - init_dist_to_end)

        # Reset environment with current curriculum parameters
        state = env.reset(cur_difficulty, cur_dist_to_end)
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
            next_state, reward, done = env.step(action)

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            transitions.append((state, action, reward, next_state, done))

            # Update state and tracking variables
            state = next_state
            episode_reward += reward
            steps += 1

        # Batch learning after episode completion
        target_upated = False
        for _ in transitions:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            target_updated = target_upated or (agent.steps_done % agent.update_target_every == 0)

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
                f"Reward: {avg_reward:.2f} | "
                f"Length: {avg_length:.1f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Difficulty: {cur_difficulty:.2f} | "
                f"Dist To End: {cur_dist_to_end:.2f}"
            )
            if PLOTTING:
                update_plots(
                    plot_elements,
                    episode,
                    episode_reward,
                    steps,
                    episode_loss if losses else None,
                    agent.epsilon,
                    target_upated=target_upated
                )

        #  Save checkpoint
        if (episode + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode+1}.pth")
            torch.save({"model_state_dict": agent.policy_net.state_dict()}, checkpoint_path)
            print(f"Checkpoint saved at episode {episode+1}")

            if PLOTTING:
                plot_path = os.path.join(checkpoint_dir, f"training_plot_{episode+1}.png")
                plot_elements["fig"].savefig(plot_path, dpi=300, bbox_inches='tight')

        # Evaluation phase
        if (episode + 1) % eval_every == 0:
            eval_rewards = []
            eval_lengths = []
            agent.epsilon = agent.epsilon_min # don't explore much in eval

            # For evaluation, use the final difficulty but vary start distances
            for i in range(5):  # Run 5 evaluation episodes
                # Use different start distances for evaluation
                eval_start_distance = 0.2 * (i + 1)  # 0.2, 0.4, 0.6, 0.8, 1.0
                eval_state = env.reset(final_difficulty, eval_start_distance)
                eval_reward = 0
                eval_done = False
                eval_steps = 0

                while not eval_done:
                    eval_action = agent.select_action(eval_state)
                    eval_state, reward, eval_done = env.step(eval_action)
                    eval_reward += reward
                    eval_steps += 1

                eval_rewards.append(eval_reward)
                eval_lengths.append(eval_steps)

            avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
            avg_eval_length = sum(eval_lengths) / len(eval_lengths)
            print(f"Evaluation: Avg Reward = {avg_eval_reward:.2f}")
            print(f"Evaluation: Avg Length = {avg_eval_length:.1f}")
            # Update evaluation plots
            if PLOTTING:
                update_plots(
                    plot_elements,
                    episode,
                    None,
                    None,
                    None,
                    None,
                    is_eval=True,
                    eval_reward=avg_eval_reward
                )

    print("Training completed!")
    if PLOTTING:
        plt.ioff()
        plt.close()

# You can run this with profiling to verify the improvements
if __name__ == "__main__":
    import cProfile
    cProfile.run("main()", "stats")
    import pstats
    p = pstats.Stats("stats")
    p.sort_stats("cumulative").print_stats(10)

