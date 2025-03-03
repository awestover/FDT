from deps import *
import random
import torch
import os

PROFILING_ONLY = True
PLOTTING = False
if PLOTTING:
    from plotting import setup_plotting, update_plots

# TODO:
# choose BUFFER_CAPACITY to max out GPU memory
# choose NUM_EPISODES to max out time


def main():
    ON_GPU = torch.cuda.is_available()
    device = torch.device("cuda" if ON_GPU else "cpu")
    print(f"Using device: {device}")
    MAX_STEPS = 100
    BSZ = 1024 if ON_GPU else 32
    UPDATE_TARGET_EVERY = MAX_STEPS * BSZ // 10
    BUFFER_CAPACITY = 100_000
    NUM_EPISODES = 100_000 if not PROFILING_ONLY else 50
    SAVE_EVERY = NUM_EPISODES // 10 if not PROFILING_ONLY else 5000
    EVAL_EVERY = NUM_EPISODES // 100 if not PROFILING_ONLY else 5000

    if PLOTTING:
        plot_elements = setup_plotting()

    # Training parameters
    checkpoint_dir = "./checkpoints"

    # Curriculum learning parameters
    initial_difficulty = 0.25
    final_difficulty = 1.0
    init_dist_to_end = 0.25
    final_dist_to_end = 1.0

    # Enable CUDA benchmarking to optimize CUDA operations
    torch.backends.cudnn.benchmark = True

    # Create checkpoints directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize batched environment and agent
    env = GridWorldEnv(device, max_steps=MAX_STEPS, batch_size=BSZ)

    agent = BatchedDQNAgent(
        device,
        lr=2e-4,
        gamma=0.99,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BSZ,
        update_target_every=UPDATE_TARGET_EVERY,
    )

    # Tracking variables
    episode_lengths = []
    all_rewards = []
    all_losses = []

    # Warmup phase with vectorized operations
    print("Warming up replay buffer...")
    difficulties = torch.ones(BSZ, device=device) * initial_difficulty
    distances = torch.ones(BSZ, device=device) * init_dist_to_end
    states = env.reset(difficulties, distances)

    # Fill buffer with a couple random actions
    for _ in range(MAX_STEPS // 10):
        actions = torch.randint(0, 4, (BSZ,), device=device)
        next_states, rewards, dones = env.step(actions)
        agent.push_transitions(states, actions, rewards, next_states, dones)
        states = next_states

        # Reset states for done environments
        if dones.any():
            # Selectively reset only the done environments
            new_states = env.reset_subset(dones, difficulties, distances)
            # Update only the states that are done
            states = torch.where(
                dones.unsqueeze(1).unsqueeze(2).unsqueeze(3), new_states, states
            )

    print(f"Starting training with {BSZ} parallel environments")
    print(f"Target: {NUM_EPISODES} total episodes")

    # Keep track of steps per episode for each environment
    env_episode_steps = torch.zeros(BSZ, dtype=torch.int64, device=device)
    env_episode_rewards = torch.zeros(BSZ, device=device)
    episodes_completed = 0

    # Main training loop - we'll run this until we complete the target number of episodes
    for episode in range(NUM_EPISODES):
        agent.set_epsilon(episode, NUM_EPISODES)

        # Calculate current curriculum parameters
        progress = episode / NUM_EPISODES
        cur_difficulty = initial_difficulty + progress * (
            final_difficulty - initial_difficulty
        )
        cur_dist_to_end = init_dist_to_end + progress * (
            final_dist_to_end - init_dist_to_end
        )

        # Set up difficulty and distance tensors for all environments
        difficulties = torch.ones(BSZ, device=device) * cur_difficulty
        distances = torch.ones(BSZ, device=device) * cur_dist_to_end

        # Reset all environments with current parameters
        states = env.reset(difficulties, distances)
        env_episode_steps.zero_()
        env_episode_rewards.zero_()

        done_mask = torch.zeros(BSZ, dtype=torch.bool, device=device)
        losses = []

        # Run all environments until all episodes are complete
        while not done_mask.all():
            # Select and execute actions for all environments
            actions = agent.select_actions(states)

            # Only execute actions for environments that are not done
            active_mask = ~done_mask
            next_states, rewards, dones = env.step(actions, active_mask)

            # Update accumulated rewards and steps for active environments
            env_episode_rewards[active_mask] += rewards[active_mask]
            env_episode_steps[active_mask] += 1

            # Store transitions in replay buffer
            agent.push_transitions(
                states[active_mask],
                actions[active_mask],
                rewards[active_mask],
                next_states[active_mask],
                dones[active_mask],
            )

            # Update model
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            # Track newly completed episodes
            new_dones = dones & ~done_mask
            done_mask = done_mask | dones

            # Record completed episode stats
            num_new_done = new_dones.sum().item()
            if num_new_done > 0:
                for i in range(BSZ):
                    if new_dones[i]:
                        episode_lengths.append(env_episode_steps[i].item())
                        all_rewards.append(env_episode_rewards[i].item())
                        episodes_completed += 1

            # Update states for environments that are still active
            # We only need to update active environments as done ones will be reset
            states = torch.where(
                active_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3), next_states, states
            )

            # Reset environments that are newly done
            if new_dones.any():
                # Get new states for the newly done environments using reset_subset
                new_env_states = env.reset_subset(new_dones, difficulties, distances)

                # Create a mask for updating only newly done environments
                update_mask = new_dones.unsqueeze(1).unsqueeze(2).unsqueeze(3)

                # Expand new_env_states to match the batch size
                expanded_new_states = torch.zeros_like(states)
                expanded_new_states[new_dones] = new_env_states

                # Update states for newly done environments
                states = torch.where(update_mask, expanded_new_states, states)

        # Calculate loss for this batch of episodes
        if losses:
            batch_loss = sum(losses) / len(losses)
            all_losses.append(batch_loss)

        # Print progress
        if episodes_completed % 10 == 0 or (episodes_completed % BSZ == 0):
            recent_rewards = all_rewards[-BSZ:]
            recent_lengths = episode_lengths[-BSZ:]

            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_length = sum(recent_lengths) / len(recent_lengths)
            avg_loss = sum(all_losses[-10:]) / max(1, len(all_losses[-10:]))

            print(
                f"Episodes {episodes_completed}/{NUM_EPISODES} | "
                f"Reward: {avg_reward:.2f} | "
                f"Length: {avg_length:.1f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Difficulty: {cur_difficulty:.2f}"
            )

            if PLOTTING:
                update_plots(
                    plot_elements,
                    episodes_completed,
                    avg_reward,
                    avg_length,
                    avg_loss,
                    agent.epsilon,
                )

        # Save checkpoint
        if episodes_completed > 0 and episodes_completed % SAVE_EVERY < BSZ:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_{episodes_completed}.pth"
            )
            torch.save(
                {"model_state_dict": agent.policy_net.state_dict()}, checkpoint_path
            )
            print(f"Checkpoint saved at episode {episodes_completed}")

            if PLOTTING:
                plot_path = os.path.join(
                    checkpoint_dir, f"training_plot_{episodes_completed}.png"
                )
                plot_elements["fig"].savefig(plot_path, dpi=300, bbox_inches="tight")

        # Evaluation phase
        if episodes_completed > 0 and episodes_completed % EVAL_EVERY < BSZ:
            print(f"Running evaluation at {episodes_completed} episodes...")
            eval_rewards = []
            eval_lengths = []

            # Save current epsilon and use minimum for evaluation
            original_epsilon = agent.epsilon
            agent.epsilon = agent.epsilon_min

            # Reset all environments with current difficulty but vary start distances
            eval_difficulties = torch.ones(BSZ, device=device) * cur_difficulty
            eval_distances = torch.ones(BSZ, device=device) * cur_dist_to_end

            # Run multiple evaluation batches to get enough samples
            num_eval_batches = 5  # Run 5 batches of BSZ episodes

            for _ in range(num_eval_batches):
                eval_states = env.reset(eval_difficulties, eval_distances)
                eval_episode_rewards = torch.zeros(BSZ, device=device)
                eval_episode_steps = torch.zeros(BSZ, dtype=torch.int64, device=device)
                eval_done_mask = torch.zeros(BSZ, dtype=torch.bool, device=device)

                while not eval_done_mask.all():
                    eval_actions = agent.select_actions(eval_states)

                    # Only execute actions for environments that are not done
                    eval_active_mask = ~eval_done_mask
                    eval_next_states, eval_rewards, eval_dones = env.step(
                        eval_actions, eval_active_mask
                    )

                    # Update accumulators for active environments
                    eval_episode_rewards[eval_active_mask] += eval_rewards[
                        eval_active_mask
                    ]
                    eval_episode_steps[eval_active_mask] += 1

                    # Track newly completed episodes
                    eval_new_dones = eval_dones & ~eval_done_mask
                    eval_done_mask = eval_done_mask | eval_dones

                    # Record completed episode stats
                    for i in range(BSZ):
                        if eval_new_dones[i]:
                            eval_rewards.append(eval_episode_rewards[i].item())
                            eval_lengths.append(eval_episode_steps[i].item())

                    # Update states for environments that are still active
                    eval_states = torch.where(
                        eval_active_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                        eval_next_states,
                        eval_states,
                    )

                    # Reset environments that are newly done
                    if eval_new_dones.any():
                        # Get new states for the newly done environments
                        new_eval_states = env.reset_subset(
                            eval_new_dones, eval_difficulties, eval_distances
                        )

                        # Create a mask for updating only newly done environments
                        update_mask = (
                            eval_new_dones.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        )

                        # Expand new_eval_states to match the batch size
                        expanded_new_states = torch.zeros_like(eval_states)
                        expanded_new_states[eval_new_dones] = new_eval_states

                        # Update states for newly done environments
                        eval_states = torch.where(
                            update_mask, expanded_new_states, eval_states
                        )

            # Restore original epsilon
            agent.epsilon = original_epsilon

            # Calculate evaluation metrics
            avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
            avg_eval_length = sum(eval_lengths) / len(eval_lengths)

            print(f"Evaluation: Avg Reward = {avg_eval_reward:.2f}")
            print(f"Evaluation: Avg Length = {avg_eval_length:.1f}")

            # Update evaluation plots
            if PLOTTING:
                update_plots(
                    plot_elements,
                    episodes_completed,
                    None,
                    None,
                    None,
                    None,
                    is_eval=True,
                    eval_reward=avg_eval_reward,
                )

    print("Training completed!")

    # Final save
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(
        {"model_state_dict": agent.policy_net.state_dict()}, final_checkpoint_path
    )
    print(f"Final model saved at {final_checkpoint_path}")

    if PLOTTING:
        plt.ioff()
        plt.close()


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()", "stats")
    import pstats

    p = pstats.Stats("stats")
    p.sort_stats("cumulative").print_stats(20)
