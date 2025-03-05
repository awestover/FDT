from deps import *
import torch
import os

PROFILING_ONLY = False
PLOTTING = False
if PLOTTING:
    from plotting import setup_plotting, update_plots
    import matplotlib.pyplot as plt
ON_GPU = torch.cuda.is_available()
device = torch.device("cuda" if ON_GPU else "cpu")
print(f"Using device: {device}")
MAX_STEPS = 50
BSZ = 1<<12 if ON_GPU else 1
UPDATE_TARGET_EVERY = 256
# TODO:
# choose BUFFER_CAPACITY to max out GPU memory
BUFFER_CAPACITY = 10**6
# choose NUM_EPISODES to max out time
NUM_EPISODES = 10**3 if not PROFILING_ONLY else 50
SAVE_EVERY = NUM_EPISODES // 10 if not PROFILING_ONLY else 5000
EVAL_EVERY = NUM_EPISODES // 100 if not PROFILING_ONLY else 5000
MAZE_CACHE_SIZE = 10**6 
if PROFILING_ONLY:
    MAZE_CACHE_SIZE = 1000
if not ON_GPU:
    MAZE_CACHE_SIZE = 200
PRINT_PROGRESS_EVERY = 1

# Curriculum learning parameters
initial_difficulty = 0.95
final_difficulty = 1.0
init_dist_to_end = 0.25
final_dist_to_end = 1.0

checkpoint_dir = "./checkpoints"

## MAIN
if PLOTTING:
    plot_elements = setup_plotting()
torch.backends.cudnn.benchmark = True
os.makedirs(checkpoint_dir, exist_ok=True)
maze_cache = MazeCache(device, BSZ, MAZE_CACHE_SIZE)
env = GridWorldEnv(device, MAX_STEPS, BSZ, maze_cache)
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
states = env.reset(initial_difficulty, init_dist_to_end)
active_envs = torch.ones(BSZ, dtype=torch.bool, device=device)

# Fill buffer with a couple random actions
for _ in range(MAX_STEPS // 10):
    actions = torch.randint(0, 4, (BSZ,), device=device)
    next_states, rewards, dones = env.step(actions, active_envs)
    agent.push_transitions(
        states, actions, rewards, next_states, dones, active_envs
    )
    states = next_states
    active_envs &= ~dones

print(f"Starting training with {BSZ} parallel environments, for {NUM_EPISODES} total episodes")

def train_loop():
    # Keep track of steps per episode for each environment
    env_episode_steps = torch.zeros(BSZ, dtype=torch.int64, device=device)
    env_episode_rewards = torch.zeros(BSZ, device=device)

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

        # Reset all environments with current parameters
        states = env.reset(cur_difficulty, cur_dist_to_end)
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
                states,
                actions,
                rewards,
                next_states,
                dones,
                active_mask,
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

            # Update states for environments that are still active
            # We only need to update active environments as done ones will be reset
            states = torch.where(
                active_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3), next_states, states
            )

        # Calculate loss for this batch of episodes
        if losses:
            batch_loss = sum(losses) / len(losses)
            all_losses.append(batch_loss)

        # Print progress
        if (episode + 1) % PRINT_PROGRESS_EVERY == 0:
            recent_rewards = all_rewards[-BSZ:]
            recent_lengths = episode_lengths[-BSZ:]

            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_length = sum(recent_lengths) / len(recent_lengths)
            avg_loss = sum(all_losses[-10:]) / max(1, len(all_losses[-10:]))

            print(
                f"Episodes {episode}/{NUM_EPISODES} | "
                f"Reward: {avg_reward:.2f} | "
                f"Length: {avg_length:.1f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Difficulty: {cur_difficulty:.2f}"
            )

            if PLOTTING:
                update_plots(
                    plot_elements,
                    episode,
                    avg_reward,
                    avg_length,
                    avg_loss,
                    agent.epsilon,
                )

        # Save checkpoint
        if (episode + 1) % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
            torch.save(
                {"model_state_dict": agent.policy_net.state_dict()}, checkpoint_path
            )
            print(f"Checkpoint saved at episode {episode}")

            if PLOTTING:
                plot_path = os.path.join(checkpoint_dir, f"training_plot_{episode}.png")
                plot_elements["fig"].savefig(plot_path, dpi=300, bbox_inches="tight")

        # Evaluation phase
        # Evaluation phase
        if (episode + 1) % EVAL_EVERY == 0:
            print(f"Running evaluation at {episode} episodes...")
            eval_rewards = []
            eval_lengths = []

            # Save current epsilon and use minimum for evaluation
            original_epsilon = agent.epsilon
            agent.epsilon = agent.epsilon_min

            # Run multiple evaluation batches to get enough samples
            num_eval_batches = 5  # Run 5 batches of BSZ episodes

            for _ in range(num_eval_batches):
                # Reset all environments with current difficulty
                eval_states = env.reset(cur_difficulty, cur_dist_to_end)

                eval_episode_rewards = torch.zeros(BSZ, device=device)
                eval_episode_steps = torch.zeros(BSZ, dtype=torch.int64, device=device)
                eval_done_mask = torch.zeros(BSZ, dtype=torch.bool, device=device)

                # Run until all environments are done or max steps reached
                max_eval_steps = MAX_STEPS
                for _ in range(max_eval_steps):
                    # Break if all environments are done
                    if eval_done_mask.all():
                        break

                    eval_actions = agent.select_actions(eval_states)

                    # Only execute actions for environments that are not done
                    eval_active_mask = ~eval_done_mask
                    eval_next_states, eval_rewards_step, eval_dones = env.step(
                        eval_actions, eval_active_mask
                    )

                    # Update accumulators for active environments
                    eval_episode_rewards[eval_active_mask] += eval_rewards_step[
                        eval_active_mask
                    ]
                    eval_episode_steps[eval_active_mask] += 1

                    # Update done mask - environments stay done once they're done
                    eval_done_mask = eval_done_mask | eval_dones

                    # Update states only for environments that are still active
                    eval_states = torch.where(
                        eval_active_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                        eval_next_states,
                        eval_states,
                    )

                # After loop completion, record all environment stats
                for i in range(BSZ):
                    eval_rewards.append(eval_episode_rewards[i].item())
                    eval_lengths.append(eval_episode_steps[i].item())

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
                    episode,
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


import cProfile

cProfile.run("train_loop()", "stats")
import pstats

p = pstats.Stats("stats")
p.sort_stats("cumulative").print_stats(25)

