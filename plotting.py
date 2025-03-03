import matplotlib.pyplot as plt
import numpy as np

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

