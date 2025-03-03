import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import random as nprand
from random import random as rand
from math import exp

GRID_SIZE = 16
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import random as nprand
from random import random as rand
from math import exp

GRID_SIZE = 16
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


def fancy_generate_maze_vectorized(
    buffer, device, nchannels=2, size=16, difficulty=0.5, batch_size=1
):
    """
    Vectorized version of fancy_generate_maze that handles batch processing.

    Args:
        buffer: Optional pre-allocated tensor to write into, shape [batch_size, nchannels, size, size]
        device: The torch device to place the tensor on
        nchannels (int): Number of channels for one-hot encoding
        size (int): Size of the maze (size x size grid)
        difficulty (float or tensor): Value between 0.0 and 1.0 controlling maze complexity
                                      Can be a single float or a tensor of shape [batch_size]
        batch_size (int): Number of mazes to generate in parallel

    Returns:
        torch.Tensor: Batch of one-hot encoded mazes with shape [batch_size, nchannels, size, size]
    """
    # Convert difficulty to tensor if it's a scalar
    if not isinstance(difficulty, torch.Tensor):
        difficulty = torch.full((batch_size,), difficulty, dtype=DTYPE, device=device)

    # Initialize the output tensor
    one_hot = buffer
    if one_hot is None:
        one_hot = torch.zeros(
            (batch_size, nchannels, size, size), dtype=DTYPE, device=device
        )
    else:
        one_hot.zero_()

    # Start with all mazes filled with walls
    one_hot[:, 0, :, :] = 1.0

    # Create grid pattern - clear cells at even coordinates using meshgrid
    # Create indices for even coordinates
    even_indices = torch.arange(0, size, 2, device=device)

    # Use meshgrid to create coordinate matrices
    i_coords, j_coords = torch.meshgrid(even_indices, even_indices, indexing="ij")

    # Clear all even cells across all batches in one operation
    one_hot[:, 0, i_coords, j_coords] = 0

    # For each batch element, process the guaranteed path and connections
    # This part is harder to fully vectorize due to the path creation logic
    # We'll use a batched loop approach here

    # Calculate waypoints for each maze in batch
    num_waypoints = 3 + (difficulty * 3).int()  # [batch_size]

    # Process each maze in the batch
    for b in range(batch_size):
        # Waypoints start at top-left for all mazes
        waypoints = [(0, 0)]

        # Generate intermediate waypoints for this maze
        for i in range(num_waypoints[b].item()):
            progress = (i + 1) / (num_waypoints[b].item() + 1)

            # Mix randomness with progression based on difficulty
            random_factor = difficulty[b].item() * 0.4
            progress_factor = 1.0 - random_factor

            # Calculate waypoint position
            wp_y = int(
                (
                    torch.rand(1, device=device).item() * random_factor
                    + progress * progress_factor
                )
                * (size - 1)
            )
            wp_x = int(
                (
                    torch.rand(1, device=device).item() * random_factor
                    + progress * progress_factor
                )
                * (size - 1)
            )

            # Ensure waypoints are on valid path cells (even coordinates)
            wp_y = (wp_y // 2) * 2
            wp_x = (wp_x // 2) * 2

            waypoints.append((wp_y, wp_x))

        # Add the end point
        waypoints.append((size - 1, size - 1))

        # Connect the waypoints with straight paths
        for i in range(len(waypoints) - 1):
            y1, x1 = waypoints[i]
            y2, x2 = waypoints[i + 1]

            # Connect horizontally first, then vertically
            x_start, x_end = min(x1, x2), max(x1, x2)
            for x in range(x_start, x_end + 1):
                one_hot[b, 0, y1, x] = 0

            # Create vertical path
            y_start, y_end = min(y1, y2), max(y1, y2)
            for y in range(y_start, y_end + 1):
                one_hot[b, 0, y, x_end] = 0

        # Add connections based on difficulty
        connect_chance = 0.7 - 0.5 * difficulty[b].item()

        # For each path cell, randomly connect to neighboring cells
        for i in range(0, size - 2, 2):
            for j in range(0, size - 2, 2):
                # Each cell gets one random connection (north or east)
                if torch.rand(1, device=device).item() < 0.5 and i > 0:  # Connect north
                    one_hot[b, 0, i - 1, j] = 0
                else:  # Connect east
                    one_hot[b, 0, i, j + 1] = 0

                # Add extra connections with probability based on difficulty
                if torch.rand(1, device=device).item() < connect_chance:
                    # Pick a random direction
                    direction = int(torch.rand(1, device=device).item() * 4)

                    if direction == 0 and i > 0:  # North
                        one_hot[b, 0, i - 1, j] = 0
                    elif direction == 1 and j < size - 1:  # East
                        one_hot[b, 0, i, j + 1] = 0
                    elif direction == 2 and i < size - 1:  # South
                        one_hot[b, 0, i + 1, j] = 0
                    elif direction == 3 and j > 0:  # West
                        one_hot[b, 0, i, j - 1] = 0

    # Clear entrance and exit for all mazes (vectorized)
    one_hot[:, 0, 0, 0] = 0  # Entrance (top-left)
    one_hot[:, 0, size - 1, size - 1] = 0  # Exit (bottom-right)

    # Clean up paths to entrance/exit for all mazes (vectorized)
    if size > 1:
        one_hot[:, 0, 0, 1] = 0  # Clear path right of entrance
        one_hot[:, 0, 1, 0] = 0  # Clear path below entrance
        one_hot[:, 0, size - 2, size - 1] = 0  # Clear path to exit
        one_hot[:, 0, size - 1, size - 2] = 0  # Clear path to exit

    return one_hot


def f_vectorized(d, bsz, device):
    """
    Vectorized version of the f function to compute starting positions

    Args:
        d: Float or tensor of shape [batch_size] representing distance parameter
        bsz: Batch size
        device: Torch device

    Returns:
        Tensor of positions with shape [batch_size]
    """
    # Convert d to tensor if it's a scalar
    if not isinstance(d, torch.Tensor):
        d = torch.full((bsz,), d, dtype=DTYPE, device=device)

    # Create result tensor
    result = torch.zeros(bsz, dtype=torch.int16, device=device)

    # Process each element
    # Apply mask for d >= 1 (return 0)
    mask_geq_1 = d >= 1
    result[mask_geq_1] = 0

    # For d < 1, apply the transformation
    mask_lt_1 = ~mask_geq_1
    if mask_lt_1.sum() > 0:
        center = (1 - d[mask_lt_1]) * GRID_SIZE
        std_dev = torch.max(
            torch.tensor(0.5, device=device), (1 - d[mask_lt_1]) * GRID_SIZE * 0.2
        )

        # Generate normal distribution values
        normal_values = torch.normal(center, std_dev)

        # Clamp values to be within grid bounds
        clamped_values = torch.clamp(normal_values, 1, GRID_SIZE - 1)

        # Round to integers
        result[mask_lt_1] = clamped_values.round().to(torch.int16)

    return result


class GridWorldEnv:
    def __init__(self, device, max_steps, batch_size):
        """
        Batched version of GridWorldEnv that handles multiple environments in parallel

        Args:
            device: The torch device to place tensors on
            max_steps: Maximum steps per episode
            batch_size: Number of parallel environments
        """
        self.num_channels = 2  # wall, agent
        self.max_steps = max_steps
        self.grid_size = GRID_SIZE
        self.device = device
        self.batch_size = batch_size

        self.move_map = {
            0: (-1, 0),  # up
            1: (1, 0),  # down
            2: (0, -1),  # left
            3: (0, 1),  # right
        }

        # Batched tensors for all environments
        self.grids = torch.zeros(
            (batch_size, self.num_channels, GRID_SIZE, GRID_SIZE),
            dtype=DTYPE,
            device=self.device,
        )
        self.visit_counts = torch.zeros(
            (batch_size, GRID_SIZE, GRID_SIZE), dtype=torch.int16, device=self.device
        )

        # Track agent positions for all environments
        self.agent_positions = torch.zeros(
            (batch_size, 2), dtype=torch.int16, device=self.device
        )
        self.goal_positions = torch.full(
            (batch_size, 2), GRID_SIZE - 1, dtype=torch.int16, device=self.device
        )
        self.steps_count = torch.zeros(
            batch_size, dtype=torch.int16, device=self.device
        )

        # Track active environments (not done yet)
        self.active_envs = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        self.reset()

    def reset(self, maze_difficulty=0.5, dist_to_end=0.0):
        """
        Vectorized reset for all environments

        Args:
            maze_difficulty: Float or tensor of maze difficulties (0.0-1.0)
            dist_to_end: Float or tensor of distances to end (0.0-1.0)

        Returns:
            Tensor of shape [batch_size, channels, grid_size, grid_size]
        """
        # Generate mazes for all environments in one batch operation
        fancy_generate_maze_vectorized(
            self.grids,
            device=self.device,
            nchannels=self.num_channels,
            size=GRID_SIZE,
            difficulty=maze_difficulty,
            batch_size=self.batch_size,
        )

        # Clear agent channel before placing agents
        self.grids[:, 1].zero_()

        # Compute starting positions for all agents in batch
        self.agent_positions[:, 0] = f_vectorized(
            dist_to_end, self.batch_size, self.device
        )
        self.agent_positions[:, 1] = f_vectorized(
            dist_to_end, self.batch_size, self.device
        )

        # Place agents using advanced indexing
        batch_indices = torch.arange(self.batch_size, device=self.device)
        self.grids[
            batch_indices, 1, self.agent_positions[:, 0], self.agent_positions[:, 1]
        ] = 1

        # Reset step counters and visit counts
        self.steps_count.zero_()
        self.visit_counts.zero_()

        # All environments active at start
        self.active_envs.fill_(True)

        return self.grids.clone()


def generate_maze(buffer, device, nchannels=2, size=16, difficulty=0.5):
    """
    Generate a maze directly as a one-hot encoded tensor with fully vectorized operations.

    Args:
        device: The torch device to place the tensor on
        nchannels (int): Number of channels for one-hot encoding
        size (int): Size of the maze (size x size grid)
        difficulty (float): Value between 0.0 and 1.0 controlling maze difficulty

    Returns:
        torch.Tensor: One-hot encoded maze of shape [nchannels, size, size]
    """
    one_hot = buffer
    if one_hot is None:
        one_hot = torch.zeros((nchannels, size, size), dtype=DTYPE, device=device)
    wall_cols = torch.arange(1, size, 2, device=device)
    num_walls = len(wall_cols)
    one_hot[0, :, wall_cols] = 1
    difficulty = max(0.0, min(1.0, difficulty))
    hole_percentage = 0.75 - 0.65 * difficulty
    holes_per_wall = max(1, int(hole_percentage * size))
    repeated_wall_cols = wall_cols.repeat_interleave(holes_per_wall)

    all_positions = torch.randint(0, size, (size * num_walls,), device=device)
    hole_rows = all_positions[: num_walls * holes_per_wall]
    one_hot[0, hole_rows, repeated_wall_cols] = 0
    one_hot[0, -1, -1] = 0
    if size % 2 == 0:
        one_hot[0, -1, -2] = 0
    return one_hot


class BatchedDQNAgent:
    def __init__(
        self,
        device,
        lr,
        gamma,
        buffer_capacity,
        batch_size,
        update_target_every,
    ):
        """
        DQN Agent that can handle batched environments

        Args:
            device: Torch device
            lr: Learning rate
            gamma: Discount factor
            buffer_capacity: Size of replay buffer
            batch_size:
            update_target_every: Frequency of target network updates
        """
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps_done = 0
        self.env_batch_size = env_batch_size

        # Input channels and actions
        self.input_channels = 2  # wall, agent
        self.num_actions = 4  # up, down, left, right

        # Initialize networks
        self.policy_net = DQN(self.input_channels, self.num_actions).to(self.device)
        self.target_net = DQN(self.input_channels, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Experience replay buffer
        state_shape = (self.input_channels, GRID_SIZE, GRID_SIZE)
        self.replay_buffer = TensorReplayBuffer(
            buffer_capacity, state_shape, self.device
        )

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998

    def select_actions(self, states):
        """
        Select actions for all environments using epsilon-greedy policy

        Args:
            states: Batch of states [batch_size, channels, grid_size, grid_size]

        Returns:
            Tensor of action indices [batch_size]
        """
        # Create a tensor to hold all actions
        actions = torch.zeros(self.env_batch_size, dtype=torch.long, device=self.device)

        # Generate random numbers for epsilon-greedy decisions
        rand_values = torch.rand(self.env_batch_size, device=self.device)
        explore_mask = rand_values < self.epsilon
        exploit_mask = ~explore_mask

        # For exploration actions, select random actions
        num_explore = explore_mask.sum().item()
        if num_explore > 0:
            actions[explore_mask] = torch.randint(
                0, self.num_actions, (num_explore,), device=self.device
            )

        # For exploitation actions, use the policy network
        num_exploit = exploit_mask.sum().item()
        if num_exploit > 0:
            with torch.no_grad():
                q_values = self.policy_net(states[exploit_mask])
                actions[exploit_mask] = q_values.max(1)[1]

        return actions

    def update(self):
        """
        Update network weights using experiences from the buffer
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Current Q values
        current_q_values = (
            self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        )

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (~done_batch)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def push_transitions(self, states, actions, rewards, next_states, dones):
        """
        Push multiple transitions to the replay buffer

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        self.replay_buffer.push_batch(states, actions, rewards, next_states, dones)

    def set_epsilon(self, episode, total_episodes):
        """
        Set exploration rate based on training progress
        """
        self.epsilon = self.epsilon_min + 0.8 * (1.0 - self.epsilon_min) * exp(
            -3.0 * episode / total_episodes
        )


# -----------------------
# NEURAL NETWORK
# -----------------------


class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()

        # Calculate the feature size after convolutions and pooling
        feature_size = ((GRID_SIZE // 4) ** 2) * 4 * input_channels

        # CNN layers
        self.cnn_layers = nn.Sequential(
            # First block: 4 → 8 channels, gs → gs/2
            nn.Conv2d(input_channels, 2 * input_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * input_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second block: 8 → 16 channels, gs/2 → gs/4
            nn.Conv2d(2 * input_channels, 4 * input_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * input_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.mlp(self.cnn_layers(x))


# -----------------------
# REPLAY BUFFER
# -----------------------


class TensorReplayBuffer:
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Pre-allocate tensors for all storage
        self.states = torch.zeros((capacity, *state_shape), dtype=DTYPE, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=DTYPE, device=device)
        self.next_states = torch.zeros(
            (capacity, *state_shape), dtype=DTYPE, device=device
        )
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        # Convert inputs to tensors if needed
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], device=self.device, dtype=torch.long)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=DTYPE, device=self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor([done], dtype=torch.bool, device=self.device)

        # Store directly in pre-allocated tensors
        self.states[self.position] = state
        self.actions[self.position, 0] = action
        self.rewards[self.position, 0] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position, 0] = done

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones):
        """
        Store multiple transitions in the buffer efficiently.

        Args:
            states: Batch of states [batch_size, channels, grid_size, grid_size]
            actions: Batch of actions [batch_size]
            rewards: Batch of rewards [batch_size]
            next_states: Batch of next states [batch_size, channels, grid_size, grid_size]
            dones: Batch of done flags [batch_size]
        """
        batch_size = len(states)

        # Ensure all inputs are tensors
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=DTYPE, device=self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Handle wrap-around at capacity boundaries
        positions = (
            torch.arange(self.position, self.position + batch_size, device=self.device)
            % self.capacity
        )

        # Store batched data with advanced indexing
        self.states[positions] = states
        self.actions[positions, 0] = actions
        self.rewards[positions, 0] = rewards
        self.next_states[positions] = next_states
        self.dones[positions, 0] = dones

        # Update position and size
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer efficiently."""
        batch_size = min(batch_size, self.size)
        batch_indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # Index directly from pre-allocated tensors
        return (
            self.states[batch_indices],
            self.actions[batch_indices].squeeze(1),  # Remove extra dimension
            self.rewards[batch_indices].squeeze(1),  # Remove extra dimension
            self.next_states[batch_indices],
            self.dones[batch_indices].squeeze(1),  # Remove extra dimension
        )

    def __len__(self):
        return self.size


if __name__ == "__main__":
    m = fancy_generate_maze_vectorized(None, torch.device("cpu"), batch_size=1)
    import matplotlib.pyplot as plt

    plt.imshow(m[0, 0, :, :].numpy(), cmap="gray")
    plt.show()
