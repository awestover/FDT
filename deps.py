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

    # This method handles resetting a subset of environments
    # Used when we need to reset just the environments that are done
    def reset_subset(self, indices, maze_difficulty=0.5, dist_to_end=0.0):
        """
        Reset only specific environments indexed by indices

        Args:
            indices: Boolean mask or integer indices of environments to reset
            maze_difficulty: Float or tensor of maze difficulties (0.0-1.0)
            dist_to_end: Float or tensor of distances to end (0.0-1.0)

        Returns:
            Tensor of shape [num_reset, channels, grid_size, grid_size]
        """
        # Convert indices to boolean mask if it's a tensor of indices
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.bool:
            mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            mask[indices] = True
        else:
            mask = indices

        num_to_reset = mask.sum().item()

        # Handle scalar difficulty and distance
        if not isinstance(maze_difficulty, torch.Tensor):
            maze_difficulty = torch.full(
                (num_to_reset,), maze_difficulty, dtype=DTYPE, device=self.device
            )
        elif (
            len(maze_difficulty.shape) > 0
            and maze_difficulty.shape[0] == self.batch_size
        ):
            maze_difficulty = maze_difficulty[mask]

        if not isinstance(dist_to_end, torch.Tensor):
            dist_to_end = torch.full(
                (num_to_reset,), dist_to_end, dtype=DTYPE, device=self.device
            )
        elif len(dist_to_end.shape) > 0 and dist_to_end.shape[0] == self.batch_size:
            dist_to_end = dist_to_end[mask]

        # Generate new mazes just for the specified environments
        new_grids = torch.zeros(
            (num_to_reset, self.num_channels, GRID_SIZE, GRID_SIZE),
            dtype=DTYPE,
            device=self.device,
        )

        fancy_generate_maze_vectorized(
            new_grids,
            device=self.device,
            nchannels=self.num_channels,
            size=GRID_SIZE,
            difficulty=maze_difficulty,
            batch_size=num_to_reset,
        )

        # Clear agent channel before placing agents
        new_grids[:, 1].zero_()

        # Compute starting positions for the reset agents
        new_positions = torch.zeros(
            (num_to_reset, 2), dtype=torch.int16, device=self.device
        )
        new_positions[:, 0] = f_vectorized(dist_to_end, num_to_reset, self.device)
        new_positions[:, 1] = f_vectorized(dist_to_end, num_to_reset, self.device)

        # Place agents in the new grids
        batch_indices = torch.arange(num_to_reset, device=self.device)
        new_grids[batch_indices, 1, new_positions[:, 0], new_positions[:, 1]] = 1

        # Update the main grids and agent positions
        self.grids[mask] = new_grids
        self.agent_positions[mask] = new_positions

        # Reset step counters and visit counts for reset environments
        self.steps_count[mask] = 0
        self.visit_counts[mask].zero_()

        # Mark reset environments as active
        self.active_envs[mask] = True

        return new_grids

    def step(self, actions, active_mask=None):
        """
        Execute actions in all active environments

        Args:
            actions: Tensor of action indices [batch_size]
            active_mask: Boolean mask indicating which environments are active [batch_size]
                         If None, all environments are considered active

        Returns:
            next_states: Updated grid states
            rewards: Rewards for each environment
            dones: Done flags for each environment
        """
        # If no active mask is provided, consider all environments active
        if active_mask is None:
            active_mask = torch.ones(
                self.batch_size, dtype=torch.bool, device=self.device
            )

        # Initialize rewards and dones for all environments
        rewards = torch.zeros(self.batch_size, device=self.device)
        dones = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Only process active environments
        if active_mask.any():
            # Get batch indices for active environments
            batch_indices = torch.arange(self.batch_size, device=self.device)[
                active_mask
            ]

            # Clear agent positions in active environments
            self.grids[active_mask, 1].zero_()

            # Get wall positions for collision checking
            walls = self.grids[active_mask, 0]

            # Process movement for each active environment
            for i, b_idx in enumerate(batch_indices):
                action = actions[b_idx]
                y, x = self.agent_positions[b_idx]

                # Get movement direction
                dy, dx = self.move_map[action.item()]

                # Calculate new position
                new_y = torch.clamp(y + dy, 0, self.grid_size - 1)
                new_x = torch.clamp(x + dx, 0, self.grid_size - 1)

                # Check for wall collision
                if walls[i, new_y, new_x] == 0:  # No wall at new position
                    # Update agent position
                    self.agent_positions[b_idx, 0] = new_y
                    self.agent_positions[b_idx, 1] = new_x

                # Update visit counts
                self.visit_counts[
                    b_idx,
                    self.agent_positions[b_idx, 0],
                    self.agent_positions[b_idx, 1],
                ] += 1

                # Place agent in new position
                self.grids[
                    b_idx,
                    1,
                    self.agent_positions[b_idx, 0],
                    self.agent_positions[b_idx, 1],
                ] = 1

                # Check for goal reached
                if (self.agent_positions[b_idx] == self.goal_positions[b_idx]).all():
                    rewards[b_idx] = 1.0
                    dones[b_idx] = True

                # Check for max steps reached
                self.steps_count[b_idx] += 1
                if self.steps_count[b_idx] >= self.max_steps:
                    dones[b_idx] = True

        return self.grids.clone(), rewards, dones


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
        self.env_batch_size = batch_size

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
