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

def fancy_generate_maze_vectorized(buffer=None, device=None, nchannels=2, size=16, difficulty=0.5, batch_size=1):
    """
    Fully vectorized maze generator that ensures a path from start to end.
    
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
    import torch
    
    # Handle device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert difficulty to tensor if it's a scalar
    if not isinstance(difficulty, torch.Tensor):
        difficulty = torch.full((batch_size,), difficulty, device=device)
    
    # Initialize the output tensor
    if buffer is None:
        one_hot = torch.ones((batch_size, nchannels, size, size), device=device)
    else:
        one_hot = buffer
        one_hot.fill_(1.0)
    
    # Start with walls everywhere (channel 0 = walls)
    one_hot[:, 0, :, :] = 1.0
    
    # Create grid pattern - make all even coordinates paths
    even_indices = torch.arange(0, size, 2, device=device)
    i_coords, j_coords = torch.meshgrid(even_indices, even_indices, indexing="ij")
    
    # Set all even cells to be paths (0 = path, 1 = wall in channel 0)
    one_hot[:, 0, i_coords, j_coords] = 0.0
    
    # Number of waypoints based on difficulty (vectorized)
    num_waypoints = 3 + (difficulty * 3).int()
    max_waypoints = num_waypoints.max().item()
    
    # Generate waypoint coordinates for all mazes in batch
    # Shape: [batch_size, max_waypoints+2, 2]
    # +2 for start and end points
    waypoints = torch.zeros((batch_size, max_waypoints+2, 2), device=device)
    
    # All mazes start at top-left (0,0)
    waypoints[:, 0, :] = 0
    
    # All mazes end at bottom-right (size-1, size-1)
    waypoints[:, -1, 0] = size - 1
    waypoints[:, -1, 1] = size - 1
    
    # Generate intermediate waypoints
    for i in range(max_waypoints):
        # Create a mask for valid batch items (those that need this waypoint)
        mask = (i < num_waypoints).float().unsqueeze(-1)
        
        # Calculate progress along path
        progress = torch.tensor([(i + 1) / (n.item() + 1) for n in num_waypoints], device=device).unsqueeze(-1)
        
        # Mix randomness with progression based on difficulty
        random_factor = difficulty.unsqueeze(-1) * 0.4
        progress_factor = 1.0 - random_factor
        
        # Random positions scaled by maze size
        rand_pos = torch.rand((batch_size, 2), device=device)
        
        # Calculate waypoint positions
        wp_pos = (rand_pos * random_factor + progress * progress_factor) * (size - 1)
        
        # Round to even coordinates for valid paths
        wp_pos = (wp_pos // 2) * 2
        
        # Apply mask and store waypoints
        waypoints[:, i+1, :] = wp_pos * mask
    
    # Create paths between waypoints for all mazes in batch
    for i in range(max_waypoints + 1):
        # Only process waypoints that exist for each batch item
        if i == 0:  # First segment (start to first waypoint)
            current_mask = torch.ones(batch_size, device=device)
        else:
            current_mask = (i < num_waypoints+1).float()
        
        # Get current and next waypoints
        curr_points = waypoints[:, i, :]
        next_points = waypoints[:, i+1, :]
        
        # Connect horizontally first, then vertically
        # For each batch, we need to create paths between waypoints
        
        # This part is trickier to fully vectorize due to variable path lengths
        # Let's implement it efficiently by broadcasting
        
        # Horizontal paths
        x_starts = curr_points[:, 1].int()
        x_ends = next_points[:, 1].int()
        y_starts = curr_points[:, 0].int()
        
        # Determine direction (left to right or right to left)
        x_steps = (x_ends >= x_starts).float() * 2 - 1  # 1 if going right, -1 if going left
        
        # Maximum path length needed
        max_x_dist = (x_ends - x_starts).abs().max().int().item()
        
        # Create horizontal paths
        for step in range(max_x_dist + 1):
            # Calculate x position at each step for each batch
            x_offset = torch.clamp(step * x_steps, torch.zeros_like(x_steps), (x_ends - x_starts).abs())
            x_pos = (x_starts + x_offset * ((x_ends >= x_starts).float() * 2 - 1)).long()
            
            # Apply mask for valid positions
            valid_steps = (step <= (x_ends - x_starts).abs()) & current_mask.bool()
            
            # Carve path at these positions
            for b in range(batch_size):
                if valid_steps[b]:
                    one_hot[b, 0, y_starts[b], x_pos[b]] = 0
        
        # Vertical paths (from where horizontal paths ended to the next waypoint)
        y_ends = next_points[:, 0].int()
        
        # Determine direction (top to bottom or bottom to top)
        y_steps = (y_ends >= y_starts).float() * 2 - 1  # 1 if going down, -1 if going up
        
        # Maximum path length needed
        max_y_dist = (y_ends - y_starts).abs().max().int().item()
        
        # Create vertical paths
        for step in range(max_y_dist + 1):
            # Calculate y position at each step for each batch
            y_offset = torch.clamp(step * y_steps, torch.zeros_like(y_steps), (y_ends - y_starts).abs())
            y_pos = (y_starts + y_offset * ((y_ends >= y_starts).float() * 2 - 1)).long()
            
            # Apply mask for valid positions
            valid_steps = (step <= (y_ends - y_starts).abs()) & current_mask.bool()
            
            # Carve path at these positions
            for b in range(batch_size):
                if valid_steps[b]:
                    one_hot[b, 0, y_pos[b], x_ends[b]] = 0
    
    # Add random connections based on difficulty
    # Calculate connection probability (inversely related to difficulty)
    connect_chance = 0.7 - difficulty * 0.5
    
    # Generate random connection masks
    rand_connect = torch.rand((batch_size, size//2, size//2, 4), device=device)
    connect_mask = rand_connect < connect_chance.view(batch_size, 1, 1, 1)
    
    # Process each cell in a vectorized way as much as possible
    for i in range(0, size-2, 2):
        for j in range(0, size-2, 2):
            # Apply north connections
            if i > 0:
                north_mask = connect_mask[:, i//2, j//2, 0]
                one_hot[north_mask, 0, i-1, j] = 0
            
            # Apply east connections
            east_mask = connect_mask[:, i//2, j//2, 1]
            one_hot[east_mask, 0, i, j+1] = 0
            
            # Apply south connections
            south_mask = connect_mask[:, i//2, j//2, 2]
            one_hot[south_mask, 0, i+1, j] = 0
            
            # Apply west connections
            if j > 0:
                west_mask = connect_mask[:, i//2, j//2, 3]
                one_hot[west_mask, 0, i, j-1] = 0
    
    # Ensure entrance and exit are clear for all mazes
    one_hot[:, 0, 0, 0] = 0  # Entrance
    one_hot[:, 0, size-1, size-1] = 0  # Exit
    
    # Ensure paths leading to entrance/exit are clear
    if size > 1:
        one_hot[:, 0, 0, 1] = 0  # Path to the right of entrance
        one_hot[:, 0, 1, 0] = 0  # Path below entrance
        one_hot[:, 0, size-2, size-1] = 0  # Path to exit
        one_hot[:, 0, size-1, size-2] = 0  # Path to exit
    
    # Set channel 1 to be the opposite of channel 0 (if using multi-channel encoding)
    if nchannels > 1:
        one_hot[:, 1, :, :] = 1.0 - one_hot[:, 0, :, :]
    
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
    result = torch.zeros(bsz, dtype=torch.long, device=device)

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
        result[mask_lt_1] = clamped_values.round().to(torch.long)

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
            (batch_size, GRID_SIZE, GRID_SIZE), dtype=torch.long, device=self.device
        )

        # Track agent positions for all environments
        self.agent_positions = torch.zeros(
            (batch_size, 2), dtype=torch.long, device=self.device
        )
        self.goal_positions = torch.full(
            (batch_size, 2), GRID_SIZE - 1, dtype=torch.long, device=self.device
        )
        self.steps_count = torch.zeros(batch_size, dtype=torch.long, device=self.device)

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
    # def reset_subset(self, indices, maze_difficulty=0.5, dist_to_end=0.0):
    #     """
    #     Reset only specific environments indexed by indices

    #     Args:
    #         indices: Boolean mask or integer indices of environments to reset
    #         maze_difficulty: Float or tensor of maze difficulties (0.0-1.0)
    #         dist_to_end: Float or tensor of distances to end (0.0-1.0)

    #     Returns:
    #         Tensor of shape [num_reset, channels, grid_size, grid_size]
    #     """
    #     # Convert indices to boolean mask if it's a tensor of indices
    #     if not isinstance(indices, torch.Tensor) or indices.dtype != torch.bool:
    #         mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
    #         mask[indices] = True
    #     else:
    #         mask = indices

    #     num_to_reset = mask.sum().item()

    #     # Handle scalar difficulty and distance
    #     if not isinstance(maze_difficulty, torch.Tensor):
    #         maze_difficulty = torch.full(
    #             (num_to_reset,), maze_difficulty, dtype=DTYPE, device=self.device
    #         )
    #     elif (
    #         len(maze_difficulty.shape) > 0
    #         and maze_difficulty.shape[0] == self.batch_size
    #     ):
    #         maze_difficulty = maze_difficulty[mask]

    #     if not isinstance(dist_to_end, torch.Tensor):
    #         dist_to_end = torch.full(
    #             (num_to_reset,), dist_to_end, dtype=DTYPE, device=self.device
    #         )
    #     elif len(dist_to_end.shape) > 0 and dist_to_end.shape[0] == self.batch_size:
    #         dist_to_end = dist_to_end[mask]

    #     # Generate new mazes just for the specified environments
    #     new_grids = torch.zeros(
    #         (num_to_reset, self.num_channels, GRID_SIZE, GRID_SIZE),
    #         dtype=DTYPE,
    #         device=self.device,
    #     )

    #     fancy_generate_maze_vectorized(
    #         new_grids,
    #         device=self.device,
    #         nchannels=self.num_channels,
    #         size=GRID_SIZE,
    #         difficulty=maze_difficulty,
    #         batch_size=num_to_reset,
    #     )

    #     # Clear agent channel before placing agents
    #     new_grids[:, 1].zero_()

    #     # Compute starting positions for the reset agents
    #     new_positions = torch.zeros(
    #         (num_to_reset, 2), dtype=torch.long, device=self.device
    #     )
    #     new_positions[:, 0] = f_vectorized(dist_to_end, num_to_reset, self.device)
    #     new_positions[:, 1] = f_vectorized(dist_to_end, num_to_reset, self.device)

    #     # Place agents in the new grids
    #     batch_indices = torch.arange(num_to_reset, device=self.device)
    #     new_grids[batch_indices, 1, new_positions[:, 0], new_positions[:, 1]] = 1

    #     # Update the main grids and agent positions
    #     self.grids[mask] = new_grids
    #     self.agent_positions[mask] = new_positions

    #     # Reset step counters and visit counts for reset environments
    #     self.steps_count[mask] = 0
    #     self.visit_counts[mask].zero_()

    #     # Mark reset environments as active
    #     self.active_envs[mask] = True

    #     return new_grids

    def step(self, actions, active_mask=None):
        """
        Execute actions in all active environments - fully vectorized version
        
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
            active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Initialize rewards and dones for all environments
        rewards = torch.zeros(self.batch_size, device=self.device)
        dones = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Skip if no environments are active
        if not active_mask.any():
            return self.grids.clone(), rewards, dones
        
        # Clear agent positions in active environments
        self.grids[active_mask, 1].zero_()
        
        # Create a lookup tensor for the move_map - precompute this once in __init__ for efficiency
        move_lookup = torch.tensor([
            [-1, 0],  # up
            [1, 0],   # down
            [0, -1],  # left
            [0, 1]    # right
        ], device=self.device)
        
        # Get moves for all active environments (batch_size x 2)
        active_moves = torch.zeros((self.batch_size, 2), dtype=torch.long, device=self.device)
        active_moves[active_mask] = move_lookup[actions[active_mask]]
        
        # Calculate new potential positions for all environments
        new_positions = torch.clamp(
            self.agent_positions + active_moves,
            0, self.grid_size - 1
        )
        
        # Create batch indices tensor for advanced indexing
        batch_indices = torch.arange(self.batch_size, device=self.device)
        
        # Check for wall collisions (0 = no wall, can move)
        # First create a mask of environments where we need to check walls
        check_mask = active_mask.clone()
        # Then check walls at the new positions
        wall_free = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        wall_free[check_mask] = self.grids[
            batch_indices[check_mask], 
            0, 
            new_positions[check_mask, 0], 
            new_positions[check_mask, 1]
        ] == 0
        
        # Create mask of valid moves (active and no wall)
        valid_move_mask = active_mask & wall_free
        
        # Update agent positions where moves are valid
        self.agent_positions[valid_move_mask] = new_positions[valid_move_mask]
        
        # Update visit counts using advanced indexing
        row_indices = self.agent_positions[:, 0]
        col_indices = self.agent_positions[:, 1]
        self.visit_counts[batch_indices[active_mask], row_indices[active_mask], col_indices[active_mask]] += 1
        
        # Place agents in their positions
        self.grids[batch_indices[active_mask], 1, row_indices[active_mask], col_indices[active_mask]] = 1
        
        # Check for goals reached - compare each agent position to its goal position
        at_goal_mask = torch.all(self.agent_positions == self.goal_positions, dim=1)
        goal_reached_mask = active_mask & at_goal_mask
        
        # Update rewards and mark as done for environments reaching goals
        rewards[goal_reached_mask] = 1.0
        dones[goal_reached_mask] = True
        
        # Update steps count for active environments
        self.steps_count[active_mask] += 1
        
        # Check for max steps reached
        max_steps_mask = self.steps_count >= self.max_steps
        dones[active_mask & max_steps_mask] = True
        
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

    def push_transitions(
        self, states, actions, rewards, next_states, dones, active_mask
    ):
        """
        Push multiple transitions to the replay buffer

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        self.replay_buffer.push_batch(
            states[active_mask],
            actions[active_mask],
            rewards[active_mask],
            next_states[active_mask],
            dones[active_mask],
        )

    def set_epsilon(self, episode, total_episodes):
        """
        Set exploration rate based on training progress
        """
        self.epsilon = self.epsilon_min + 0.8 * (1.0 - self.epsilon_min) * exp(
            -3.0 * episode / total_episodes
        )


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

    def __len__(self):
        """Return the current size of the buffer."""
        return self.size

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

