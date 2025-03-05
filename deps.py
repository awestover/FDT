import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import random as nprand
from random import random as rand
from math import exp

GRID_SIZE = 15
GS = GRID_SIZE
HALF_GS = GRID_SIZE // 2
assert GRID_SIZE % 2 == 1, "GRID_SIZE must be an odd number for historical reasons"
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

def generate_maze(buffer, batch_size, device, difficulty):
    assert 0 <= difficulty <= 1, "Difficulty must be between 0 and 1"

    # For directions
    directions = torch.tensor([
        [-1, 0],  # North
        [0, 1],   # East
        [1, 0],   # South
        [0, -1]   # West
    ], device=device)

    # Initialize mazes with all walls
    mazes = buffer
    mazes.fill_(0)
    
    # Initialize visit state tensor
    # 0 = unvisited, 1 = in stack, 2 = visited and backtracked
    visit_state = torch.zeros((batch_size, GRID_SIZE, GRID_SIZE), dtype=torch.uint8, device=device)
    y_coords = torch.arange(HALF_GS, device=device) * 2 + 1
    x_coords = torch.arange(HALF_GS, device=device) * 2 + 1
    
    # Select random starting cells for each maze in batch
    idx_y = torch.randint(0, len(y_coords), (batch_size,), device=device)
    idx_x = torch.randint(0, len(x_coords), (batch_size,), device=device)
    start_y = y_coords[idx_y]
    start_x = x_coords[idx_x]
    
    # Create batch indices for fancy indexing
    batch_indices = torch.arange(batch_size, device=device)
    
    # Mark starting points as paths and in-stack using fancy indexing
    mazes[batch_indices, start_y, start_x] = 1
    visit_state[batch_indices, start_y, start_x] = 1
    
    # Create GPU stack representation - use a 3D tensor to represent stack state
    # We'll use the visit_state to track which cells are in the stack
    # and top_ptr to track the top of each stack
    max_stack_size = HALF_GS**2  # Maximum possible stack size
    stack = torch.zeros((batch_size, max_stack_size, 2), dtype=torch.int32, device=device)
    top_ptr = torch.ones(batch_size, dtype=torch.int32, device=device)  # All stacks start with 1 element
    
    # Initialize stacks with starting positions using fancy indexing
    # Convert to int32 to match stack tensor type
    stack[batch_indices, 0, 0] = start_y.to(torch.int32)
    stack[batch_indices, 0, 1] = start_x.to(torch.int32)
    
    # Track active mazes
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # Main generation loop
    while active.any():
        # Get current position for all active mazes using fancy indexing
        # Create indices for the top of each stack
        stack_top_indices = top_ptr - 1
        stack_top_indices = torch.where(stack_top_indices >= 0, stack_top_indices, torch.zeros_like(stack_top_indices))
        
        # Extract current positions for all mazes
        current_pos = torch.zeros((batch_size, 2), dtype=torch.int32, device=device)
        mask = active & (top_ptr > 0)
        if mask.any():
            current_pos[mask, 0] = stack[mask, stack_top_indices[mask], 0]  # y
            current_pos[mask, 1] = stack[mask, stack_top_indices[mask], 1]  # x
        
        # Generate random direction order for each maze
        dir_indices = torch.arange(4, device=device).expand(batch_size, 4)
        shuffle_values = torch.rand(batch_size, 4, device=device)
        _, dir_order = shuffle_values.sort(dim=1)
        
        # Check all 4 directions in parallel
        has_unvisited = torch.zeros(batch_size, dtype=torch.bool, device=device)
        next_y = torch.zeros(batch_size, dtype=torch.int32, device=device)
        next_x = torch.zeros(batch_size, dtype=torch.int32, device=device)
        wall_y = torch.zeros(batch_size, dtype=torch.int32, device=device)
        wall_x = torch.zeros(batch_size, dtype=torch.int32, device=device)
        
        # Process each direction in tensor operations
        for dir_idx in range(4):
            # Get direction based on random ordering
            d = dir_order[:, dir_idx] % 4
            
            # Calculate neighbor positions (2 cells away)
            dy = directions[d, 0]
            dx = directions[d, 1]
            
            # Calculate neighbor and wall positions for all mazes simultaneously
            ny = current_pos[:, 0] + 2 * dy
            nx = current_pos[:, 1] + 2 * dx
            wy = current_pos[:, 0] + dy
            wx = current_pos[:, 1] + dx
            
            # Check boundary conditions
            valid = (ny >= 0) & (ny < GRID_SIZE) & (nx >= 0) & (nx < GRID_SIZE)
            
            # Check if neighbor is unvisited 
            # Only update mazes that are active, have a non-empty stack, and haven't found an unvisited neighbor yet
            mask = active & (top_ptr > 0) & ~has_unvisited & valid

            valid_ny = torch.clamp(ny, 0, GRID_SIZE-1)
            valid_nx = torch.clamp(nx, 0, GRID_SIZE-1)
            
            # Get visit state for valid neighbors using fancy indexing
            if mask.any():
                # Create a submask for mazes with unvisited neighbors in this direction
                unvisited_mask = mask & (visit_state[batch_indices, valid_ny, valid_nx] == 0)
                
                if unvisited_mask.any():
                    # Update for mazes that have unvisited neighbors and haven't found one yet
                    update_mask = unvisited_mask & ~has_unvisited
                    if update_mask.any():
                        has_unvisited[update_mask] = True
                        next_y[update_mask] = ny[update_mask].to(torch.int32)
                        next_x[update_mask] = nx[update_mask].to(torch.int32)
                        wall_y[update_mask] = wy[update_mask].to(torch.int32)
                        wall_x[update_mask] = wx[update_mask].to(torch.int32)
        
        # Process all mazes in parallel using fancy indexing
        # Create masks for different operations
        active_mask = active & (top_ptr > 0)
        carve_mask = active_mask & has_unvisited
        backtrack_mask = active_mask & ~has_unvisited
        
        # For mazes with unvisited neighbors: carve paths
        if carve_mask.any():
            # Carve paths to neighbors
            mazes[carve_mask, wall_y[carve_mask], wall_x[carve_mask]] = 1
            mazes[carve_mask, next_y[carve_mask], next_x[carve_mask]] = 1
            
            # Mark as in-stack
            visit_state[carve_mask, next_y[carve_mask], next_x[carve_mask]] = 1
            
            # Push to stack
            stack[carve_mask, top_ptr[carve_mask], 0] = next_y[carve_mask]
            stack[carve_mask, top_ptr[carve_mask], 1] = next_x[carve_mask]
            top_ptr[carve_mask] += 1
        
        # For mazes with no unvisited neighbors: backtrack
        if backtrack_mask.any():
            # Get current cells at top of stack
            backtrack_indices = top_ptr[backtrack_mask] - 1
            cy = stack[backtrack_mask, backtrack_indices, 0]
            cx = stack[backtrack_mask, backtrack_indices, 1]
            
            # Mark current cells as backtracked
            visit_state[backtrack_mask, cy, cx] = 2
            
            # Pop from stack
            top_ptr[backtrack_mask] -= 1
            
            # Check for completed mazes
            completed_mask = backtrack_mask & (top_ptr == 0)
            active[completed_mask] = False
    
    
    # Apply difficulty by drilling holes in the walls based on the difficulty parameter
    # Create a random mask where 1 = keep wall, 0 = drill hole
    # Higher difficulty = fewer holes (more walls kept)
    if difficulty < 1.0:
        # Only consider non-border cells for drilling
        border_mask = torch.ones_like(mazes, dtype=torch.bool)
        border_mask[:, 1:-1, 1:-1] = False  # Interior cells
        
        # Create random bernoulli mask with probability = difficulty
        # 1 = keep wall, 0 = drill hole
        wall_mask = torch.bernoulli(torch.full_like(mazes.float(), difficulty))
        
        # Only drill holes in walls (where maze value is 0), and not at borders
        drill_mask = (mazes == 0) & ~border_mask & (wall_mask == 0)
        
        # Apply the mask to drill holes in the maze walls
        mazes[drill_mask] = 1
    
    return mazes

class MazeCache:
    def __init__(self, device, batch_size, num_mazes=1_000_000):
        """
        Pre-generate and cache a large number of mazes
        
        Args:
            device: The torch device to place tensors on
            batch_size: Number of parallel environments
            num_mazes: Total number of mazes to pre-generate
        """
        self.device = device
        self.batch_size = batch_size
        self.num_mazes = num_mazes
        
        # Calculate how many batches of mazes to generate
        self.batch_count = (num_mazes + batch_size - 1) // batch_size
        
        # Pre-allocate tensor for all mazes
        self.mazes = torch.zeros(
            (num_mazes, GRID_SIZE, GRID_SIZE),
            dtype=DTYPE,
            device=device
        )
        
        print(f"Pre-generating {num_mazes} mazes...")
        
        # Generate mazes in batches to avoid memory issues
        for i in range(self.batch_count):
            if i % 100 == 0:
                print(f"Generating batch {i+1}/{self.batch_count}")
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_mazes)
            current_batch_size = end_idx - start_idx
            
            # Generate one batch of mazes
            maze_batch = torch.zeros(
                (current_batch_size, GRID_SIZE, GRID_SIZE),
                dtype=DTYPE,
                device=device
            )
            generate_maze(maze_batch, current_batch_size, device, difficulty=1.0)
            
            # Store in cache
            self.mazes[start_idx:end_idx] = maze_batch
            
        print(f"Finished generating {num_mazes} mazes")

    def get_random_mazes(self, batch_size):
        """Return a batch of random mazes from the cache"""
        indices = torch.randint(0, self.num_mazes, (batch_size,), device=self.device)
        return self.mazes[indices]


class GridWorldEnv:
    def __init__(self, device, max_steps, batch_size, maze_cache):
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

        self.maze_cache = maze_cache

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
        self.grids[:,0,:,:] = self.maze_cache.get_random_mazes(self.batch_size)

        # Clear agent channel before placing agents
        self.grids[:, 1].zero_()

        # Compute starting positions for all agents in batch
        smallest_pos = int((GRID_SIZE-1)*(1-dist_to_end))
        self.agent_positions = torch.randint(low=smallest_pos, high=GRID_SIZE, size=(self.batch_size,2), device=self.device)

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
        # note that self.grids[active_mask, 1].zero_() doesn't work!!! 
        # because mask creates a copy?
        self.grids[active_mask, 1] = 0
        
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
        rewards[active_mask] = -1.0
        rewards[goal_reached_mask] = 10.0
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


if __name__ == "__main__":
    bsz = 4
    grids = torch.zeros((bsz, 2, GS, GS), dtype=DTYPE)
    generate_maze(
        grids[:,0,:,:],
        bsz,
        "cpu",
        difficulty=1
    )
    import matplotlib.pyplot as plt
    plt.imshow(grids[0, 0,:,:].numpy(), cmap="binary")
    plt.show()

