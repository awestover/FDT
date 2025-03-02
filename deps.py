import numpy as np  # isn't it kind of weird to have both numpy and torch
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from mazegen import generate_easy_maze as generate_maze

GRID_SIZE = 8


class GridWorldEnv:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.grid_size = GRID_SIZE
        self.move_map = {
            0: (-1, 0),  # up
            1: (1, 0),  # down
            2: (0, -1),  # left
            3: (0, 1),  # right
        }
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.visit_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
        self.reset()

    def reset(self):
        generate_maze(GRID_SIZE, buffer=self.grid)
        self.agent_pos = (0, 0)
        self.grid[self.agent_pos] = 2
        self.goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.grid[self.goal_pos] = 3
        self.steps = 0
        self.visit_count.zero_()
        return self.get_state()

    def get_state(self):
        return self.grid.copy()

    def step(self, action, gamma=0.99):
        """
        Take an action in the environment and update the state.

        Args:
            action (int): The action to take (0: up, 1: down, 2: left, 3: right).
            gamma (float): Discount factor for future rewards.

        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): The new grid state.
                - reward (float): The reward for taking the action.
                - done (bool): Whether the episode has ended.
        """
        self.steps += 1

        # Check if episode has exceeded maximum steps
        if self.steps >= self.max_steps:
            return self.get_state(), 0, True

        # Constants for rewards
        INVALID_PENALTY = 1
        SLOW_PENALTY = 1
        REVISIT_PENALTY = 1

        # Map action to movement (row, col)
        dr, dc = self.move_map[action]
        r, c = self.agent_pos
        new_r, new_c = r + dr, c + dc

        # Check for out-of-bounds or hitting a wall (cell==1)
        if (
            new_r < 0
            or new_r >= GRID_SIZE
            or new_c < 0
            or new_c >= GRID_SIZE
            or self.grid[new_r, new_c] == 1
        ):
            return self.get_state(), -INVALID_PENALTY, False

        self.agent_pos = (new_r, new_c)
        self.grid[r, c] = 0
        self.grid[new_r, new_c] = 2
        done = (new_r, new_c) == self.goal_pos
        self.visit_count[new_r, new_c] += 1
        reward = -REVISIT_PENALTY * self.visit_count[new_r, new_c] - SLOW_PENALTY
        return self.get_state(), reward, done


class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
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
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.mlp(x)
        return x


class DQNAgent:
    """
    DQN Agent using a simplified neural network architecture with state caching.
    """

    def __init__(
        self,
        lr=3e-4,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=32,
        update_target_every=500,
        device=None,
    ):
        """
        Initialize the DQN agent.
        """
        # Set the device
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"DQNAgent using device: {self.device}")

        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps_done = 0

        # Input channels and actions
        self.input_channels = 4  # one-hot encoded grid values
        self.num_actions = 4  # up, down, left, right

        # Initialize networks
        self.policy_net = DQN(self.input_channels, self.num_actions).to(self.device)
        self.target_net = DQN(self.input_channels, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998

        # Reward tracking
        self.reward_window = deque(maxlen=100)

        # State cache for faster processing
        self.state_cache = {}

    def preprocess(self, state):
        """
        Preprocess a single state into a tensor with caching.
        """
        # Convert numpy array to bytes for hashing
        state_bytes = state.tobytes()

        # Check if we've already processed this state
        if state_bytes in self.state_cache:
            return self.state_cache[state_bytes]

        # Process the state if not in cache
        state_tensor = torch.from_numpy(state).long()
        one_hot = F.one_hot(state_tensor, num_classes=4)
        # Rearrange to (channels, height, width)
        one_hot = one_hot.permute(2, 0, 1).float()
        # Move to the appropriate device
        preprocessed = one_hot.unsqueeze(0).to(self.device)  # add batch dimension

        # Store in cache
        self.state_cache[state_bytes] = preprocessed

        return preprocessed

    def batch_preprocess(self, states):
        """
        Efficiently preprocess a batch of states.
        """
        processed_states = []
        for state in states:
            processed_states.append(self.preprocess(state))

        return torch.cat(processed_states)

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy based on current state.
        """
        sample = random.random()

        # If we're exploring
        if sample < self.epsilon:
            #  exploration with some bias toward goal
            r, c = None, None
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    if state[i, j] == 2:  # Agent position
                        r, c = i, j
                        break
                if r is not None:
                    break

            if r is not None:
                # Bias exploration toward goal
                goal_r, goal_c = GRID_SIZE - 1, GRID_SIZE - 1

                # Calculate direction weights
                weights = np.ones(4) * 0.25  # Equal probability by default

                # Up, Down, Left, Right
                if r > 0 and state[r - 1, c] != 1:  # Can move up
                    if r > goal_r:  # If agent is below goal, increase up probability
                        weights[0] = 0.4
                if r < GRID_SIZE - 1 and state[r + 1, c] != 1:  # Can move down
                    if r < goal_r:  # If agent is above goal, increase down probability
                        weights[1] = 0.4
                if c > 0 and state[r, c - 1] != 1:  # Can move left
                    if (
                        c > goal_c
                    ):  # If agent is to the right of goal, increase left probability
                        weights[2] = 0.4
                if c < GRID_SIZE - 1 and state[r, c + 1] != 1:  # Can move right
                    if (
                        c < goal_c
                    ):  # If agent is to the left of goal, increase right probability
                        weights[3] = 0.4

                weights = weights / weights.sum()  # Normalize

                return np.random.choice(self.num_actions, p=weights)

            return random.randrange(self.num_actions)
        else:
            # Exploit - use the policy network to make a prediction
            with torch.no_grad():
                state_tensor = self.preprocess(state)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def update(self):
        """
        Update network weights using experiences from the buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Preprocess states and next_states using batch processing
        state_batch = self.batch_preprocess(states)
        next_state_batch = self.batch_preprocess(next_states)

        # Convert other variables to tensors
        action_batch = torch.tensor(actions, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute target Q values
        with torch.no_grad():
            # Double DQN
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch.unsqueeze(1) + self.gamma * next_q_values * (
                1 - done_batch.unsqueeze(1)
            )

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #  gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def set_epsilon(self, episode, total_episodes):
        """
        Adjust exploration rate based on training progress.
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_min
            + (1.0 - self.epsilon_min) * np.exp(-3.0 * episode / total_episodes),
        )

    def zero_epsilon(self):
        """
        Set epsilon to zero for evaluation.
        """
        self.epsilon = 0

    def clear_cache(self):
        """
        Clear the state cache to free memory.
        """
        self.state_cache = {}


def efficient_one_hot(state, num_classes=4):
    """
    A more efficient implementation of one-hot encoding for grid states.

    Args:
        state: NumPy array representing the grid state
        num_classes: Number of possible values in the grid

    Returns:
        torch.Tensor: One-hot encoded tensor ready for neural network input
    """
    # Get dimensions
    height, width = state.shape

    # Pre-allocate the tensor on CPU for efficiency
    result = torch.zeros(num_classes, height, width, dtype=torch.float32)

    # Fill the tensor directly - avoiding the overhead of F.one_hot
    for c in range(num_classes):
        result[c] = torch.from_numpy((state == c).astype(np.float32))

    return result


class FastDQNAgent(DQNAgent):
    """
    DQN Agent with optimized preprocessing for faster training.
    """

    def __init__(self, *args, **kwargs):
        super(FastDQNAgent, self).__init__(*args, **kwargs)
        self.state_cache = {}  # Cache processed states
        self.cache_hits = 0
        self.cache_misses = 0

    def preprocess(self, state):
        """
        Preprocess a single state into a tensor with efficient encoding and caching.
        """
        # Convert to bytes for hashing (numpy arrays aren't hashable)
        state_bytes = state.tobytes()

        # Return from cache if available
        if state_bytes in self.state_cache:
            self.cache_hits += 1
            return self.state_cache[state_bytes]

        # Process the state efficiently if not in cache
        self.cache_misses += 1
        one_hot = efficient_one_hot(state, self.input_channels)
        preprocessed = one_hot.unsqueeze(0).to(
            self.device
        )  # Add batch dimension and move to device

        # Cache the result
        self.state_cache[state_bytes] = preprocessed

        return preprocessed

    def batch_preprocess(self, states):
        """
        Efficiently preprocess multiple states at once.

        Args:
            states: List of grid states (numpy arrays)

        Returns:
            torch.Tensor: Batch of preprocessed states
        """
        # Use cached states where available or process new ones
        batch = []
        for state in states:
            batch.append(self.preprocess(state))

        # Concatenate along batch dimension
        return torch.cat(batch)

    def cache_stats(self):
        """
        Return cache hit/miss statistics.
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": hit_rate,
        }

    def limit_cache_size(self, max_size=1000):
        """
        Limit the cache size to avoid memory issues.
        """
        if len(self.state_cache) > max_size:
            # Keep most recently used items by recreating the cache
            # This is a simple approach - could be more sophisticated
            items = list(self.state_cache.items())
            self.state_cache = dict(items[-max_size:])


# Update the init_agent function to use the faster agent
def init_fast_agent(device):
    agent = FastDQNAgent(
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=64,
        update_target_every=200,
        device=device,
    )
    return agent


class ReplayBuffer:
    """
    replay buffer that stores individual transitions rather than sequences.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Return current buffer size.
        """
        return len(self.buffer)


class VectorizedReplayBuffer(ReplayBuffer):
    """
    Enhanced replay buffer with vectorized batch processing capabilities.
    """

    def __init__(self, capacity):
        super(VectorizedReplayBuffer, self).__init__(capacity)

    def get_batch_indices(self, batch_size):
        """
        Get indices for a batch sample.
        """
        return random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))

    def get_vectorized_batch(self, batch_size):
        """
        Sample a batch in a format that allows for vectorized processing.
        """
        indices = self.get_batch_indices(batch_size)
        batch = [self.buffer[idx] for idx in indices]

        # Extract components efficiently
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])

        return states, actions, rewards, next_states, dones

    def get_transitions_as_arrays(self):
        """
        Get all transitions as separate arrays for vectorized processing.
        """
        if not self.buffer:
            return None

        states, actions, rewards, next_states, dones = zip(*self.buffer)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )


class OptimizedDQNAgent(FastDQNAgent):
    """
    Further optimized DQN agent with vectorized operations.
    """

    def __init__(self, *args, **kwargs):
        super(OptimizedDQNAgent, self).__init__(*args, **kwargs)
        # Replace standard replay buffer with vectorized version
        capacity = self.replay_buffer.capacity
        self.replay_buffer = VectorizedReplayBuffer(capacity)

        # Pre-allocate tensors for common operations
        self.action_tensor = None
        self.reward_tensor = None
        self.done_tensor = None

    def update(self):
        """
        Optimized update method using vectorized operations.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch in vectorized format
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.get_vectorized_batch(self.batch_size)
        )

        # Preprocess states efficiently
        state_batch = self.batch_preprocess(states)
        next_state_batch = self.batch_preprocess(next_states)

        # Convert other variables to tensors with reuse
        if self.action_tensor is None or self.action_tensor.shape[0] != len(actions):
            self.action_tensor = torch.tensor(actions, device=self.device).unsqueeze(1)
            self.reward_tensor = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            )
            self.done_tensor = torch.tensor(
                dones, dtype=torch.float32, device=self.device
            )
        else:
            # Reuse existing tensors
            self.action_tensor.copy_(
                torch.tensor(actions, device=self.device).unsqueeze(1)
            )
            self.reward_tensor.copy_(
                torch.tensor(rewards, dtype=torch.float32, device=self.device)
            )
            self.done_tensor.copy_(
                torch.tensor(dones, dtype=torch.float32, device=self.device)
            )

        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, self.action_tensor)

        # Compute target Q values
        with torch.no_grad():
            # Double DQN
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = self.reward_tensor.unsqueeze(
                1
            ) + self.gamma * next_q_values * (1 - self.done_tensor.unsqueeze(1))

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


# Update the init_agent function to use the optimized agent
def init_optimized_agent(device):
    agent = OptimizedDQNAgent(
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=64,
        update_target_every=200,
        device=device,
    )
    return agent


class CurriculumGridWorldEnv(OptimizedGridWorldEnv):
    """
    Grid world environment that supports curriculum learning by:
    1. Adjusting maze difficulty
    2. Placing the agent closer to the goal in early training
    """

    def __init__(self, max_steps=200, revisit_penalty_factor=0.2, decay_rate=0.95):
        super(CurriculumGridWorldEnv, self).__init__(
            max_steps=max_steps,
            revisit_penalty_factor=revisit_penalty_factor,
            decay_rate=decay_rate,
        )

    def reset(self, difficulty=0.5, start_distance=1.0):
        """
        Reset the environment with adjustable difficulty and starting position.

        Args:
            difficulty (float): Value between 0.0 and 1.0 controlling maze difficulty
            start_distance (float): Value between 0.0 and 1.0 controlling how far
                                   the agent starts from the goal (1.0 = furthest)

        Returns:
            np.ndarray: A copy of the initial grid state.
        """
        # Generate maze with specified difficulty
        self.grid = generate_maze(GRID_SIZE, difficulty)

        # Place the goal (3) at the bottom-right
        self.goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.grid[self.goal_pos] = 3

        # Calculate starting position based on start_distance
        if start_distance < 1.0:
            # Calculate maximum possible Manhattan distance
            max_distance = 2 * (GRID_SIZE - 1)

            # Calculate target distance from goal
            target_distance = int(max_distance * start_distance)

            # Find a valid starting position with approximately the target distance
            valid_positions = []
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if self.grid[r, c] == 0:  # Empty cell
                        dist = abs(r - self.goal_pos[0]) + abs(c - self.goal_pos[1])
                        if abs(dist - target_distance) <= 2:  # Allow some flexibility
                            valid_positions.append((r, c, dist))

            # If we found valid positions, choose one randomly
            if valid_positions:
                # Sort by how close they are to the target distance
                valid_positions.sort(key=lambda x: abs(x[2] - target_distance))
                # Choose from the top 3 closest matches or all if fewer
                top_n = min(3, len(valid_positions))
                r, c, _ = random.choice(valid_positions[:top_n])
                self.agent_pos = (r, c)
            else:
                # Fallback to default starting position
                self.agent_pos = (0, 0)
        else:
            # Default starting position (top-left)
            self.agent_pos = (0, 0)

        # Place the agent
        self.grid[self.agent_pos] = 2

        # Initialize visitation map
        self.visit_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.visit_count[self.agent_pos] = 1.0

        self.steps = 0
        return self.get_state()
