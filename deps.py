import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from mazegen import generate_simple_maze as generate_maze

GRID_SIZE = 8
# --- Environment Definition ---
class GridWorldEnv:
    """
    Environment representing an 8x8 grid world.
    EDIT: actually the grid_size doesn't need to be 8

    The grid consists of:
        0: empty cell (agent can move here)
        1: blocked cell (wall)
        2: the agent's current position
        3: the goal position

    The agent can take one of four actions:
        0: move up
        1: move down
        2: move left
        3: move right

    An action that would move the agent off the board or into a wall fails,
    leaving the agent in its current position. Every step incurs a reward of -1,
    and the episode ends when the agent reaches the goal or when the maximum steps are reached.
    """

    def __init__(self, max_steps=200):
        """
        Initialize the GridWorld environment.

        Args:
            max_steps (int): Maximum number of steps allowed per episode.
        """
        self.max_steps = max_steps
        self.action_space = 4  # 0: up, 1: down, 2: left, 3: right
        self.reset()
        self.grid_size = GRID_SIZE
    
    def reset(self):
        """
        Reset the environment to the initial state.

        This function creates an 8x8 grid with walls in every odd-numbered column
        (with one randomly removed wall per such column). It places the agent at the
        top-left (position (0,0)) and the goal at the bottom-right.
        
        Returns:
            np.ndarray: A copy of the initial grid state.
        """
        self.grid = generate_maze(GRID_SIZE)

        # Place the agent (2) at a fixed starting position (top-left)
        self.agent_pos = (0, 0)
        self.grid[self.agent_pos] = 2
        
        # Place the goal (3) at a fixed location (bottom-right)
        self.goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.grid[self.goal_pos] = 3
        
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        return self.grid.copy()
    
    def step(self, action):
        """
        Take an action in the environment and update the state.

        Args:
            action (int): The action to take (0: up, 1: down, 2: left, 3: right).

        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): The new grid state.
                - reward (float): The reward for taking the action.
                - done (bool): Whether the episode has ended.
        """
        self.steps += 1
        if self.steps >= self.max_steps:
            return self.get_state(), 0, True

        INVALID_PENALTY = -2
        WIN_BONUS = 2*self.max_steps
        # normal score: change in distance to goal, with a bit of discounting

        # Map action to movement (row, col)
        move_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        dr, dc = move_map[action]
        r, c = self.agent_pos
        new_r, new_c = r + dr, c + dc
        
        # Check for out-of-bounds or hitting a wall (cell==1)
        if (new_r < 0 or new_r >= GRID_SIZE or 
            new_c < 0 or new_c >= GRID_SIZE or
            self.grid[new_r, new_c] == 1):
            return self.get_state(), INVALID_PENALTY, False

        # Valid move: update the grid
        old_dist = self.dist_to_goal()
        self.agent_pos = (new_r, new_c)
        new_dist = self.dist_to_goal()
        self.grid[r, c] = 0
        self.grid[new_r, new_c] = 2
        
        done = (new_r, new_c) == self.goal_pos
        
        if done:
            reward = 2*max_steps  # Big reward for reaching goal
        else:
            reward = old_dist - self.gamma*new_dist
                
        return self.get_state(), reward, done

    def dist_to_goal(self):
        r, c = self.agent_pos
        gr, gc = self.goal_pos
        return abs(gr - r) + abs(gc - c)        

    def render(self):
        for row in self.grid:
            print(" ".join(str(int(cell)) for cell in row))


# --- Modified DQN Network Definition ---
class DQN(nn.Module):
    """
    Improved DQN architecture optimized for maze navigation with better spatial reasoning.
    """
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        
        # More filters to capture complex spatial patterns
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Add batch normalization for stability
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Add another convolutional layer for deeper spatial understanding
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate input size for fully connected layer
        fc_input_size = 64 * (GRID_SIZE // 4) * (GRID_SIZE // 4)
        
        # Enhanced fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout = nn.Dropout(0.2)  # Add dropout to prevent overfitting
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Improved DQN Agent ---
class DQNAgent:
    """
    Enhanced DQN agent with improved exploration, learning strategy, and reward shaping.
    """

    def __init__(self, lr=3e-4, gamma=0.99, buffer_capacity=50000, batch_size=128, update_target_every=500):
        """
        Initialize the DQN agent with improved parameters.
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps_done = 0
        self.learning_steps = 0
        
        # Our grid has 4 distinct values (0,1,2,3) -> one-hot encoded to 4 channels.
        self.input_channels = 4
        self.num_actions = 4
        
        # Use the improved network architecture
        self.policy_net = DQN(self.input_channels, self.num_actions)
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # set target net to evaluation mode
        
        # Use a more sophisticated optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler to reduce LR over time
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        
        # Improved replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=0.6, beta=0.4)
        
        # Better exploration strategy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998  # Slower decay for better exploration
        
        # For tracking progress
        self.best_reward = -float('inf')
        self.reward_window = deque(maxlen=100)
    
    def preprocess(self, state):
        """
        Preprocess the grid state by converting it into a one-hot encoded tensor.
        """
        state_tensor = torch.from_numpy(state).long()
        one_hot = F.one_hot(state_tensor, num_classes=4)
        # Rearrange to (channels, height, width)
        one_hot = one_hot.permute(2, 0, 1).float()
        return one_hot.unsqueeze(0)  # add batch dimension
    
    def select_action(self, state):
        """
        Select an action using an improved epsilon-greedy strategy.
        """
        sample = random.random()
        if sample < self.epsilon:
            # More sophisticated exploration strategy - include some directional bias
            # toward unexplored areas based on current position
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
                if r > 0 and state[r-1, c] != 1:  # Can move up
                    if r > goal_r:  # If agent is below goal, increase up probability
                        weights[0] = 0.4
                if r < GRID_SIZE-1 and state[r+1, c] != 1:  # Can move down
                    if r < goal_r:  # If agent is above goal, increase down probability
                        weights[1] = 0.4
                if c > 0 and state[r, c-1] != 1:  # Can move left
                    if c > goal_c:  # If agent is to the right of goal, increase left probability
                        weights[2] = 0.4
                if c < GRID_SIZE-1 and state[r, c+1] != 1:  # Can move right
                    if c < goal_c:  # If agent is to the left of goal, increase right probability
                        weights[3] = 0.4
                
                weights = weights / weights.sum()  # Normalize
                
                return np.random.choice(self.num_actions, p=weights)
            
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = self.preprocess(state)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update(self):
        """
        Update the policy network with improved experience replay and learning process.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample experiences with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Preprocess states and next_states
        state_batch = torch.cat([self.preprocess(s) for s in states])
        next_state_batch = torch.cat([self.preprocess(s) for s in next_states])
        action_batch = torch.tensor(actions).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        done_batch = torch.tensor(dones, dtype=torch.float32)
        weights_batch = torch.tensor(weights, dtype=torch.float32)
        
        # Compute current Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next state values using target network (Double DQN)
        with torch.no_grad():
            # Get actions from policy net
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            # Get Q-values from target net for those actions
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            # Use target Q-values in Bellman equation
            target_q_values = reward_batch.unsqueeze(1) + self.gamma * next_q_values * (1 - done_batch.unsqueeze(1))
        
        # Calculate TD errors for priority updates
        td_errors = torch.abs(q_values - target_q_values).detach().numpy()
        
        # Compute weighted Huber loss
        loss = F.smooth_l1_loss(q_values, target_q_values, reduction='none')
        weighted_loss = (loss * weights_batch.unsqueeze(1)).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities in the replay buffer with a small constant to prevent zero priority
        self.replay_buffer.update_priorities(indices, td_errors + 1e-5)
        
        # Periodically update the target network
        self.steps_done += 1
        self.learning_steps += 1
        
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Update learning rate
        self.scheduler.step()
            
        return weighted_loss.item()
    
    def set_epsilon(self, episode, total_episodes):
        """
        Improved epsilon decay strategy for better exploration-exploitation balance.
        """
        # More sophisticated decay that starts slow and accelerates
        if episode < total_episodes * 0.2:
            # Slow decay in the beginning to encourage exploration
            self.epsilon = max(self.epsilon_min, 1.0 - episode / (total_episodes * 0.5))
        else:
            # Faster decay later to focus on exploitation
            self.epsilon = max(self.epsilon_min, 
                               self.epsilon_min + (1.0 - self.epsilon_min) * 
                               np.exp(-5.0 * (episode - total_episodes * 0.2) / total_episodes))

    def zero_epsilon(self):
        self.epsilon = 0

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer that stores transitions based on TD error.
    This helps the agent learn more efficiently from important experiences.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity (int): Maximum size of the buffer
            alpha (float): How much prioritization to use (0 = uniform sampling)
            beta (float): Importance sampling correction factor (0 = no correction)
            beta_increment (float): How much to increase beta per sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.ones(capacity)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: Batch of experiences and importance sampling weights
        """
        if len(self.buffer) < batch_size:
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        else:
            # Calculate sampling probabilities
            priorities = self.priorities[:len(self.buffer)]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            # Sample based on priorities
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get the sampled experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states), 
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices (list): Indices to update
            td_errors (list): TD errors for each experience
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5  # Add small constant to avoid zero priority
    
    def __len__(self):
        """
        Return current buffer size.
        """
        return len(self.buffer)

    
