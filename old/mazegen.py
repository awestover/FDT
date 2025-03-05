import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class GPUMazeGenerator:
    def __init__(self, height, width, batch_size=100, device='cuda'):
        """
        Initialize a GPU-based maze generator.
        
        Args:
            height (int): Height of the maze (must be odd)
            width (int): Width of the maze (must be odd)
            batch_size (int): Number of mazes to generate in parallel
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.device = device
        assert height % 2 == 1, "Height must be an odd number"
        assert width % 2 == 1, "Width must be an odd number"
        
        # For visualization
        self.directions = torch.tensor([
            [-1, 0],  # North
            [0, 1],   # East
            [1, 0],   # South
            [0, -1]   # West
        ], device=self.device)

    
    def initialize_mazes(self):
        """Initialize mazes with all walls using tensor operations."""
        # Create batch indices for fancy indexing
        batch_indices = torch.arange(self.batch_size, device=self.device)
        
        # 0 = wall, 1 = path
        # Start with all walls (0s)
        mazes = torch.zeros((self.batch_size, self.height, self.width), 
                          dtype=torch.uint8, device=self.device)
        
        # Set random starting points (odd coordinates only)
        y_start = torch.randint(0, (self.height-1)//2, (self.batch_size,), device=self.device) * 2 + 1
        x_start = torch.randint(0, (self.width-1)//2, (self.batch_size,), device=self.device) * 2 + 1
        
        # Mark starting points as paths using fancy indexing
        mazes[batch_indices, y_start, x_start] = 1
            
        # Initialize visited tensor
        visited = torch.zeros((self.batch_size, self.height, self.width), 
                             dtype=torch.bool, device=self.device)
        
        # Mark starting points as visited using fancy indexing
        visited[batch_indices, y_start, x_start] = True
        
        # Initialize stacks for tracking
        # Convert to tensor format for stack items
        stack_y = y_start.unsqueeze(1)  # Shape: [batch_size, 1]
        stack_x = x_start.unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Create stacks as a list of tensors
        # Each tensor has shape [batch_size, 1] and represents the stack state for all mazes
        stacks = [(stack_y, stack_x)]
            
        return mazes, stacks, visited
    
    def parallel_dfs(self, max_iters=None):
        """
        Generate mazes using a truly parallelized randomized DFS approach.
        
        Args:
            max_iters (int, optional): Maximum iterations for generation
                                      (prevents infinite loops)
        
        Returns:
            torch.Tensor: Batch of generated mazes
        """
        # Initialize mazes with all walls
        mazes = torch.zeros((self.batch_size, self.height, self.width), 
                          dtype=torch.uint8, device=self.device)
        
        # Initialize visit state tensor
        # 0 = unvisited, 1 = in stack, 2 = visited and backtracked
        visit_state = torch.zeros((self.batch_size, self.height, self.width), 
                                dtype=torch.uint8, device=self.device)
        
        height_cells = (self.height - 1) // 2
        width_cells = (self.width - 1) // 2
        y_coords = torch.arange(height_cells, device=self.device) * 2 + 1
        x_coords = torch.arange(width_cells, device=self.device) * 2 + 1
        
        # Select random starting cells for each maze in batch
        idx_y = torch.randint(0, len(y_coords), (self.batch_size,), device=self.device)
        idx_x = torch.randint(0, len(x_coords), (self.batch_size,), device=self.device)
        start_y = y_coords[idx_y]
        start_x = x_coords[idx_x]
        
        # Create batch indices for fancy indexing
        batch_indices = torch.arange(self.batch_size, device=self.device)
        
        # Mark starting points as paths and in-stack using fancy indexing
        mazes[batch_indices, start_y, start_x] = 1
        visit_state[batch_indices, start_y, start_x] = 1
        
        # Create GPU stack representation - use a 3D tensor to represent stack state
        # We'll use the visit_state to track which cells are in the stack
        # and top_ptr to track the top of each stack
        max_stack_size = (self.height // 2) * (self.width // 2)  # Maximum possible stack size
        stack = torch.zeros((self.batch_size, max_stack_size, 2), dtype=torch.int32, device=self.device)
        top_ptr = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)  # All stacks start with 1 element
        
        # Initialize stacks with starting positions using fancy indexing
        # Convert to int32 to match stack tensor type
        stack[batch_indices, 0, 0] = start_y.to(torch.int32)
        stack[batch_indices, 0, 1] = start_x.to(torch.int32)
        
        # Track active mazes
        active = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Setup for reporting progress
        iter_count = 0
        pbar = tqdm(total=self.batch_size, desc="Generating mazes")
        completed = 0
        
        # Main generation loop
        while active.any():
            # Check if max iterations reached
            if max_iters is not None and iter_count >= max_iters:
                break
            
            # Get current position for all active mazes using fancy indexing
            # Create indices for the top of each stack
            stack_top_indices = top_ptr - 1
            stack_top_indices = torch.where(stack_top_indices >= 0, stack_top_indices, torch.zeros_like(stack_top_indices))
            
            # Extract current positions for all mazes
            current_pos = torch.zeros((self.batch_size, 2), dtype=torch.int32, device=self.device)
            mask = active & (top_ptr > 0)
            if mask.any():
                current_pos[mask, 0] = stack[mask, stack_top_indices[mask], 0]  # y
                current_pos[mask, 1] = stack[mask, stack_top_indices[mask], 1]  # x
            
            # Generate random direction order for each maze
            dir_indices = torch.arange(4, device=self.device).expand(self.batch_size, 4)
            shuffle_values = torch.rand(self.batch_size, 4, device=self.device)
            _, dir_order = shuffle_values.sort(dim=1)
            
            # Check all 4 directions in parallel
            has_unvisited = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            next_y = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
            next_x = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
            wall_y = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
            wall_x = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
            
            # Process each direction in tensor operations
            for dir_idx in range(4):
                # Get direction based on random ordering
                d = dir_order[:, dir_idx] % 4
                
                # Calculate neighbor positions (2 cells away)
                dy = self.directions[d, 0]
                dx = self.directions[d, 1]
                
                # Calculate neighbor and wall positions for all mazes simultaneously
                ny = current_pos[:, 0] + 2 * dy
                nx = current_pos[:, 1] + 2 * dx
                wy = current_pos[:, 0] + dy
                wx = current_pos[:, 1] + dx
                
                # Check boundary conditions
                valid = (ny >= 0) & (ny < self.height) & (nx >= 0) & (nx < self.width)
                
                # Check if neighbor is unvisited 
                # Only update mazes that are active, have a non-empty stack, and haven't found an unvisited neighbor yet
                mask = active & (top_ptr > 0) & ~has_unvisited & valid

                valid_ny = torch.clamp(ny, 0, self.height-1)
                valid_nx = torch.clamp(nx, 0, self.width-1)
                
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
                if completed_mask.any():
                    newly_completed = completed_mask.sum().item()
                    active[completed_mask] = False
                    completed += newly_completed
                    pbar.update(newly_completed)
            
            iter_count += 1
        
        pbar.close()
        print(f"Completed {completed}/{self.batch_size} mazes in {iter_count} iterations")
        
        return mazes
    
    def save_mazes(self, mazes, output_dir="generated_mazes"):
        """
        Save generated mazes as images.
        
        Args:
            mazes (torch.Tensor): Batch of mazes to save
            output_dir (str): Directory to save maze images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Move to CPU for visualization
        mazes_cpu = mazes.cpu().numpy()
        
        for i in range(self.batch_size):
            plt.figure(figsize=(10, 10))
            plt.imshow(mazes_cpu[i], cmap='binary')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/maze_{i}.png", dpi=150)
            plt.close()


# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set maze parameters
    height, width = 15, 15  # Must be odd numbers for proper maze structure
    batch_size = 5
    
    # Create and use the generator
    generator = GPUMazeGenerator(height=height, width=width, batch_size=batch_size, device=device)
    mazes = generator.parallel_dfs()
    
    # Save a few sample mazes as images
    generator.save_mazes(mazes[:5], output_dir="sample_mazes")
    
