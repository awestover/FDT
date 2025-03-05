import torch
import numpy as np
from tqdm import tqdm

def generate_maze(height, width, batch_size=100, difficulty=0.5, device='cuda'):
    """
    Generate mazes using a GPU-based randomized DFS approach.
    
    Args:
        height (int): Height of the maze (must be odd)
        width (int): Width of the maze (must be odd)
        batch_size (int): Number of mazes to generate in parallel
        difficulty (float): Value between 0 and 1 controlling maze complexity.
                           Higher values create more difficult mazes with fewer shortcuts.
                           Lower values "drill holes" through walls to create shortcuts.
        device (str): Device to run on ('cuda' or 'cpu')
    
    Returns:
        torch.Tensor: Batch of generated mazes
    """
    assert height % 2 == 1, "Height must be an odd number"
    assert width % 2 == 1, "Width must be an odd number"
    assert 0 <= difficulty <= 1, "Difficulty must be between 0 and 1"
    
    # For directions
    directions = torch.tensor([
        [-1, 0],  # North
        [0, 1],   # East
        [1, 0],   # South
        [0, -1]   # West
    ], device=device)

    # Initialize mazes with all walls
    mazes = torch.zeros((batch_size, height, width), 
                      dtype=torch.uint8, device=device)
    
    # Initialize visit state tensor
    # 0 = unvisited, 1 = in stack, 2 = visited and backtracked
    visit_state = torch.zeros((batch_size, height, width), 
                            dtype=torch.uint8, device=device)
    
    height_cells = (height - 1) // 2
    width_cells = (width - 1) // 2
    y_coords = torch.arange(height_cells, device=device) * 2 + 1
    x_coords = torch.arange(width_cells, device=device) * 2 + 1
    
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
    max_stack_size = (height // 2) * (width // 2)  # Maximum possible stack size
    stack = torch.zeros((batch_size, max_stack_size, 2), dtype=torch.int32, device=device)
    top_ptr = torch.ones(batch_size, dtype=torch.int32, device=device)  # All stacks start with 1 element
    
    # Initialize stacks with starting positions using fancy indexing
    # Convert to int32 to match stack tensor type
    stack[batch_indices, 0, 0] = start_y.to(torch.int32)
    stack[batch_indices, 0, 1] = start_x.to(torch.int32)
    
    # Track active mazes
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # Setup for reporting progress
    iter_count = 0
    pbar = tqdm(total=batch_size, desc="Generating mazes")
    completed = 0
    
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
            valid = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            
            # Check if neighbor is unvisited 
            # Only update mazes that are active, have a non-empty stack, and haven't found an unvisited neighbor yet
            mask = active & (top_ptr > 0) & ~has_unvisited & valid

            valid_ny = torch.clamp(ny, 0, height-1)
            valid_nx = torch.clamp(nx, 0, width-1)
            
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
    print(f"Completed {completed}/{batch_size} mazes in {iter_count} iterations")
    
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


# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set maze parameters
    height, width = 15, 15  # Must be odd numbers for proper maze structure
    batch_size = 5
    difficulty = 0.95  # Higher value = fewer shortcuts, more difficult maze
    
    # Generate mazes
    mazes = generate_maze(height=height, width=width, batch_size=batch_size, 
                          difficulty=difficulty, device=device)
    import matplotlib.pyplot as plt
    plt.imshow(mazes[0].cpu().numpy(), cmap='binary')
    plt.show()


