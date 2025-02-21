import numpy as np
import random

def generate_maze(size=32, num_paths=3, num_dead_ends=100, dead_end_length=8):
    """
    Generate a maze grid by carving paths through waypoints and adding dead ends.
    Returns a matrix where:
    0 = path
    1 = wall
    """
    # Start with all walls
    grid = np.ones((size, size))
    
    def generate_waypoints():
        """Generate a sequence of waypoints moving toward the goal"""
        waypoints = [(0, 0)]  # Start point
        current = (0, 0)
        goal = (size-1, size-1)
        
        while current != goal:
            x, y = current
            gx, gy = goal
            
            # Determine valid range for next waypoint
            x_min = x
            x_max = min(x + max(1, (gx - x) // 2), size-1)
            y_min = y
            y_max = min(y + max(1, (gy - y) // 2), size-1)
            
            # Add some randomness to waypoint placement
            next_x = random.randint(x_min, x_max)
            next_y = random.randint(y_min, y_max)
            
            # If we're stuck, force movement toward goal
            if (next_x, next_y) == current:
                if gx > x:
                    next_x = x + 1
                if gy > y:
                    next_y = y + 1
            
            current = (next_x, next_y)
            waypoints.append(current)
            
            # If we're close to goal, connect directly
            if abs(gx - next_x) <= 1 and abs(gy - next_y) <= 1:
                waypoints.append(goal)
                break
                
        return waypoints
    
    def carve_path_between_points(start, end):
        """Carve a path between two points with some random variation"""
        x, y = start
        end_x, end_y = end
        path_cells = [(x, y)]
        
        while (x, y) != (end_x, end_y):
            grid[x, y] = 0
            
            # Determine primary direction(s) to move
            possible_moves = []
            if x < end_x:
                possible_moves.append((1, 0))
            elif x > end_x:
                possible_moves.append((-1, 0))
            if y < end_y:
                possible_moves.append((0, 1))
            elif y > end_y:
                possible_moves.append((0, -1))
                
            # Sometimes add perpendicular moves for variation
            if random.random() < 0.3 and len(possible_moves) == 1:
                if possible_moves[0][0] == 0:  # If moving vertically
                    if x < size-1:
                        possible_moves.append((1, 0))
                    if x > 0:
                        possible_moves.append((-1, 0))
                else:  # If moving horizontally
                    if y < size-1:
                        possible_moves.append((0, 1))
                    if y > 0:
                        possible_moves.append((0, -1))
            
            # Choose and make move
            dx, dy = random.choice(possible_moves)
            new_x, new_y = x + dx, y + dy
            
            # Stay within bounds
            if 0 <= new_x < size and 0 <= new_y < size:
                x, y = new_x, new_y
                path_cells.append((x, y))
        
        return path_cells
    
    def add_dead_end(start_point):
        """Add a dead end branch starting from the given point"""
        x, y = start_point
        length = random.randint(2, dead_end_length)
        
        # Choose a random direction for the dead end
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            # Try to create dead end in this direction
            current_length = 0
            cx, cy = x, y
            
            while current_length < length:
                new_x, new_y = cx + dx, cy + dy
                
                # Check if we can continue in this direction
                if (0 <= new_x < size and 0 <= new_y < size and 
                    grid[new_x, new_y] == 1 and
                    # Check surrounding cells to avoid connecting to other paths
                    sum(grid[new_x + d[0], new_y + d[1]] == 0 
                        for d in directions 
                        if 0 <= new_x + d[0] < size and 0 <= new_y + d[1] < size) <= 1):
                    
                    grid[new_x, new_y] = 0
                    cx, cy = new_x, new_y
                    current_length += 1
                else:
                    break
            
            if current_length > 0:  # If we successfully created a dead end
                return
    
    # Generate main paths
    all_path_cells = []
    for _ in range(num_paths):
        waypoints = generate_waypoints()
        path_cells = []
        
        # Carve paths between consecutive waypoints
        for i in range(len(waypoints) - 1):
            path_cells.extend(carve_path_between_points(waypoints[i], waypoints[i+1]))
        
        all_path_cells.extend(path_cells)
    
    # Add dead ends
    dead_ends_added = 0
    attempts = 0
    while dead_ends_added < num_dead_ends and attempts < num_dead_ends * 3:
        # Pick a random point from the existing paths
        start_point = random.choice(all_path_cells)
        add_dead_end(start_point)
        dead_ends_added += 1
        attempts += 1
    
    # Ensure start and end are open
    grid[0, 0] = 0
    grid[size-1, size-1] = 0
    
    return grid

# Example usage
if __name__ == "__main__":
    maze = generate_maze()
    import matplotlib.pyplot as plt
    plt.imshow(maze)
    plt.show()
