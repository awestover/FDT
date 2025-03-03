import numpy as np


def generate_maze(size=16, branch_probability=0.5):
    """
    Generate a maze grid by carving paths through waypoints and adding branchy, curvy dead ends.
    Returns a matrix where:
      0 = path
      1 = wall
    """
    num_dead_ends = int(size**1.5)
    dead_end_length = size // 2
    num_paths = int(size**0.5)

    # Use an integer grid for speed.
    grid = np.ones((size, size), dtype=np.int8)

    def generate_waypoints():
        """Generate a sequence of waypoints moving toward the goal using NumPy's randomness."""
        waypoints = [(0, 0)]
        current = (0, 0)
        goal = (size - 1, size - 1)
        while current != goal:
            x, y = current
            gx, gy = goal

            # Define valid range for next waypoint
            x_min = x
            x_max = min(x + max(1, (gx - x) // 2), size - 1)
            y_min = y
            y_max = min(y + max(1, (gy - y) // 2), size - 1)

            # Use np.random.randint which is efficient and vectorized
            next_x = np.random.randint(x_min, x_max + 1)
            next_y = np.random.randint(y_min, y_max + 1)

            # If stuck, force movement toward the goal
            if (next_x, next_y) == current:
                if gx > x:
                    next_x = x + 1
                if gy > y:
                    next_y = y + 1

            current = (next_x, next_y)
            waypoints.append(current)

            # If near goal, directly connect
            if abs(gx - next_x) <= 1 and abs(gy - next_y) <= 1:
                waypoints.append(goal)
                break
        return waypoints

    def carve_path_between_points(start, end):
        """Carve a path between two points with some random variation using NumPy's random functions."""
        x, y = start
        end_x, end_y = end
        path_cells = [(x, y)]

        while (x, y) != (end_x, end_y):
            grid[x, y] = 0

            # Determine primary direction(s) toward the end
            possible_moves = []
            if x < end_x:
                possible_moves.append((1, 0))
            elif x > end_x:
                possible_moves.append((-1, 0))
            if y < end_y:
                possible_moves.append((0, 1))
            elif y > end_y:
                possible_moves.append((0, -1))

            # Occasionally add perpendicular moves for variation
            if np.random.rand() < 0.3 and len(possible_moves) == 1:
                if possible_moves[0][0] == 0:  # moving vertically
                    if x < size - 1:
                        possible_moves.append((1, 0))
                    if x > 0:
                        possible_moves.append((-1, 0))
                else:  # moving horizontally
                    if y < size - 1:
                        possible_moves.append((0, 1))
                    if y > 0:
                        possible_moves.append((0, -1))

            # Choose a move using np.random for speed
            dx, dy = possible_moves[np.random.randint(0, len(possible_moves))]
            new_x, new_y = x + dx, y + dy

            # Ensure we stay within bounds
            if 0 <= new_x < size and 0 <= new_y < size:
                x, y = new_x, new_y
                path_cells.append((x, y))

        return path_cells

    def add_branchy_dead_end(start_point, max_length=None, is_branch=False):
        """
        Add a curvy, potentially branching dead end starting from the given point.
        Uses NumPy's randomness to shuffle directions and decide on curviness.
        """
        if max_length is None:
            max_length = np.random.randint(3, dead_end_length + 1)

        x, y = start_point
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # Randomize directions using permutation
        directions = [directions[i] for i in np.random.permutation(len(directions))]

        for main_dir in directions:
            dx, dy = main_dir
            current_length = 0
            cx, cy = x, y

            while current_length < max_length:
                # Occasionally change direction for curviness
                if np.random.rand() < 0.3 and current_length > 0:
                    if dx == 0:  # vertical movement, so choose horizontal alternatives
                        new_dirs = [(1, 0), (-1, 0)]
                    else:
                        new_dirs = [(0, 1), (0, -1)]
                    new_dirs = [
                        new_dirs[i] for i in np.random.permutation(len(new_dirs))
                    ]
                    dx, dy = new_dirs[0]

                new_x, new_y = cx + dx, cy + dy

                # Check boundaries and ensure the new cell is a wall and not connecting to too many paths
                if (
                    0 <= new_x < size
                    and 0 <= new_y < size
                    and grid[new_x, new_y] == 1
                    and sum(
                        grid[new_x + d[0], new_y + d[1]] == 0
                        for d in directions
                        if 0 <= new_x + d[0] < size and 0 <= new_y + d[1] < size
                    )
                    <= 1
                ):

                    grid[new_x, new_y] = 0
                    cx, cy = new_x, new_y
                    current_length += 1

                    # Occasionally add a branch off the current dead end (only on main dead ends)
                    if (
                        not is_branch
                        and np.random.rand() < branch_probability
                        and current_length > 2
                    ):
                        branch_max_length = max(2, max_length - current_length - 1)
                        add_branchy_dead_end(
                            (cx, cy), branch_max_length, is_branch=True
                        )
                else:
                    break

            if current_length > 0:
                return True  # Successful dead end carved

        return False

    # Generate main paths using waypoints
    all_path_cells = []
    for _ in range(num_paths):
        waypoints = generate_waypoints()
        path_cells = []
        for i in range(len(waypoints) - 1):
            path_cells.extend(carve_path_between_points(waypoints[i], waypoints[i + 1]))
        all_path_cells.extend(path_cells)

    # Add branchy dead ends (with a cap on attempts)
    dead_ends_added = 0
    attempts = 0
    while dead_ends_added < num_dead_ends and attempts < num_dead_ends * 3:
        start_point = all_path_cells[np.random.randint(0, len(all_path_cells))]
        if add_branchy_dead_end(start_point):
            dead_ends_added += 1
        attempts += 1

    # Ensure the start and goal remain open
    grid[0, 0] = 0
    grid[size - 1, size - 1] = 0

    return grid


def generate_simple_maze(size=16):
    grid = np.zeros((size, size), dtype=np.int8)
    holes = np.random.randint(0, size - 1, size // 2)
    for col in range(size // 2):
        grid[:, 2 * col + 1] = 1
        grid[holes[col], 2 * col + 1] = 0
    grid[-1, -1] = 0
    return grid


def generate_easy_maze(size=16, difficulty=0.5, buffer=None):
    """
    Generate a simple maze with controlled difficulty - optimized version.

    Args:
        size (int): Size of the maze (size x size grid)
        difficulty (float): Value between 0.0 and 1.0 controlling maze difficulty
                           - 0.0: Very easy (many holes in walls)
                           - 1.0: Very difficult (few holes in walls)

    Returns:
        np.ndarray: Grid representing the maze
    """
    # Create empty grid
    if buffer is None:
        grid = np.zeros((size, size), dtype=np.int8)
    else:
        grid = buffer

    # Place walls in odd-numbered columns
    for col in range(1, size, 2):
        grid[:, col] = 1

    # Calculate number of holes based on difficulty
    # Clamp difficulty between 0 and 1
    difficulty = max(0.0, min(1.0, difficulty))

    # Calculate holes per wall: inverse relationship with difficulty
    # At difficulty 0.0: ~75% of wall cells are holes
    # At difficulty 1.0: ~10% of wall cells are holes
    hole_percentage = 0.75 - 0.65 * difficulty
    holes_per_wall = max(1, int(hole_percentage * size))

    # Create holes in each wall column
    for col in range(1, size, 2):
        # Select random positions for holes
        hole_positions = np.random.choice(size, holes_per_wall, replace=False)
        grid[hole_positions, col] = 0

    # Ensure goal is accessible
    grid[-1, -1] = 0

    # If there's a wall in the second-to-last column, ensure path to goal
    if size % 2 == 0:  # When size is even, second-to-last column has a wall
        grid[-1, -2] = 0

    return grid


if __name__ == "__main__":
    #  maze = generate_maze(16)
    maze = generate_easy_maze(8, 0.4)

    print(maze)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap="binary")
    plt.title("Maze with Branchy Dead Ends")
    plt.axis("off")
    plt.show()
