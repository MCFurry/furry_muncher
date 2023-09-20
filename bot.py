from random import choice
from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from ...bot import Bot
from ...constants import Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake

head_positions = []  # List to store previous head positions
max_head_positions = 18

def is_on_grid(pos: np.array, grid_size: Tuple[int, int]) -> bool:
    """
    Check if a position is still on the grid
    """
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]

def collides(head, snakes):
    for snake in snakes:
        for segment in snake:
            if np.array_equal(head, segment):
                return True
    return False

def create_heatmap(grid_size, coords, candy_positions):
    distances = cdist(coords, candy_positions, metric='euclidean')
    min_distances = distances.min(axis=1).reshape(grid_size)

    return min_distances


class FurryMuncher(Bot):
    """
    Smartness to be determined
    """
    def __init__(self, *args, **kwargs):
        super(FurryMuncher, self).__init__(*args, **kwargs)

        x, y = np.indices(self.grid_size)
        self.xy_coordinates = np.column_stack((x.ravel(), y.ravel()))

    @property
    def name(self):
        return 'FurryMuncher'

    @property
    def contributor(self):
        return 'Ferry Schoenmakers'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        # Calculate heatmap for candies
        heatmap = create_heatmap(self.grid_size, self.xy_coordinates, candies)

        # Add self to heatmap
        for part in snake:
            heatmap[part[0], part[1]] = self.grid_size[0]*2

        # Add snakes to heatmap
        for other_snake in other_snakes:
            for part in other_snake:
                heatmap[part[0], part[1]] = self.grid_size[0]*2

        next_move = self.pick_best_move(heatmap, snake[0])

        # Update the list of head positions
        head_positions.append(snake[0] + MOVE_VALUE_TO_DIRECTION[next_move])

        # Check for deadlock situations
        if len(head_positions) > max_head_positions:
            head_positions.pop(0)
            _, counts = np.unique(head_positions, axis=0, return_counts=True)
            if (counts > 3).any():
                random_move = self.randommove(snake, other_snakes, candies)
                head_positions[-1] = snake[0] + MOVE_VALUE_TO_DIRECTION[random_move]
                return random_move

        return next_move
    
    def randommove(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        all_snakes = [snake] + other_snakes
        collision_free = [move for move, direction in MOVE_VALUE_TO_DIRECTION.items()
                          if is_on_grid(snake[0] + direction, self.grid_size)
                          and not collides(snake[0] + direction, all_snakes)]
        if collision_free:
            return choice(collision_free)
        else:
            return choice(list(Move))

    def pick_best_move(self, convolved_grid: np.ndarray, snake_head: np.array) -> Move:
        """
        Check lowest cost move
        """
        # Highest priority, a move that is on the grid
        on_grid = [move for move in MOVE_VALUE_TO_DIRECTION
                   if is_on_grid(snake_head + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)]
        if not on_grid:
            return Move

        # Choose move with lowest cost
        next_move = choice(on_grid)
        min_cost = float('inf')

        for move in on_grid:
            idx = snake_head + MOVE_VALUE_TO_DIRECTION[move]
            cost = convolved_grid[idx[0], idx[1]]
            if cost < min_cost:
                next_move = move
                min_cost = cost

        return next_move
