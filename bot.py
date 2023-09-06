from random import choice
from typing import List, Tuple

import numpy as np
from scipy import ndimage

from ...bot import Bot
from ...constants import Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    kernel[size//2, size//2] = 1
    gaussian = ndimage.gaussian_filter(kernel, sigma)
    return gaussian / gaussian[size//2, size//2]

def is_on_grid(pos: np.array, grid_size: Tuple[int, int]) -> bool:
    """
    Check if a position is still on the grid
    """
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


class FurryMuncher(Bot):
    """
    Smartness to be determined
    """
    def __init__(self, *args, **kwargs):
        super(FurryMuncher, self).__init__(*args, **kwargs)

        self.candy_kernel = gaussian_kernel(self.grid_size[0], self.grid_size[0]/5.0)
        self.snake_kernel = np.array([[1]])


    @property
    def name(self):
        return 'FurryMuncher'

    @property
    def contributor(self):
        return 'Ferry Schoenmakers'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        # Setup current grid
        grid = np.zeros(self.grid_size)
        # Add candies to grid
        for candy in candies:
            grid[candy[0], candy[1]] = 1
        # Convolve grid with candy kernel
        candy_conv_grid = ndimage.convolve(grid, self.candy_kernel, mode='constant', cval=0.0)

        # Add self to grid
        for part in snake:
            grid[part[0], part[1]] = -10
        # Add snakes to grid
        for other_snake in other_snakes:
            for part in other_snake:
                grid[part[0], part[1]] = -10
        snake_conv_grid = ndimage.convolve(grid, self.snake_kernel, mode='constant', cval=0.0)

        next_move = self.pick_hightest_move(candy_conv_grid + snake_conv_grid, snake[0])

        return next_move

    def pick_hightest_move(self, convolved_grid: np.ndarray, snake_head: np.array) -> Move:
        """
        Check highest rewarded move
        """
        # Highest priority, a move that is on the grid
        on_grid = [move for move in MOVE_VALUE_TO_DIRECTION
                   if is_on_grid(snake_head + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)]
        if not on_grid:
            return Move

        # Choose hightest rewarded move
        next_move = choice(on_grid)
        highest_score = -1

        for move in on_grid:
            idx = snake_head + MOVE_VALUE_TO_DIRECTION[move]
            score = convolved_grid[idx[0], idx[1]]
            if score > highest_score:
                next_move = move
                highest_score = score

        return next_move
