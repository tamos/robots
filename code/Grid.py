
import numpy as np

class Grid:

    def __init__(self, num_col, num_row):

        self.world = np.full((num_col, num_row), False, dtype = np.bool)

    def add_location(self, x,y):
        # use 1-index coordinates
        self.world[x - 1][y - 1] = True
