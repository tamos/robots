
import numpy as np

class Grid:

    def __init__(self, num_col, num_row, valid_locations, dataset):

        self.world = np.full((num_col + 1, num_row + 1),
                    False, dtype = np.bool)
        self.location_to_state_mapping = dataset.
        self.state_to_location_mapping = {}

        state_no = 0

        for each_location in valid_locations:
            self.add_location(*each_location)
            self.location_to_state_mapping[each_location] = state_no
            self.state_to_location_mapping[state_no] = each_location
            state_no += 1

    def add_location(self, x,y):
        self.world[x][y] = True

    def state_number_from_xy(self, x,y):
        return self.location_to_state_mapping[(x,y)]

    def xy_from_state_number(self, state_no):
        return self.state_to_location_mapping[state_no]

    def neighbours(self, state_no):
        x,y = self.xy_from_state_number(state_no)
        # there are at most 4 options to travel
        neighbour_states = []
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if all([self.world[i][j], x != i, y != j]): # valid neighbour?
                    neighbour_states.append(self.state_number_from_xy(i,j))
        return len(neighbour_states), neighbour_states
