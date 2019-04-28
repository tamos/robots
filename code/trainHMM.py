# train hmm model
from DataSet import DataSet
from Viterbi import  Viterbi
from HMM import HMM
from sys import argv
import numpy as np


INITIAL_STAY_PROB = 0.2 # probability of staying in current location
TOTAL_LEAVE_PROB = 1.0 - INITIAL_STAY_PROB # probability of leaving current location
CAMERA_ACC_PROB = 0.7 # probability of camera being right
TOTAL_CAMERA_ERR_PROB = 1.0 - CAMERA_ACC_PROB # probability of camera wrong



VALID_LOCATIONS = [(2,1), (3,1), (4,1), (1,2), (3,2), (4,2), (1,3), (2,3),
                    (3,3),(2,4),(3,4),(4,4)]

INVALID_LOCATIONS = [(1,4),(1,1),(2,2),(4,3)]

NEIGHBOURS = {1: {2: [(1,3)], 3: [(1,2),(2,3)]},
            2: {1: [(3,1)], 3:[(1,3),(3,3),(2,4)],4: [(2,3),(3,4)]},
            3: {1: [(2,1),(4,1),(3,2)], 2: [(3,1),(4,2)], 3: [(2,3),(3,2),(3,4)],
                    4:[(3,3),(2,4),(4,4)]},
            4: {1: [(3,1),(4,2)], 2: [(4,1),(3,2)], 4: [(3,4)]} }


def go(dataset):
    states, outputs = dataset.read_file()
    states = states[0]
    outputs = outputs[0]
    num_states = dataset.xyToInt.ravel().shape[0]
    tmp_mat = np.identity(num_states + 1)
    tmp_mat *= INITIAL_STAY_PROB
    for each_loc in VALID_LOCATIONS:
        int_repres = int(dataset.xyToInt[each_loc[0] - 1,each_loc[1] - 1])
        # distribute probs to neighbours
        neighbours = NEIGHBOURS[each_loc[0]][each_loc[1]]
        num_neighbours = len(neighbours)
        for each_neighbour in neighbours:
            int_repres_neigh = int(dataset.xyToInt[each_neighbour[0] - 1,each_neighbour[1] - 1])
            tmp_mat[int_repres, int_repres_neigh] += (1 - INITIAL_STAY_PROB) * (1.0/num_neighbours)
    num_outputs = len(dataset.obsToInt.keys())

    # make matrix of remaining probs

    other_probs = (np.ones((num_states, num_states)) * ((1 - CAMERA_ACC_PROB) / 3))

    # remove diagonals
    other_probs -= (np.identity(num_states) * ((1 - CAMERA_ACC_PROB) / 3))
    measure_p = (np.identity(num_states)  * CAMERA_ACC_PROB) + other_probs

    start_p = np.ones((num_states,1)) / num_states

    model = HMM(num_states, num_outputs, states, outputs, tmp_mat, measure_p, start_p)
    model.train()



if __name__ == "__main__":

    dataset = DataSet(argv[1])
    go(dataset)
