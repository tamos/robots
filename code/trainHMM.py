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

ACTUAL_COLOURS = { 2: {1: 'g', 3: 'b', 4: 'r'},
                    1: {2: 'r', 3: 'g'},
                    3: {1: 'y', 2: 'g', 3: 'r', 4: 'y'},
                    4: {1: 'b', 2: 'y', 4: 'b'}}
N_ITER = 50

def go(dataset, threshold):

    ### Set up ###
    states, outputs = dataset.read_file()
    num_states = dataset.xyToInt.ravel().shape[0]
    num_outputs = len(dataset.obsToInt.keys())

    measure_p = np.zeros((num_outputs, num_states))
    start_p = np.ones((num_states,1)) * 1.0/16.0
    # make the matrix of transition probs
    trans_p = np.identity(num_states)
    trans_p *= INITIAL_STAY_PROB
    for each_loc in VALID_LOCATIONS:
        int_repres = int(dataset.xyToInt[each_loc[0] - 1,each_loc[1] - 1])

        # distribute probs to neighbours
        neighbours = NEIGHBOURS[each_loc[0]][each_loc[1]]
        num_neighbours = float(len(neighbours))

        # do measurement probs
        int_col_repres = int(dataset.obsToInt[ACTUAL_COLOURS[each_loc[0]][each_loc[1]]])
        measure_p[:,int_repres] = TOTAL_CAMERA_ERR_PROB / 3.0
        measure_p[int_col_repres,int_repres] = CAMERA_ACC_PROB

        for each_neighbour in neighbours:
            int_repres_neigh = int(dataset.xyToInt[each_neighbour[0] - 1 ,each_neighbour[1] - 1])
            trans_p[int_repres, int_repres_neigh] += (1.0 - INITIAL_STAY_PROB) * (1.0/num_neighbours)

    ### Model training ###

    llikes = []
    ll_old = 10e10

    for _ in range(N_ITER):
        model = HMM(num_states, num_outputs, outputs, trans_p,  measure_p, start_p)
        trans_p, measure_p, start_p , ll = model.train()
        print "Log Likely is ", ll, np.exp(ll) * 100
        llikes.append(ll)
        if abs(ll_old - ll) < threshold:
            print "Threshold change reached, stopping"
            break
        ll_old = ll
    return llikes


if __name__ == "__main__":

    dataset = DataSet(argv[1])
    loglikes = go(dataset, 1e-5)
    with open("llikes.csv", 'w') as f:
        for line in loglikes:
            f.write(str(line) + "\n")
