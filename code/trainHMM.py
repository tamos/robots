# train hmm model
from DataSet import DataSet
from Viterbi import  Viterbi
from HMM import HMM
from sys import argv
from Grid import Grid

VALID_LOCATIONS = [(2,1), (3,1), (4,1), (1,2), (3,2), (4,2), (1,3), (2,3),
                    (3,3),(2,4),(3,4),(4,4)]




if __name__ == "__main__":

    grid = Grid(4,4)

    for each_location in VALID_LOCATIONS:
        grid.add_location(*each_location)

    dataset = DataSet(argv[1])
    coords, states = dataset.read_file()
