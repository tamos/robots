import sys
from DataSet import *
from HMM import *
from Viterbi import *
from trainHMM import train_model
import numpy as np

N_ITER = 50

class RunViterbi(object):

    def __init__(self):
        self.maxSequence = []
        self.realStates = []


    def trainHMM(self, filename):
        print "Reading training data from %s" % (filename)

        # Read in the training data from the file
        dataset = DataSet(filename)
        states, obs = dataset.read_file()

        # Instatiate and train the HMM
        self.hmm, ll = train_model(dataset, 1e-5)

    def estMaxSequence(self, filename):

        print "Reading testing data from %s" % (filename)

        # Read in the testing dta from the file
        dataset = DataSet(filename)
        states, obs = dataset.read_file()

        # Run Viterbi to estimate most likely sequence
        viterbi = Viterbi(self.hmm)
        for idx in range(len(obs)):
            self.maxSequence.append(viterbi.mostLikelySequence(obs[idx]))
            self.realStates.append(states[idx])
            print "Sequences done", idx


if __name__ == '__main__':

    # This function should be called with two arguments: trainingdata.txt testingdata.txt
    # eg python code/RunViterbi.py data/randomwalk.train.txt data/randomwalk.test.txt
    viterbi = RunViterbi()
    viterbi.trainHMM(sys.argv[1])
    viterbi.estMaxSequence(sys.argv[2])
    right = 0
    wrong = 0
    for i in zip(viterbi.maxSequence, viterbi.realStates):
        for prediction, reality in zip(i[0],i[1]):
            print prediction, reality
            if prediction == reality:
                right += 1
            else:
                wrong += 1
    acc = float(right)/float(right + wrong)
    print "Percent correct: ", acc * 100
