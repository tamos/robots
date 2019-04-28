import numpy as np

INITIAL_STAY_PROB = 0.2 # probability of staying in current location
TOTAL_LEAVE_PROB = 1 - INITIAL_STAY_PROB # probability of leaving current location
CAMERA_ACC_PROB = 0.7 # probability of camera being right
TOTAL_CAMERA_ERR_PROB = 1 - CAMERA_ACC_PROB # probability of camera wrong


class HMM(object):
    # Construct an HMM with the following set of variables
    #    numStates:          The size of the state space
    #    numOutputs:         The size of the output space
    #    trainStates[i][j]:  The jth element of the ith state sequence
    #    trainOutputs[i][j]: Similarly, for output
    def __init__(self, numStates, numOutputs, states, outputs):
        self.numStates = numStates
        self.numOutputs = numOutputs
        self.states = states # tuples of shape
        self.outputs = outputs


        # Your code goes here
        print "Please add code"


    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self):

        # Your code goes here
        print "Please add code"


    # Returns the log probability associated with a transition from
    # the dummy start state to the given state according to this HMM
    def getLogStartProb (state):
        return np.log(INITIAL_STAY_PROB)

    # Returns the log probability associated with a transition from
    # fromState to toState according to this HMM
    def getLogTransProb (fromState, toState):

        # Your code goes here
        print "Please add code"

    # Returns the log probability of state state emitting output
    # output
    def getLogOutputProb (state, output):

        # Your code goes here
        print "Please add code"
