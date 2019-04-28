import numpy as np

class HMM(object):
    # Construct an HMM with the following set of variables
    #    numStates:          The size of the state space
    #    numOutputs:         The size of the output space
    #    trainStates[i][j]:  The jth element of the ith state sequence
    #    trainOutputs[i][j]: Similarly, for output
    def __init__(self, numStates, numOutputs, states, outputs, transition_p, measure_p, start_p):
        self.num_states = numStates
        self.num_outputs = numOutputs
        self.states = states # coordinates
        self.outputs = outputs # readings

        # Your code goes here
        # check to make sure all sequences are same length
        self.sequence_length = len(self.states)

        self.transition_p = transition_p
        self.measure_p = measure_p
        self.start_p = start_p

    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self):

        alphas = self._train_alpha()
        betas = self._train_beta()

        probs_given_seq = np.zeros((self.num_states, self.sequence_length))
        est_transition_probs = np.zeros((self.num_states, self.num_states, self.sequence_length))


            #likelihoods = np.zeros((self.num_subjects, self.num_states, 1))
            #for each_state in range(self.num_states):
                # sum probs over all steps, for each state
            #    likelihoods[each_state] =  - alphas[:][each_state].sum()

            #likelihoods = np.log(likelihoods)

    def _train_beta(self):
        betas = np.zeros((self.sequence_length,self.num_states))
        # base case
        for each_state in range(self.num_states):
            betas[-1][each_state] = 1
        # iterate backwards over sequences
        rev_idx = list(range(self.sequence_length - 1))
        rev_idx.reverse()
        for each_step in rev_idx:
            for each_state in range(self.num_states):
                rv = 0.0
                for each_inner_state in range(self.num_states):
                    rv_tmp =  betas[each_step + 1][each_inner_state]
                    next_obs = self.outputs[each_step + 1]
                    rv_tmp *= self.measure_p[next_obs][each_inner_state]
                    rv_tmp *= self.transition_p[each_state][each_inner_state]
                    rv += rv_tmp
                betas[each_step][each_state] = rv
        return betas


    def _train_alpha(self):
        alphas = np.zeros((self.sequence_length,self.num_states))
        for each_state in range(self.num_states):
            # calculate prob of each state given observation at time 0
            first_obs = self.outputs[0] # first observation
            rv = self.start_p[each_state] * self.measure_p[first_obs][each_state]
            alphas[0][each_state] = rv
        # normalize
        #alphas[0][:] = alphas[0][:] * (1.0/(alphas[0][:].sum()))


        for each_step in range(self.sequence_length - 1):
            for each_state in range(self.num_states):
                for each_inner_state in range(self.num_states):
                    rv_tmp = alphas[ each_step ][each_inner_state]
                    rv_tmp *= self.transition_p[each_state][each_inner_state]
                    rv += rv_tmp
                next_obs = self.outputs[each_step + 1]
                rv *= self.measure_p[next_obs][each_state]
                alphas[each_step][each_state] = rv
        # normalize
        #for each_step in range(self.sequence_length - 1):
        #    normer = (alphas[each_step + 1][:].sum())
        #    alphas[each_step + 1][:] = alphas[each_step + 1][:] * normer
        return alphas




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
