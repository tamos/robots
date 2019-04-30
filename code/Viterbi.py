# This is a template for a Vitirbi class that can be used to compute
# most likely sequences.
import numpy as np

ARBITRARY_OFFSET = 1e-10

class Viterbi(object):
    # Construct an instance of the viterbi class
    # based upon an instantiatin of the HMM class
    def __init__(self, hmm):
        self.hmm = hmm
    # Returns the most likely state sequence for a given output
    # (observation) sequence, i.e.,
    #    arg max_{X_1, X_2, ..., X_T} P(X_1,...,X_T | Z_1,...Z_T)
    # according to the HMM model that was passed to the constructor.
    def mostLikelySequence(self, output):
        deltas = np.zeros((self.hmm.sequence_length, self.hmm.num_states))
        Pre = np.zeros((self.hmm.sequence_length, self.hmm.num_states))

        #deltas[0,:] = self.hmm.start_p * self.hmm.measure_p[output[0],:]
        deltas[0,:] = (self.hmm.start_p * self.hmm.measure_p[output[0],:].reshape(self.hmm.start_p.shape)).reshape(deltas[0,:].shape)
        for each_step in range(1, self.hmm.sequence_length):
            candidate_vals = np.zeros((self.hmm.num_states,1)) *  ARBITRARY_OFFSET
            for i in range(self.hmm.num_states):
                for j in range(self.hmm.num_states):
                    tval = self.hmm.transition_p[j,i] * deltas[each_step - 1,j]
                    candidate_vals[j] = tval
                candidate_vals /= candidate_vals.sum() + ARBITRARY_OFFSET
                Pre[each_step,i] = np.argmax(candidate_vals)
            deltas[each_step,:] = self.hmm.measure_p[output[each_step],:] * candidate_vals.max()
            #deltas[each_step,:] /= deltas[each_step,:].sum() + ARBITRARY_OFFSET
        # get most likely terminal state
        most_likely_sequence = [None] * self.hmm.sequence_length
        most_likely_sequence[-1] = np.argmax(deltas[-1,:]) # b/c indices are the state numbers

        for each_step in range(2, self.hmm.sequence_length):
            idx = self.hmm.sequence_length - each_step
            most_likely_sequence[idx] = Pre[idx, np.argmax(deltas[idx,:])]
        return np.array(most_likely_sequence)
