import numpy as np
from itertools import product


ARBITRARY_OFFSET = 1e-10

class HMM(object):
    def __init__(self, numStates, numOutputs, outputs, transition_p, measure_p, start_p):
        ''' This is based largely on https://cran.r-project.org/web/packages/seqHMM/vignettes/seqHMM_algorithms.pdf
        where pi = start_p, b = measure_p, and a = transition_p
        and I reviewed the wikipedia article on Baum-Welch to better understand the algorithm
        '''
        self.num_states = numStates
        self.num_outputs = numOutputs
        self.num_sequences = len(outputs)
        self.outputs = outputs # readings
        self.sequence_length = len(outputs[0]) # assume all same length
        self.transition_p = transition_p
        self.measure_p = measure_p
        self.start_p = start_p

    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data

    def train(self):
        probs_given_seq = np.zeros((self.num_sequences, self.num_states, self.sequence_length))
        est_transition_probs = np.zeros((self.num_sequences, self.num_states, self.num_states, self.sequence_length))
        log_likes = np.zeros((self.num_sequences,1))

        idx = 0
        for each_walk in self.outputs:
            probs_given_seq[idx,:,:], est_transition_probs[idx,:,:,:], log_likes[idx] = self.train_single_sequence(each_walk)
            idx += 1

        probs_given_seq = probs_given_seq.sum(axis = 0)
        est_transition_probs = est_transition_probs.sum(axis = 0)

        new_start_p = np.zeros(self.start_p.shape)
        new_measure_p = np.zeros((self.num_sequences, self.measure_p.shape[0],
                                self.measure_p.shape[1]))
        new_transition_p = np.zeros(self.transition_p.shape)

        # now set the new params
        for i in range(self.num_states):
            # start probs
            new_start_p[i] = probs_given_seq[i][0]

            # transition probs
            for j in range(self.num_states):
                denom =  probs_given_seq[i][:-1].sum()
                numer = est_transition_probs[i][j][:-1].sum()
                new_transition_p[i][j] =  float(numer) / float(denom + ARBITRARY_OFFSET)

            # measurement probs
            for each_walk in range(self.num_sequences):
                this_walk = np.array(self.outputs[each_walk])
                for z in range(self.num_outputs):
                    relevant_probs = probs_given_seq[i][:]
                    denom = relevant_probs.sum()
                    numer = relevant_probs[this_walk == z].sum()
                    new_measure_p[each_walk,z,i] = float(numer) / float(denom + ARBITRARY_OFFSET)

        new_measure_p = new_measure_p.sum(axis = 0) # get rid of walk axis

        # last, normalize each summed matrix
        # ref https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose
        new_transition_p /= new_transition_p.sum(axis=0) + ARBITRARY_OFFSET # row normalize
        new_measure_p /= new_measure_p.sum(axis = 1).reshape(new_measure_p.shape[0],1) + ARBITRARY_OFFSET # column normalize
        new_start_p /= new_start_p.sum()  + ARBITRARY_OFFSET # plain normalize

        np.nan_to_num(new_transition_p, copy = False)
        np.nan_to_num(new_measure_p, copy = False)

        self.transition_p =  new_transition_p
        self.measure_p = new_measure_p
        self.start_p = new_start_p
        return log_likes.sum()

    def train_single_sequence(self, output):

        alphas, normalizers = self._train_alpha(output)
        betas = self._train_beta(output, normalizers)

        likeli = normalizers.sum()

        gamma = np.zeros((self.num_states, self.sequence_length))
        xi = np.zeros((self.num_states, self.num_states, self.sequence_length))

        gamma = (alphas * betas).T
        gamma /= gamma.sum(axis = 1).reshape(gamma.shape[0],1) + ARBITRARY_OFFSET

        for t in range(self.sequence_length):
            for i,j in product(range(self.num_states),range(self.num_states)):
                alpha_ti = float(alphas[t][i])
                a_ij = float(self.transition_p[i][j])
                if t == self.sequence_length - 1:
                    beta_tplusonej = 1.0
                    b_jyplusone = 1.0
                else:
                    beta_tplusonej = float(betas[t + 1][j])
                    b_jyplusone = float(self.measure_p[output[t + 1]][j])
                xi[i][j][t] = alpha_ti * a_ij * beta_tplusonej * b_jyplusone

        xi /= xi.sum(axis=2)[:,:,np.newaxis] + ARBITRARY_OFFSET
        return gamma, xi, likeli

    def _train_beta(self, outputs, normalizers):
        betas = np.zeros((self.sequence_length,self.num_states))
        # base case
        betas[-1,:] = normalizers[-1]

        # recursive case
        # iterate backwards over sequences
        rev_idx = list(range(self.sequence_length - 1))
        rev_idx.reverse()
        for each_step in rev_idx:
            previous_obs = outputs[each_step + 1]
            this_obs = outputs[each_step]
            rv = (self.transition_p * self.measure_p[previous_obs] * betas[each_step + 1,:]).sum()
            betas[each_step,:] = rv
        return betas * normalizers


    def _train_alpha(self, outputs):
        alphas = np.zeros((self.sequence_length, self.num_states))
        normalization_c = np.zeros((self.sequence_length,1))

        # base case
        first_obs = outputs[0] # first observation
        alphas[0,:] = (self.start_p * self.measure_p[first_obs,:].reshape(self.start_p.shape)).reshape(alphas[0,:].shape)
        normalization_c[0] = 1.0 / alphas[0,:].sum()
        alphas[0,:] = alphas[0,:] * normalization_c[0]

        # recursive case

        for each_step in range(self.sequence_length - 1):
            next_obs = outputs[each_step + 1]
            this_obs = outputs[each_step]
            rv = (alphas[each_step,:] * self.transition_p).sum()
            rv *= self.measure_p[next_obs,:]
            alphas[each_step + 1,:] = rv
            normalization_c[each_step + 1] = 1.0 / alphas[each_step + 1].sum()
            alphas[each_step + 1,:] *= normalization_c[each_step + 1]
        return alphas, normalization_c
