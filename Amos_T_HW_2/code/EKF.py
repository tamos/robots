import numpy as np

import math

class EKF(object):
    # Construct an EKF instance with the following set of variables
    #    mu:                 The initial mean vector
    #    Sigma:              The initial covariance matrix
    #    R:                  The process noise covariance
    #    Q:                  The measurement noise covariance
    def __init__(self, mu, Sigma, R, Q):
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

    def getMean(self):
        return self.mu


    def getCovariance(self):
        return self.Sigma


    def getVariances(self):
        return np.array([[self.Sigma[0,0],self.Sigma[1,1],self.Sigma[2,2]]])

    @property
    def x(self):
        return self.mu[0][0]
    @property
    def y(self):
        return self.mu[0][1]
    @property
    def theta(self):
        return self.mu[0][2]

    # Perform the prediction step to determine the mean and covariance
    # of the posterior belief given the current estimate for the mean
    # and covariance, the control data, and the process model
    #    u:  The forward distance and change in heading, tuple
    def prediction(self,u):
        d, thet = u
        # define jacobian
        F = np.array([[1.0, (d * -1.0) * math.sin(self.x), 0.0],
                    [0.0, 1.0, d * math.cos(self.theta)],
                    [0.0, 0.0, 1.0]])
        # step 2 from text
        mubar = [0.0] * 3
        noise = np.random.multivariate_normal(np.zeros(self.R.shape[1]), self.R)
        mubar[0] =  self.x + d * math.cos(self.theta) + noise[0]
        mubar[1] =  self.y + d * math.sin(self.theta) + noise[1]
        mubar[2] = self.theta + thet + noise[2]
        for i in range(3):
            self.mu[0][i] = mubar[i] # replace mu

        # step 3 from text
        self.Sigma = np.matmul(F, self.Sigma)
        self.Sigma = np.matmul(self.Sigma, F.T) + self.R

    # Perform the measurement update step to compute the posterior
    # belief given the predictive posterior (mean and covariance) and
    # the measurement data
    #  z:  The squared distance to the sensor and the
    #     robot's heading
    def update(self,z):

        d, head = z

        H = np.array([[2.0 * self.x, 2.0 * self.y, 0.0],
                            [0.0, 0.0, 1.0]])

        K = np.matmul(H, self.Sigma)
        K = np.matmul(K, H.T) + self.Q
        K = np.matmul(np.matmul(self.Sigma, H.T), np.linalg.inv(K))

        h = [0.0] * 2

        # assuming this function is ok b/c we can use numpy?

        noise =  np.random.multivariate_normal(np.zeros(self.Q.shape[1]), self.Q)

        h[0] = self.x**2 + self.y**2 + noise[0] # wonder why this not euclid?
        h[1] = self.theta + noise[1]
        h = np.array(h)

        self.mu = self.mu + np.matmul(K, z - h)

        kh = np.matmul(K, H)

        self.Sigma = np.matmul((np.identity(kh.shape[0]) - kh),
                            self.Sigma)
