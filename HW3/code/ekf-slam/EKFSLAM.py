import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Visualization import Visualization
from math import sin, cos

class EKFSLAM(object):
    # Construct an EKF instance with the following set of variables
    #    mu:                 The initial mean vector
    #    Sigma:              The initial covariance matrix
    #    R:                  The process noise covariance
    #    Q:                  The measurement noise covariance
    #    visualize:          Boolean variable indicating whether to visualize
    #                        the filter
    def __init__(self, mu, Sigma, R, Q, visualize=True):
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q
        self.nfeatures = 0

        # step 6
         # expand matrix to 3,3
        #self.Q = np.pad(self.Q, ((0,1),(0,1)), mode = 'constant',
        #                                 constant_values = 0.0)

        # You may find it useful to keep a dictionary that maps a
        # a feature ID to the corresponding index in the mean_pose_handle
        # vector and covariance matrix
        self.mapLUT = {}

        self.visualize = visualize
        if self.visualize == True:
            self.vis = Visualization()
        else:
            self.vis = None


    # Visualize filter strategies
    #   deltat:  Step size
    #   XGT:     Array with ground-truth pose
    def render(self, XGT=None):
        deltat = 0.1
        self.vis.drawEstimates(self.mu, self.Sigma)
        if XGT is not None:
            #print XGT
            self.vis.drawGroundTruthPose (XGT[0], XGT[1], XGT[2])
        plt.pause(deltat)


    def mu_extended(self,mu, n):
        placeholders = np.array([[0.0]* n]).ravel()
        return np.concatenate([mu, placeholders])


    # Perform the prediction step to determine the mean and covariance
    # of the posterior belief given the current estimate for the mean
    # and covariance, the control data, and the process model
    #    u: The forward distance and change in heading
    def prediction(self, u):
        u1, u2 = u
        v1, v2 = np.random.multivariate_normal([0.0,0.0], self.R)

        # 10.2.3
        F_x = np.concatenate([np.eye(3), 
                            np.zeros((3,self.nfeatures * 3))],
                            axis = 1)

        # put mu in higher dimensional space
        #mu = self.mu_extended(self.mu)

        mu =  self.mu.ravel()[:3]

        x,y,theta = mu

        noise_jac = np.array([ [cos(theta),0],
                                [sin(theta),0],
                                [0,1]])

        mot = [] # column vector of len 3
        # apply motion model (eq 8 in pset), 10.2.4 in book
        mot.append(x + (u1 + v1)*cos(theta))
        mot.append(y + (u1 + v1)*sin(theta))
        mot.append(theta + u2 + v2)
        mot = np.array(mot)

        # jacobian of motion model f(x,y,d,t)
        #F = np.array([[1.0, 0.0, np.cos(self.mu[2]), -(u1 + v1) * np.sin(self.mu[2])],
        #             [0.0, 1.0, np.sin(self.mu[2]), (u1 + v1) * np.cos(self.mu[2])],
        #             [0.0, 0.0, 1.0, 1.0]])
        # 10.16 
        g_t = np.array([
                    [1.0, 0.0, cos(theta)],
                    [0.0, 1.0, sin(theta)],
                    [0.0, 0.0, 1.0]])

        #sigma_new = np.matmul(F, self.Sigma)
        #sigma_new = np.matmul(sigma_new, F.T)
        #self.Sigma = sigma_new + self.R

        # noise jacobian

        # following from pp 314 in prob. robotics, s10.2

        mubar_t = mu + np.matmul(F_x.T, mot) # 10.2.4

        I = np.eye(((3 * self.nfeatures) + 3))

        # step 4
        G_t = I + np.matmul(np.matmul(F_x.T, g_t),
                            F_x)
        R_t = np.matmul(noise_jac, np.matmul(self.R, noise_jac.T))

        sigmabar_t = ( np.matmul(np.matmul(G_t, self.Sigma), G_t.T) +
                        np.matmul(np.matmul(F_x.T, R_t), F_x))

        self.Sigma = sigmabar_t
        self.mu = mubar_t
        print self.mu.shape


    # Perform the measurement update step to compute the posterior
    # belief given the predictive posterior (mean and covariance) and
    # the measurement data
    #    z:     The (x,y) position of the landmark relative to the robot, and
    #    i:     The ID of the observed landmark
    def update(self, z, i):
        z_x, z_y  = z

        x,y,theta = self.mu.ravel()[:3]

        if i not in self.mapLUT:
            w1, w2 = np.random.multivariate_normal([0.0,0.0], self.Q)
            i_loc = self.mu.shape[0]
            self.mapLUT[i] = {'x': i_loc, 'y': i_loc + 1}
            self.mu = self.mu_extended(self.mu, 2)
            self.nfeatures += 2
            self.mu[i_loc] = cos(theta) * z_x - sin(theta) * z_y - x + w1
            self.mu[i_loc + 1] = sin(theta) * z_x - cos(theta) * z_y - y + w2
            #self.mu = np.pad(self.mu, ((0,1),(0,1)), mode = 'constant',
            #                constant_values = 0.0)

            #mu[i_loc,:2] = (np.matmul(np.array([[cos(theta), sin(theta)],
            #                    -sin(theta), cos(theta)]),
            #                    np.array([[z_x - x, z_y - y]])) +
            #                   np.random.multivariate_normal([0.0,0.0], self.Q))

        #else:
        #    i_loc = self.mapLUT[i]

        x_idx = self.mapLUT[i]['x']
        y_idx = self.mapLUT[i]['y']

        j = x_idx + 1
        N = self.mu.shape[0] + 1

        delta = np.array([self.mu[x_idx] - x,
                        self.mu[y_idx] - y])

        q = np.matmul(delta.T, delta)

        # measurement jacobian

        H = np.array([[-1.0, 0.0, -z_x * cos(theta) - z_x * sin(theta), cos(theta), -sin(theta)],
                    [0.0, -1.0, -z_y * sin(theta) + z_x * cos(theta), sin(theta), -cos(theta)]])

        # matrix to map to higher dimensions

        F_l = np.pad(np.eye(3), ((0,2),(0, 2 * j - 2)), 'constant', constant_values = 0.0)
        F_r = np.pad(np.eye(3), ((2,0),(0,2 * N - 2 * j)),'constant', constant_values = 0.0)


        F = np.concatenate([F_l, F_r], axis = 1)

        H = np.matmul(H, F)

        right = np.linalg.inv(np.matmul(np.matmul(H,self.Sigma), H.T) + self.Q)
        left = np.matmul(self.Sigma, H.T)

        K = np.matmul(left,right)

        # measurement model + measure jac

        #z_i = np.matmul(np.array([[cos(theta), sin(theta)],
        #                -sin(theta), cos(theta)]),



        #x,y, theta = self.mu[i_loc:i_loc + 3,]

        # expand mu



        #self.mu[i_loc,]

        #print H.shape
        #print self.Sigma.shape
        #print self.Q.shape
        #print H.T.shape
        #left = np.matmul(self.Sigma, H.T)
        #right = np.linalg.inv(np.matmul(np.matmul(H,self.Sigma), H.T) + self.Q)
        #raise TypeError
        #K = np.matmul(left, right)

        #w = np.random.multivariate_normal([0.0, 0.0], self.Q)

        #measure_model = np.array([[cos(theta), sin(theta)],
        #                        -sin(theta), cos(theta)])
        #measure_model = np.matmul(measure_model,
        #                 np.array([[z_x - x, z_y - y]]))  + w

        #self.mu = self.mu + np.matmul(K, z - measure_model)

        #self.Sigma = np.matmul((np.eye(K.shape) - np.matmul(K,H)),
        #                       self.Sigma)

    # Augment the state vector to include the new landmark
    #    z:     The (x,y) position of the landmark relative to the robot
    #    i:     The ID of the observed landmark
    def augmentState(self, z, i):

        # define h
        #jacob = np.array([[-1, -np.cos(self.mu[2]), -sin(self.mu[2])],
        #                [-1, np.sin(self.mu[2]), -np.cos(self.mu[2])]]) d

        x,y,theta = self.mu
        z1, z2 = z

        G = np.array([[-1.0, 0.0, -z2 * np.cos(theta) - z1 * np.sin(theta), np.cos(theta), -np.sin(theta)],
                    [0.0, -1.0, -z2 * np.sin(theta) + z1 * np.cos(theta), np.sin(theta), -np.cos(theta)],
                    ])
        if i not in self.mapLUT:

            mu_new = np.array([[self.mu],
                        [np.cos(theta) * z1 - np.sin(theta * z2 - x1 + w1) ,
                        np.sin(theta) * z1 - np.cos(theta * z2 - x2 + w2)]])

            sigma_new = np.array([[self.Sigma, np.matmul(self.Sigma, G.T)],
                            [np.matmul(G, self.Sigma),
                            np.matmul(np.matmul(G, self.Sigma), G.T) + self.Q ]])
            self.mapLUT[i] = [mu_new, sigma_new]


    # Runs the EKF SLAM algorithm
    #   U:        Array of control inputs, one column per time step
    #   Z:        Array of landmark observations in which each column
    #             [t; id; x; y] denotes a separate measurement and is
    #             represented by the time step (t), feature id (id),
    #             and the observed (x, y) position relative to the robot
    #   XGT:      Array of ground-truth poses (may be None)
    def run(self, U, Z, XGT=None, MGT=None):

        # Draws the ground-truth map
        if MGT is not None:
            self.vis.drawMap (MGT)

        # Iterate over the data
        for t in range(U.shape[1]):
            u = U[:,t]
            num_match = np.where(Z[0,:] == t)
            z = Z[:,np.where(Z[0,:] == t)]

            self.prediction(u)

            for k in range(z.shape[2]):
                k = z[:,:,k].ravel()
                self.update(k[2:], k[1])

            # You may want to call the visualization function
            # between filter steps
            if self.visualize:
                if XGT is None:
                    self.render (None)
                else:
                    self.render (XGT[:,t])
