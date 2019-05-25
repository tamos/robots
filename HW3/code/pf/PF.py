import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Gridmap import Gridmap
from Laser import Laser
from Visualization import Visualization
from Queue import Queue

#matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIV_OFFSET = 1E-15

def normprob(a,b): # after 5.2 in probabilistic robotics
    rv = 1/(np.sqrt(2 * np.pi * b**2))
    return rv * np.exp(-(1/2) * (a**2/b**2))



class PF(object):
    # Construct an PF instance with the following set of variables
    #    numParticles:       Number of particles
    #    Alpha:              Vector of 6 noise coefficients for the motion
    #                        model (See Table 5.3 in Probabilistic Robotics)
    #    laser:              Instance of the laser class that defines
    #                        LIDAR params, observation likelihood, and utils
    #    gridmap:            An instance of the Gridmap class that specifies
    #                        an occupancy grid representation of the map
    #                        where 1: occupied and 0: free
    #    visualize:          Boolean variable indicating whether to visualize
    #                        the particle filter
    def __init__(self, numParticles, Alpha, laser, gridmap, visualize = True):
        self.numParticles = numParticles
        self.Alpha = Alpha
        self.laser = laser
        self.gridmap = gridmap
        self.visualize = visualize
        self.ntries = 5

        # particles is a numParticles x 3 array, where each column denote a particle_handle
        # weights is a numParticles x 1 array of particle weights
        self.particles = None
        self.weights = None

        if self.visualize == True:
            self.vis = Visualization()
            self.vis.drawGridmap(self.gridmap)
        else:
            self.vis = None

    # Samples the set of particles according to a uniform distribution
    # and sets the weigts to 1/numParticles. Particles in collision are rejected
    def sampleParticlesUniform (self):

        (m,n) = self.gridmap.getShape()

        self.particles = np.empty([3,self.numParticles])

        for i in range(self.numParticles):
            theta = np.random.uniform(-np.pi,np.pi)
            inCollision = True
            while inCollision:
                x = np.random.uniform(0,(n-1)*self.gridmap.xres)
                y = np.random.uniform(0,(m-1)*self.gridmap.yres)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:,i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))


    # Samples the set of particles according to a Gaussian distribution
    # Orientation are sampled from a uniform distribution
    #    (x0, y0):    Mean position
    #    sigma:       Standard deviation
    def sampleParticlesGaussian (self, x0, y0, sigma):

        (m,n) = self.gridmap.getShape()

        self.particles = np.empty([3,self.numParticles])

        for i in range(self.numParticles):
            #theta = np.random.uniform(-np.pi,np.pi)
            inCollision = True
            while inCollision:
                x = np.random.normal(x0,sigma)
                y = np.random.normal(y0,sigma)
                theta = np.random.uniform(-np.pi, np.pi)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:,i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))



    # Returns desired particle (3 x 1 array) and weight
    def getParticle (self, k):

        if k < self.particles.shape[1]:
            return (self.particles[:,k], self.weights[:,k])
        else:
            print 'getParticle: Request for k=%d exceeds number of particles (%d)' % (k, self.particles.shape[1])
            return (None, None)

    # Return an array of normalized weights. Does not normalize the weights
    # maintained in the PF instance
    #
    # Returns:
    #   weights:   Array of normalized weights
    def getNormalizedWeights (self):

        return self.weights/np.sum(self.weights)


    # Returns the particle filter mean
    def getMean(self):

        weights = self.getNormalizedWeights()
        return np.sum(np.tile(weights, (self.particles.shape[0], 1)) * self.particles, axis=1)


    # Visualize filter strategies
    #   ranges:  Array of LIDAR ranges
    #   deltat:  Step size
    #   XGT:     Array with ground-truth pose
    def render(self, ranges, deltat, XGT):
        self.vis.drawParticles(self.particles, self.weights)
        if XGT is not None:
            self.vis.drawLidar(ranges, self.laser.Angles, XGT[0], XGT[1], XGT[2])
            self.vis.drawGroundTruthPose (XGT[0], XGT[1], XGT[2])
        mean = self.getMean()
        self.vis.drawMeanPose(mean[0], mean[1], mean[2])
        plt.pause(deltat)


    # Sample a new pose from an initial pose (x, y, theta)
    # with inputs v (forward velocity) and w (angular velocity)
    # for deltat seconds
    #
    # This model corresponds to that in Table 5.3 in Probabilistic Robotics
    #
    # Returns:
    #   (xs, ys, thetas):   Position and heading for sample
    def sampleMotion (self, x, y, theta, u1, u2, deltat):

        # Your code goes here: Implement the algorithm given in Table 5.3
        # Note that the "sample" function in the text assumes zero-mean
        # Gaussian noise. You can use the NumPy random.normal() function
        # Be sure to reject samples that are in collision
        # (see Gridmap.inCollision), and to unwrap orientation so that it
        # it is between -pi and pi.

        abs_v = np.abs(u1) # v 
        abs_omega = np.abs(u2) # omega

        theta = np.radians(theta)


        vt1_q = np.random.normal(0, self.alpha1 * u1**2 + self.alpha2 * u2**2,
                         size = self.ntries)
        vt2_q = np.random.normal(0, self.alpha3 * u1**2 + self.alpha4 * u2**2,
                        size = self.ntries)
        gammat_q = np.random.normal(0, self.alpha5 * u1**2 + self.alpha6 * u2**2,
                    size = self.ntries)

        vhat_q = np.random.normal(0, self.alpha1 * abs_v + self.alpha2 * abs_omega,
                    size = self.ntries)

        omegahat_q = np.random.normal(0, self.alpha3 * abs_v + self.alpha4 * abs_omega,
                    size = self.ntries)

        gammahat_q = np.random.normal(0, self.alpha5 * abs_v + self.alpha6 * abs_omega,
                        size = self.ntries)

        in_collision = True

        for k in range(self.ntries): # attempt a given number of times to find noncollision

            #vt1 = np.random.normal(0, self.alpha1 * u1**2 + self.alpha2 * u2**2)
            #vt2 = np.random.normal(0, self.alpha3 * u1**2 + self.alpha4 * u2**2)
            #gammat = np.random.normal(0, self.alpha5 * u1**2 + self.alpha6 * u2**2)

            vt1 = vt1_q[k]
            vt2 = vt2_q[k]
            gammat = gammat_q[k]

            ubart1 = u1 + vt1
            vbart2 = u2 + vt2

            xt  = x + ubart1/vbart2 * (np.sin(theta + vbart2 * deltat) - np.sin(theta))
            yt = y + ubart1/vbart2 * (np.cos(theta) - np.cos(theta + vbart2 * deltat))
            thetat = theta + vbart2 * deltat + gammat * deltat

            # compute vhat

            #vhat = u1  + np.random.normal(0, self.alpha1 * abs_v + self.alpha2 * abs_omega)
            vhat = u1 + vhat_q[k]
            # compute omegahat

            omegahat = u2 + omegahat_q[k]

            #omegahat = u2 + np.random.normal(0, self.alpha3 * abs_v + self.alpha4 * abs_omega)

            # compute xprime

            xprime = ( xt - (vhat / omegahat) * np.sin(theta) + 
                            (vhat / omegahat) * np.sin(theta + omegahat * deltat) )

            # compute yprime

            yprime = ( yt + (vhat / omegahat) * np.cos(theta) + 
                            (vhat / omegahat) * np.cos(theta + omegahat * deltat) )

            # compute gammahat

            #gammahat  = np.random.normal(0, self.alpha5 * abs_v + self.alpha6 * abs_omega)

            gammahat = gammahat_q[k]

            # compute thetaprime

            thetaprime  = theta + omegahat * deltat + gammahat * deltat

            if not self.gridmap.inCollision(xprime, yprime):
                in_collision = False
                break

        if in_collision: # if still in collision, return the input values unchanged
            return np.array([x, y, np.degrees(thetaprime)]).T
        else:
            return np.array([xprime, yprime, np.degrees(thetaprime)]).T


    @property
    def alpha1(self):
        return self.Alpha[0]

    @property
    def alpha2(self):
        return self.Alpha[1]

    @property
    def alpha3(self):
        return self.Alpha[2]

    @property
    def alpha4(self):
        return self.Alpha[3]

    @property
    def alpha5(self):
        return self.Alpha[4]

    @property
    def alpha6(self):
        return self.Alpha[5]


    # Function that performs resampling with replacement
    def resample (self):

        # resample self.numParticles particles according to weights
        particle_keys = np.arange(self.particles.shape[1])
        selected_particles = np.random.choice(a = particle_keys,  
                                    size = self.numParticles, 
                                    p = self.weights.ravel())
        self.particles = self.particles[:,selected_particles]

    # Perform the prediction step
    def prediction(self, u, deltat):
        u1, u2 = u

        new_particles = np.zeros(self.particles.shape)

        for k in range(self.numParticles):
            part, wts = self.getParticle(k)
            x,y,theta = part
            new_particles[:,k] = self.sampleMotion(x, y, theta, u1, u2, deltat)

        self.particles = new_particles


    # Perform the measurement update step
    #   Ranges:   Array of ranges (Laser.Angles provides bearings)
    def update(self, Ranges):
        # reweight particles according to obs likelihood

        for particle_k in range(self.numParticles):
            # get scan likelihood based on estimated pose
            particle_pose, particle_wt = self.getParticle(particle_k)
            scan_likelihood = self.laser.scanProbability(Ranges, particle_pose, self.gridmap)
            self.weights[:,particle_k] = particle_wt * scan_likelihood
        self.weights = self.getNormalizedWeights()


    # Runs the particle filter algorithm
    #   U:        Array of control inputs, one column per time step
    #   Ranges:   Array of LIDAR ranges for each time step
    #             The corresponding bearings are defined in Laser.angles
    #   deltat:   Number of seconds per time step
    #   X0:       Array indicating the initial pose (may be None)
    #   XGT:      Array of ground-truth poses (may be None)
    #   filename: Name of file for plot
    def run(self, U, Ranges, deltat, X0, XGT, filename, sampleGaussian, sigma):

        # Try different sampling strategies (including different values for sigma)
        if sampleGaussian and (X0 is not None):
            self.sampleParticlesGaussian(X0[0,0], X0[1,0], sigma)
        else:
            self.sampleParticlesUniform()

        # Iterate over the data, for each time step
        for k in range(U.shape[1]):
            print "At time step ", k
            u = U[:,k]
            ranges = Ranges[:,k+1][0]

            if self.visualize:
                if XGT is None:
                    self.render (ranges, deltat, None)
                else:
                    self.render (ranges, deltat, XGT[:,k])

            self.prediction(u, deltat)

            self.update(ranges)

            # only resample intermittently
            self.resample()

        plt.savefig(filename)
