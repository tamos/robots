import sys
import math
from numpy import *
import matplotlib.pyplot as plt
from EKF import *
import os

class RunEKF(object):

    def __init__(self, Q_factor, R_factor):
        self.R = array([[2.0, 0.0, 0.0],[0.0, 2.0, 0.0],[0.0, 0.0, (2.0 * math.pi)/180]]) * R_factor
        self.Q = array([[1.0, 0.0],[0.0, math.pi/180]]) * Q_factor
        self.U = [] # Array that stores the control data where rows increase with time
        self.Z = [] # Array that stores the measurement data where rows increase with time
        self.XYT = [] # Array that stores the ground truth pose where rows increase with time
        self.MU = [] # Array in which to append mu_t as a row vector after each iteration
        self.VAR = [] # Array in which to append var(x), var(y), var(theta)
                      # from the EKF covariance as a row vector after each iteration

    # Read in the control and measurement data from their respective text files
    # and populates self.U and self.Z
    def readData(self, filenameU, filenameZ, filenameXYT):
        print "Reading control data from %s and measurement data from %s" % (filenameU, filenameZ)

        self.U = loadtxt(filenameU, comments='#', delimiter=',')
        self.Z = loadtxt(filenameZ, comments='#', delimiter=',')
        self.XYT = loadtxt(filenameXYT, comments='#', delimiter=',')

    # Iterate from t=1 to t=T performing the two filtering steps
    def run(self):

        mu0 = array([[-4.0, -4.0, math.pi/2]])# FILL ME IN: initial mean
        Sigma0 = eye(3) #[]# FILL ME IN: initial covariance
        self.VAR = array([[Sigma0[0,0], Sigma0[1,1], Sigma0[2,2]]])
        self.MU = mu0 # Array in which to append mu_t as a row vector after each iteration
        self.ekf = EKF(mu0, Sigma0, self.R, self.Q)

        # For each t in [1,T]
        #    1) Call self.ekf.prediction(u_t)
        #    2) Call self.ekf.update(z_t)
        #    3) Add self.ekf.getMean() to self.MU
        for t in range(size(self.U,0)):
            self.ekf.prediction(self.U[t,:])
            self.ekf.update(self.Z[t,:])
            self.MU = concatenate((self.MU, self.ekf.getMean()))
            self.VAR = concatenate((self.VAR, self.ekf.getVariances()))

        print "FINAL MEAN VECTOR IS:\n", self.ekf.getMean()

        print "\nFINAL COVARIANCE MATRIX IS:\n", self.ekf.getCovariance()

    # Plot the resulting estimate for the robot's trajectory
    def plot(self):

        # Plot the estimated and ground truth trajectories
        ground_truth = plt.plot(self.XYT[:,0], self.XYT[:,1], 'g.-', label='Ground Truth')
        mean_trajectory = plt.plot(self.MU[:,0], self.MU[:,1], 'r.-', label='Estimate')
        plt.legend()

        plt.savefig(PLOT_DIR + "true_trajectory.png")
        plt.close()

        # Try changing this to different standard deviations

        sigmas = np.arange(0,3)
        for sigma in sigmas:

            # Plot the errors with error bars
            Error = self.XYT-self.MU
            T = range(size(self.XYT,0))
            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].plot(T,Error[:,0],'r-')
            axarr[0].plot(T,sigma*sqrt(self.VAR[:,0]),'b--')
            axarr[0].plot(T,-sigma*sqrt(self.VAR[:,0]),'b--')
            axarr[0].set_title('X error')
            axarr[0].set_ylabel('Error (m)')

            axarr[1].plot(T,Error[:,1],'r-')
            axarr[1].plot(T,sigma*sqrt(self.VAR[:,1]),'b--')
            axarr[1].plot(T,-sigma*sqrt(self.VAR[:,1]),'b--')
            axarr[1].set_title('Y error')
            axarr[1].set_ylabel('Error (m)')

            axarr[2].plot(T,degrees(unwrap(Error[:,2])),'r-')
            axarr[2].plot(T,sigma*degrees(unwrap(sqrt(self.VAR[:,2]))),'b--')
            axarr[2].plot(T,-sigma*degrees(unwrap(sqrt(self.VAR[:,2]))),'b--')
            axarr[2].set_title('Theta error (degrees)')
            axarr[2].set_ylabel('Error (degrees)')
            axarr[2].set_xlabel('Time')

            plt.savefig(PLOT_DIR + str(sigma) + "_errors.png")
            plt.close()

        return



if __name__ == '__main__':

    # This function should be called with three arguments:
    #    sys.argv[1]: Comma-delimited file containing control data (U.txt)
    #    sys.argv[2]: Comma-delimited file containing measurement data (Z.txt)
    #    sys.argv[3]: Comma-delimited file containing ground-truth poses (XYT.txt)
    if len(sys.argv) < 5:
        print "usage: RunEKF.py ControlData.txt MeasurementData.txt GroundTruthData.txt [classic/custom]"
        sys.exit(2)
    qr_mode = sys.argv[4]

    base_offset = 0.1

    if qr_mode == 'classic':
        base_offset_q =  1.0E-6
        base_offset_r =  1.0E-4
        PLOT_DIR = "/Users/ty/Documents/robots/Amos_T_HW_2/classic_plots/sigma_"
        ekf = RunEKF(base_offset_q, base_offset_r)
        ekf.readData(sys.argv[1], sys.argv[2], sys.argv[3])
        ekf.run()
        ekf.plot()

    elif qr_mode == 'custom':
        for i in range(1,100,10):
            base_offset_q =  1.0/float(i)
            base_offset_r =  1.0/float(i)
            PLOT_DIR = "/Users/ty/Documents/robots/Amos_T_HW_2/custom_plots/qr_factor_" + str(i) + "_sigma_"
            ekf = RunEKF(base_offset_q, base_offset_r)
            ekf.readData(sys.argv[1], sys.argv[2], sys.argv[3])
            ekf.run()
            ekf.plot()
