import numpy as np
from Gridmap import Gridmap
from scipy.stats import norm, expon
from scipy.integrate import quad as integral

class Laser(object):
    # Construct an Laser instance with the following set of variables,
    # which are described in Section 6.3.1 of Probabilistic Robotics
    #   numBeams:   Number of beams that comprise the scan
    def __init__(self, numBeams = 41, sparsity=1):
        self.pHit = 0.97
        self.pShort = 0.01
        self.pMax = 0.01
        self.pRand = 0.01
        self.sigmaHit = 0.5
        self.lambdaShort = 1
        self.zMax = 20
        self.zMaxEps = .1
        self.Angles = np.linspace(-np.pi, np.pi, numBeams) # array of angles
        self.Angles = self.Angles[::sparsity]

        # Pre-compute for efficiency
        self.normal = norm(0, self.sigmaHit)
        self.exponential = expon(self.lambdaShort)



    # The following computes the likelihood of a given LIDAR scan from
    # a given pose in the provided map according to the algorithm given
    # in Table 6.1 of Probabilistic Robotics
    #
    #   Ranges:        An array of ranges (the angles are defined by self.Angles)
    #   x, y, thta:    The robot's position (x,y) and heading
    #   gridmap:       An instance of the Gridmap class that specifies
    #                  an occupancy grid representation of the map
    #                  where 1: occupied and 0: free
    #
    # Returns:
    #   likelihood:     Scan likelihood
    def scanProbability (self, z, x, gridmap):

        x,y,theta = *x

        q = 1

        XY = self.getXY(z, self.angles)

        for k in range(len(z)):

            # define zmax values, etc.
            zmax = z[z == self.zMax].size[0] 
            zhit = z[(z >= 0) & (z <= self.zMax)].size[0]
            zshort = z[(z >= 0) & (z <= z_subt_kstar)].size[0]
            zrand = z[(z >= 0) & (z < self.zmax)].size[0]

            zarr = np.array([ zmax, zhit, zshort, zrand ])

            # 6.12

            zmax, zhit, zshort, zrand = zarr / zarr.sum()


            # compute z_t ^kstar for z_t ^k with ray tracing

            xr, yr = *XY[k]
            thetar = theta

            z_subt_kstar = self.rayTracing(xr, yr, thetar, 
                                    self.angles[k], gridmap)

            if z[k] <= self.zMax and z[k] >= 0: # 6.4 in prob robotics
                n = norm(z_subt_kstar, self.sigmaHit**2)
                phit = n.pdf(z[k])
                phit *= (integral(n,0,self.zMax)) ** (-1.0)
            else:
                phit = 0.0

            # 6.8 in prob robotics
            if z[k] <= z_subt_kstar and z[k] >= 0:
                pshort = self.lambdaShort * np.exp(-self.lambdaShort * z[k])
                pshort *= 1/(1 - np.exp(-self.lambdaShort * z_subt_kstar))
            else:
                pshort = 0.0

            # 6.11,10

            if z[k] < z_subt_kstar and z[k] >= 0:
                prand = 1.0/zmax
            else:
                prand = 0.0

            # cumprod p with q

            q *= zhit * phit + zshort * pshort + zmax * pmax + zrand * prand


        return q


    # Function to convert range and bearing to (x,y) in LIDAR frame
    #   ranges:   1xn array of range measurements
    #   bearing: 1xn array of bearings
    #
    # Returns:
    #   XY:      2xn array, where each column is an (x,y) pair
    def getXY (self, ranges, bearing):

        CosSin = np.vstack((np.cos(bearing[:]),np.sin(bearing[:])))
        XY = np.tile(ranges,(2,1))*CosSin

        return XY


    def sampleRange (self, range):
        P = np.array([self.pHit, self.pShort, self.pMax, self.pRand])


    # An vectorized implementation of ray tracing
    #   (xr, yr, thetar):   The robot's pose
    #   lAngle:             The LIDAR angle (in the LIDAR reference frame)
    #   gridmap:            An instance of the Gridmap class that specifies
    #                       an occupancy grid representation of the map
    #                       where 1: occupied and 0: free
    #
    # Returns:
    #   d:                  Range
    #   coords:             Array of (x,y) coordinates
    def rayTracing(self, xr, yr, thetar, lAngle, gridmap):

        angle = np.array(thetar[:,None] + lAngle[None])
        x0 = np.array(xr/gridmap.xres)
        y0 = np.array(yr/gridmap.yres)

        x0 = np.tile(x0[:,None], [1, angle.shape[1]])
        y0 = np.tile(y0[:,None], [1, angle.shape[1]])
        assert angle.shape == x0.shape
        assert angle.shape == y0.shape

        def inCollision(x, y):
            return gridmap.inCollision(np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), True)

        (m,n) = gridmap.getShape()
        in_collision = inCollision(x0,y0)

        x0[x0 == np.floor(x0)] += 0.001
        y0[y0 == np.floor(y0)] += 0.001
        eps = 0.0001

        def inbounds(x, low, high):
            # return x in [low, high)
            return (x < high) * (x >= low)

        # Intersection with horizontal lines
        x = x0.copy()
        y = y0.copy()
        dir = np.tan(angle)
        xh = np.zeros_like(x)
        yh = np.zeros_like(y)
        foundh = np.zeros(x.shape, dtype=np.bool)
        seps = np.sign(np.cos(angle)) * eps
        while np.any(inbounds(x, 1, n)) and not np.all(foundh):
            x = np.where(seps > 0, np.floor(x+1), np.ceil(x-1))
            y = y0 + dir*(x-x0)
            inds = inCollision(x+seps,y) * np.logical_not(foundh) * inbounds(y, 0, m)
            if np.any(inds):
                xh[inds] = x[inds]
                yh[inds] = y[inds]
                foundh[inds] = True

        # Intersection with vertical lines
        x = x0.copy()
        y = y0.copy()
        eps = 1e-6
        dir = 1. / (np.tan(angle) + eps)
        xv = np.zeros_like(x)
        yv = np.zeros_like(y)
        foundv = np.zeros(x.shape, dtype=np.bool)
        seps = np.sign(np.sin(angle)) * eps
        while np.any(inbounds(y, 1, m)) and not np.all(foundv):
            y = np.where(seps > 0, np.floor(y+1), np.ceil(y-1))
            x = x0 + dir*(y-y0)
            inds = inCollision(x,y+seps) * np.logical_not(foundv) * inbounds(x, 0, n)
            if np.any(inds):
                xv[inds] = x[inds]
                yv[inds] = y[inds]
                foundv[inds] = True

        if not np.all(foundh + foundv):
            assert False, 'rayTracing: Error finding return'

        # account for poses in collision
        xh[in_collision] = x0[in_collision]
        yh[in_collision] = y0[in_collision]

        # get dist and coords
        dh = np.square(xh - x0) + np.square(yh - y0) + 1e7 * np.logical_not(foundh)
        dv = np.square(xv - x0) + np.square(yv - y0) + 1e7 * np.logical_not(foundv)
        d = np.where(dh < dv, dh, dv)
        cx = np.where(dh < dv, xh, xv)
        cy = np.where(dh < dv, yh, yv)
        coords = np.stack([cx,cy], axis=-1)
        return np.sqrt(d), coords
