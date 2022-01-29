#!/usr/bin/env python
#coding: utf8
import numpy as np
import numpy.linalg as la
import time, os
import scipy.optimize
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class NBodyApprox:
    """
    An instance of approximation for the N-body problem.
    """
    def __init__(self, name, y, dt, m = np.empty(0), G=1., isStore=False, stTime=False):
        self.G = G
        self.dt = dt
        self.dim = len(y)/ (2 * len(m))
        self.N = len(m)
        self.y = y.copy()
        self.t = 0.
        self.m = m.copy()
        self.stTime = stTime
        self.isStore = isStore
        self.name = name
        self._storePath = ''
        self.file = None

    def steps(self, nrSteps):
        self.y = self.initStVer(self.y, self.dt)

        for i in np.xrange(nrSteps):
            self.y = self.stVer(self.y, self.dt)
            self.t += self.dt
        
        return self.y, self.t

    def setStorePath(self, path):
        self._storePath = path

    def _initStoring(self, mthd):
        """
        Create storage file in store location with current 'Unix
        time' as file name in a folder with approximations name.
        """
        fileName = "%i.dat" % (int(time.time()))
        path = os.path.join(self._storePath, self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.file = open(os.path.join(path, fileName), 'w')
        self._insertHeader(mthd)
        self._store(self.y, self.t)

    def _insertHeader(self, mthd):
        """
        Create header for storage file with all relative
        data to this approximation. First two bytes
        descripe endianness (value should be 1 for 16
        bit integer).
        """
        np.ones(1,dtype='int16').tofile(self.file)
        np.array([self.dim, self.N,self.stTime], dtype = 'int32').tofile(self.file)
        np.array([self.G, self.dt],dtype = 'float64').tofile(self.file)
        self.m.tofile(self.file)
        np.array([mthd],dtype='|S5').tofile(self.file)

    def _store(self, y, t):
        """Stor current state, and time if specified."""
        y.tofile(self.file)
        if self.stTime:
            np.array((t,)).tofile(self.file)
        
    def _endStoring(self):
        self.file.close()

    def rhs(self, y):
        """Calculate the right hand side of the N-body problem."""
        a = self.force(y)
        return np.hstack((y[self.N*self.dim:], a))

    def force(self, y):
        a = np.zeros((self.N,self.dim))
        for j in np.xrange(self.N):
            for k in np.xrange(j):
                # r = r_k - r_j -- distance between two objects
                r = y[k*self.dim:(k+1)*self.dim] - y[j*self.dim:(j+1)*self.dim]
                #get acceleration between objects
                f = self.G * r/(la.norm(r)**3)
                #save acceleration in matrix
                a[j] += f*self.m[k]
                a[k]-= f*self.m[j]

        # flatten matrix to 1d vector -- get 
        return a.flatten() 

    def euler(self, y, h):
        return y + self.rhs(y) * h

    def stVer(self, y, h):
        r = y[:self.N * self.dim] + h * y[self.N * self.dim:]
        v = y[self.N * self.dim:] + h * self.force(r)
        return np.hstack((r, v))

    def initStVer(self, y, h):
        """Take initial half step for stVer with rk4."""
        v = self.euler(y, h/2.)[self.N * self.dim:]
        return np.hstack((y[:self.N * self.dim], v))


def loadData(filePath):
    """
    Load the data from a file as it's saved in NBodyApprox. Return
    a dictionary containing the approximation data and the values
    from the header. Remember to check for endiannes.
    """
    res = {}

    with open(filePath) as f:
        res['endian'] = np.fromfile(f,dtype='int16',count=1)
        tmp = np.fromfile(f,dtype='int32',count=3)
        res['dim'] = tmp[0]
        res['N'] = tmp[1]
        res['stTime'] = tmp[2]
        tmp = np.fromfile(f,dtype='float64',count=2)
        res['G'] = tmp[0]
        res['dt'] = tmp[1]
        res['m'] = np.fromfile(f,dtype='float64',count=res['N'])
        res['mthd'] = np.fromfile(f,dtype='|S5',count=1)
        res['y'] = np.fromfile(f,dtype='float64')
    col = 2 * res['N'] * res['dim']

    if res['stTime']:
        col += 1
    rows = res['y'].size / col
    res['y'] = res['y'].reshape((rows,col))

    return res

def tot_en(y,m,d=3,G=1.):
    """Calculate the total energy of states y of a system."""
    n = len(m)
    t = np.zeros(len(y))
    for i in np.xrange(n):
        t += np.sum(y[:,(n+i)*d:(n+i+1)*d]**2,axis=-1) * m[i]
    t *= .5

    u = np.zeros(len(y))
    for i in np.xrange(n):
        for j in np.xrange(i):
            r = np.sum((y[:,j*d:(j+1)*d] - y[:,i*d:(i+1)*d])**2,axis=-1)**.5
            u += (m[i] + m[j]) / r
    u *= G
    return t - u





class N_body_problem:
    """
    Create a random N-body system out of given number of particles.
    Make sure that the center of mass is fixed at the origin.
    """
    def __init__(self, nrBodies, m=None):
        self.nrBodies = nrBodies
        if m and len(m) == nrBodies:
            self.m = m
        else:
            self.m = np.random.uniform(1,2,nrBodies)

        self.r = self._construct(nrBodies,self.m)
        self.v = .1 * self._construct(nrBodies,self.m)

    def _construct(self, nrb, m):
        cns = np.random.uniform(-2,2,nrb*3).reshape((nrb,3))
        return self._reduce(cns, m)

    def _reduce(self,z,m):
        return z - np.sum(z*m.reshape(-1,1),axis=0)/m.sum()

    def get_y(self):
        return np.hstack((self.r.flat,self.v.flat))

    def get_m(self):
        return self.m.copy()


nb = N_body_problem(3)

n = NBodyApprox(name="test", y=[], dt=0.1)