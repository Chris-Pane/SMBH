from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from scipy import optimize
import multiprocessing as mp
import time

from SMBH.lib.SMBH_Modules import *


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       FUNCTIONS AND VARS
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

# Generator for the Parameter Vector. Input the numbers of the Stars, that are included in the calculation
ParVecTemplate = [2]
OriginParVec = PV(ParVecTemplate)
OriginParVec.SetDynamicCenter(False)


# Display additional info for debugging
DEBUG_MODE = False
# File with all the data
fileName = "SMBH/Data/OrbitData2017.txt"
# Output file for MCD
outputFile = "SMBH/Data/Output.txt"
#counter
GLOB_counter = 0
# Global Bounds of SGR A* Position in mas -- (del x, del y , del z)
GLOB_SGRA_Pos = np.array([0.2, 0.2, 0.2])


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       ROUTINE
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def getOrbit(SD:StarContainer, Orb:OrbitElem, ParamVec:list, index:int, max_iter:float, stepsize:float, kwargs:dict={}) -> FitContainer:
    """
    Calculates the orbit for given parameters
    Parameters
    ----------
    SD : DataContainer
        Star Data imported from file
    Orb : OrbitElem
        Orbital Elements for initial state
    potential : function
        function that returns the strength of the potential. must take r,v,M as params in order
    ParamVec : list
        Parameter Vector (8 dim)
    index : int
        Index point from which the Algorithm starts to integrate in both directions
    max_iter : float
        Max Integration steps in either direction from index
    stepsize : float
        delta t

    Returns
    -------
    FitContainer
        Resulting Fit Data
    """

    # convert parameter vector to parameters
    _M, _r, _v, _d = ParVecToParams(ParamVec)

    # switch the Integrator method
    if 'method' in kwargs.keys():
        if kwargs['method'] == 'Schwarz':
            IntegratorToUse = VerletStepSchwarz
            PotFunc = potentialSchwarz
        else:
            IntegratorToUse = VerletStep
            PotFunc = potential
    else:
        IntegratorToUse = VerletStep
        PotFunc = potential

    # try different positions of sgr a*
    if 'varSGRA' in kwargs.keys():
        # convert to pc using the current distance while fitting
        SGRA_Pos = PosRadToReal(kwargs['varSGRA'], _d)

    else:
        # default to (0,0,0)
        SGRA_Pos = np.array([0,0,0])
    
    #------------------------------------------------------------------------
    #   CURRENT VALUES

    PeriodFront = SD.TimeP[-1] - SD.TimeP[index]            # Time to render from index point to end
    PeriodBack  = SD.TimeP[index] - SD.TimeP[0]             # Time to render from beginning to index point
    r_cur = _r                                              # Position
    v_cur = _v                                              # Velocity
    f_cur = PotFunc(_r,_v,_M, SGRA_Pos)                     # init the potential
    init_Er = _r / np.linalg.norm(_r)                       # Unit vector of initial position
    

    #------------------------------------------------------------------------
    #   TRANSIENT DATA

    cur_timer = 0                                           # current iteration step in orbit integration
    last_err = 1                                            # unit vector of last position
    cutoff_timer = max_iter                                 # max iteration step in orbit integration
    PastOneMinus = False                                    # checking if integration is over the point where dot(now, init) = -1
    stopDotProd = False                                     # once true, the orbit end point will be calculated and no dot() will be executed
    OrbitPeriod = Orb.Period                                # Period of this orbit. used to determine total length of fake data
    OneOrbitSteps = 0                                       # used in backward rendering. represents how many steps one orbit is

    if (OrbitPeriod > 0):
        QuotientFront = np.ceil(PeriodFront / OrbitPeriod)  # number of orbits to compute in front direction
        QuotientBack = PeriodBack / OrbitPeriod             # number of orbits to compute in back direction, CAN BE FLOAT since steps for one orbit have been determined
    # if Period is not defined, return high error
    else: 
        FD_Fail = FitContainer(ParamVec, success=False, _orb=Orb)
        return FD_Fail

    if DEBUG_MODE:
        print("orbit time: ", OrbitPeriod)
        print("quotient front: ", QuotientFront)
        print("quotient back: ", QuotientBack)

    # Overwrite integration length if needed
    if 'front' in kwargs.keys():
        QuotientFront = kwargs['front']
    if 'back' in kwargs.keys():
        QuotientBack = kwargs['back']

    #save steps for plot and error handling. start with init point
    PosTracker = [_r]
    PosTrackerBack = []
    VelTracker = [_v]
    VelTrackerBack = []
    

    #------------------------------------------------------------------------
    #   ORBIT INTEGRATION

    # Integrate from Index Point to End point, forward in time
    while cur_timer < cutoff_timer:
        
        cur_timer += 1
        #get the next infinitesimal step in position and velocity
        _dat = IntegratorToUse(stepsize, r_cur, v_cur, f_cur, _M, r_SGRA=SGRA_Pos)
        #new position in real space
        r_cur = _dat[0]
        #new velocity
        v_cur = _dat[1]
        #new potential for next step
        f_cur = _dat[2]
        #position in radial space

        PosTracker.append(r_cur)
        VelTracker.append(v_cur)

        # determine end of orbit integration
        if (not stopDotProd):
            temp = np.dot( init_Er, r_cur/np.linalg.norm(r_cur) )
            if (not PastOneMinus):
                if (temp - last_err < 0 ):
                    last_err = temp
                else:
                    PastOneMinus = True
                    last_err = temp
            # orbit is past the dot() = -1 point => dot() increases again
            else:
                # dot() is still increasing
                if (temp > last_err):
                    last_err = temp
                # dot() has decreased or is equal to prev step => one orbit complete. calculate cutoff timer for end of Measurement data
                else:
                    # calculate multiple orbits based on data end
                    cutoff_timer = cur_timer * QuotientFront
                    # steps for one orbit
                    OneOrbitSteps = cur_timer
                    stopDotProd = True
                    if DEBUG_MODE:
                        print("forward cutoff = ", cutoff_timer)

    print(cur_timer)
    # reset some data
    cur_timer = 0
    r_cur = _r
    v_cur = _v            
    f_cur = PotFunc(_r,_v,_M)


    # Integrate multiple orbits backwards in time depending on data beginning
    while cur_timer < np.ceil(OneOrbitSteps * QuotientBack) :
        
        cur_timer += 1
        # reverse time
        _dat = IntegratorToUse(-stepsize, r_cur, v_cur, f_cur, _M, r_SGRA=SGRA_Pos)
        #new position in real space
        r_cur = _dat[0]
        #new velocity
        v_cur = _dat[1]
        #new potential for next step
        f_cur = _dat[2]


        PosTrackerBack.append(r_cur)
        VelTrackerBack.append(v_cur)


    if DEBUG_MODE:
        print("backward cutoff = ", cur_timer)

    print(cur_timer)

    #------------------------------------------------------------------------
    #   CONCAT DATA

    # reverse backward time points
    PosTrackerBack.reverse()
    VelTrackerBack.reverse()
    # append data
    PosTracker = np.array(PosTrackerBack + PosTracker)
    VelTracker = np.array(VelTrackerBack + VelTracker)


    #------------------------------------------------------------------------
    #   RETURN FIT DATA

    FD = FitContainer(ParamVec, _oN = QuotientFront, _orb=Orb, _PosArr=PosTracker, _VelArr = VelTracker)
    return FD


def FitDataInner(SD:StarContainer, kwargs:dict={}) -> Tuple[FitContainer, list]:
    """
    Inner Routine for fitting the orbit of a specified star.
    Only needs the Star Data Object to work.
    Can plot data in multiple ways
    Parameters
    ----------
    SD : DataContainer
        Star Data Object

    Returns
    -------
    [FitData Container, [index_R, index_V] ]
    """

    #------------------------------------------------------------------------
    #   CONSTRAINTS

    #Mass constraints in solar mass
    CSRT_M_min = 1E5
    CSRT_M_max = 1E7
    #Distance in pc
    CSRT_R_min = 7000
    CSRT_R_max = 9000


    #------------------------------------------------------------------------
    #   INIT VALUES

    # First Mass estimate -- random val in range
    Mass_0 = 4E6 #np.random.uniform(CSRT_M_min, CSRT_M_max)
    if DEBUG_MODE:
        print("Init Mass [10^6 Msun] = ", Mass_0/1E6)

    #Fist Distance Reference Point
    Distance_0 = 8000 #np.random.uniform(CSRT_R_min, CSRT_R_max)
    if DEBUG_MODE:
        print("Init Distance [pc] = ", Distance_0)


    # index for beginning point of data, IMPORTANT
    _index_R = 61 # 2007.55
    _index_V = 24 # 2007.55

    if DEBUG_MODE:
        print("Index Point Pos: ", SD.TimeP[_index_R], SD.TimeP[_index_R-1])
        print("Index Point RV : ", SD.TimeV[_index_V])

    # Position Vector, use random data point for start, convert to pc
    r0 = PosRadToReal( np.array( 
        [ 
            SD.RA[_index_R],
            SD.DE[_index_R],
            0
        ] ), Distance_0 )

    if DEBUG_MODE:
        print("Init Position [pc] = ", r0)

    #Velocity Vector, use same random point + point prior to calculate first estimate of v, in km/s
    v0 = np.array(
        [   
            (SD.RA[_index_R]-SD.RA[_index_R-1])*Distance_0*GLOB_PcYrKmS*GLOB_masToRad / (SD.TimeP[_index_R]-SD.TimeP[_index_R-1]),
            (SD.DE[_index_R]-SD.DE[_index_R-1])*Distance_0*GLOB_PcYrKmS*GLOB_masToRad / (SD.TimeP[_index_R]-SD.TimeP[_index_R-1]),
            SD.VR[_index_V]
        ])

    if DEBUG_MODE:
        print("Init Velocity [km/s] = ", v0)


    stepsize = 1E-8                      # orbit integration delta t
    # Algorithm breaks when max_iteration is smaller than the needed steps to complete the simulation 
    max_iteration = 1E6                  # orbit integration max steps; max for 1E-10 approx 600.000

    # stepsize overwrite
    if 'Stepsize' in kwargs.keys():
        stepsize = kwargs['Stepsize']

    global GLOB_counter
    GLOB_counter = 0

    if 'grav-red' in kwargs.keys():
        useGravRedCorr = kwargs['grav-red']
    else:
        useGravRedCorr = False

    if 'Pbar' in kwargs.keys():
        usePbar = kwargs['Pbar']
    else:
        usePbar = True
    

    #------------------------------------------------------------------------
    #   Parameter Vector

    parVec   = np.array([ Mass_0, r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], Distance_0 ])

        # [ (min, max) ]
    BOUNDS = [
        (1E6, 7E6),                                                                             # in msun, M
        (RadToReal( SD.RA[_index_R] - 15, 8000 ), RadToReal( SD.RA[_index_R] + 15, 8000 )),     # in pc, x
        (RadToReal( SD.DE[_index_R] - 15, 8000 ), RadToReal( SD.DE[_index_R] + 15, 8000 )),     # in pc, y
        (-0.2, 0.2),                                                                            # in pc, z
        (v0[0] - 1500,v0[0] + 1500),                                                            # in km/s, vx
        (v0[1] - 1500,v0[1] + 1500),                                                            # in km/s, vy
        (v0[2] - 500,v0[2] + 500),                                                              # in km/s, vz
        (7800, 8800)                                                                            # in pc, d
    ]
    
    #------------------------------------------------------------------------
    #   MAIN LOOP

    # function to be minimized
    def _tFitFunction(_parVec, *args):

        OrbEl = getOrbitalElements(_parVec)
        if usePbar:
            global GLOB_counter
            GLOB_counter += 1
        
        # everything with ecc> 1 or T<0, T>17 is wrong
        if (OrbEl.Period > 12 and OrbEl.Ecc < 1 and OrbEl.Period <= 20):
            _FD = getOrbit(SD=SD, Orb=OrbEl, ParamVec=_parVec, index=_index_R, max_iter=max_iteration, stepsize=stepsize, kwargs=kwargs)
            x = returnCombinedError(SD, _FD, [_index_R, _index_V], redshiftCorr=useGravRedCorr)

            if usePbar:
                if args[0]:
                    ProgressBar(GLOB_counter, 8000, "chi2= " + str(x))
                else:
                    NoProgressBar(GLOB_counter, "chi2= " + str(x))

            return x

        # dont calculate orbit, return constant error
        else:
            if usePbar:
                if args[0]:
                    ProgressBar(GLOB_counter, 8000, "chi2= 1E10")
                else:
                    NoProgressBar(GLOB_counter, "chi2= 1E10")
            return 1E10


    # Kepler solution, global, 1E-8 -- chosen if no parameters are set
    #parVec = np.array([ 4.26060187e+06, -2.98146568e-04,  7.21594511e-03, -5.38160820e-03, -4.51226416e+02,  1.62323029e+02, -4.23509314e+02,  8.38337634e+03])
    
    # use first guess parameter vector
    if kwargs['method'] == 'None':
        pass

    # use random parameter vector
    if kwargs['method'] == 'Random':
        Mass_0 = np.random.uniform(CSRT_M_min, CSRT_M_max)
        Distance_0 = np.random.uniform(CSRT_R_min, CSRT_R_max)
        r0 = PosRadToReal( np.array( [ SD.RA[_index_R], SD.DE[_index_R], 0 ] ), Distance_0 )
        v0 = np.array(
            [ (SD.RA[_index_R]-SD.RA[_index_R-1])*Distance_0*GLOB_PcYrKmS*GLOB_masToRad / (SD.TimeP[_index_R]-SD.TimeP[_index_R-1]),
            (SD.DE[_index_R]-SD.DE[_index_R-1])*Distance_0*GLOB_PcYrKmS*GLOB_masToRad / (SD.TimeP[_index_R]-SD.TimeP[_index_R-1]),
            SD.VR[_index_V] ] )
        parVec   = np.array([ Mass_0, *r0, *v0, Distance_0 ])

    # Kepler solution, 1E-8
    if stepsize == 1E-8 and kwargs['method'] == 'Newton':
        parVec = np.array([ 4.26175122e+06, -2.95254493e-04,  7.21524671e-03, -5.38486006e-03, -4.51353063e+02,  1.63648795e+02, -4.21806411e+02,  8.38370406e+03])

    #Kepler solution, 1E-9
    if stepsize == 1E-9 and kwargs['method'] == 'Newton':
        parVec = np.array( [4.26180655e+06, -2.96184228e-04,  7.19338917e-03, -5.38020387e-03, -4.51561607e+02,  1.62968775e+02, -4.24584527e+02,  8.35915945e+03] )

    # Schwarzschild solution, with grav red, 1E-9
    if stepsize == 1E-9 and kwargs['method'] == 'Schwarz':
        parVec = np.array([ 4.42965827e+06, -3.01141313e-04,  7.29306129e-03, -5.48739705e-03, -4.55624852e+02,  1.64210719e+02, -4.28133758e+02,  8.47566930e+03])

    #------------------------------------------------------------------------
    #   Fit Options

    shouldFit = 0
    if 'Fit' in kwargs.keys():
        if kwargs['Fit'] == 'Local':
            shouldFit = 1
        if kwargs['Fit'] == 'Full':
            shouldFit = 2


    t0 = time.process_time()

    # local fit only
    if shouldFit == 1:
        if usePbar:
            print("Start Local Fit\n")
        RESULT = optimize.minimize(_tFitFunction, x0=parVec, args=(True,), method='Powell', options={'disp':False})
        if usePbar:
            print("\n")
            print("[%s] - Message: %s\nResult: %s\ncurrent function val: %s" % (RESULT.success, RESULT.message, RESULT.x, RESULT.fun))

        t0 = time.process_time() - t0
        print("Done in %ss, nit = %s, delT = %ss" % (round(t0, 3), RESULT.nfev, round(t0/RESULT.nfev,3)))
        parVec = RESULT.x.tolist()

    # Global Fit
    if shouldFit == 2:
        if usePbar:
            print("Start Global Fit\n")
        RESULT = optimize.dual_annealing(_tFitFunction, args=(False,), bounds=BOUNDS, initial_temp=5230, maxfun=1E5, maxiter=250, local_search_options={"method": "Powell"})
        if usePbar:
            print("\n")
            print(RESULT)
        parVec = RESULT.x.tolist()

        # reset counter
        GLOB_counter = 0

        RESULT = optimize.minimize(_tFitFunction, x0=parVec, args=(True,), method='Powell', options={'disp':False})
        if usePbar:
            print("\n")
            print("[%s] - Message: %s\nResult: %s\ncurrent function val: %s" % (RESULT.success, RESULT.message, RESULT.x, RESULT.fun))

        t0 = time.process_time() - t0
        print("Done in %ss, nit = %s, delT = %ss" % (round(t0, 3), RESULT.nfev, round(t0/RESULT.nfev,3)))
        parVec = RESULT.x.tolist()

    #------------------------------------------------------------------------
    #   Return

    OrbEl = getOrbitalElements(parVec)
    NewFitData = getOrbit(SD=SD, Orb=OrbEl, ParamVec=parVec, index=_index_R, max_iter=max_iteration, stepsize=stepsize, kwargs=kwargs)
    _Err = returnCombinedError(SD, NewFitData, [_index_R, _index_V], redshiftCorr=useGravRedCorr)


    return NewFitData, [_index_R, _index_V]


def FitDataStandalone(_starNr:int, kwargs:dict={}) -> Tuple[FitContainer, StarContainer, list]:
    """
    Standalone Routine for fitting the orbit of a specified star.
    can display plot, intended for fitting and plotting from beginning
    Parameters
    ----------
    _starNr : [int] Number of the Star to be fitted
    _fig : Reference to plt figure for plotting
    Options
    -------
    method : Newton, Schwarz -> Potential to use
    grav-red : bool -> use Gravitational Redshift correction
    Fit : None, Local, Full -> Fit Options
    Stepsize : float -> overwrite default stepsize of 1E-8
    """

    #read complete data from file
    Data = readTable(fileName)
    #return Data for Star S-NUMBER
    S_Data = return_StarExistingData(Data, _starNr)
    #Star Data Container
    SD = StarContainer(_starNr, S_Data)

    FD, selIndex = FitDataInner(SD, kwargs=kwargs)

    
    # when using gravitational redshift correction, update star data radial velocity data
    if 'grav-red' in kwargs.keys() and 'Fit' in kwargs.keys():
        if kwargs['grav-red']: # and (kwargs['Fit'] == 'Local' or kwargs['Fit'] == 'Full'):
            
            _eT = SD.TimeP[selIndex[0]] - SD.TimeP[0] + FD.OrbitNumber * FD.OrbElem.Period
            fakeTimeline = np.linspace(0,_eT, len(FD.VPath))
            j = 0
            rTime = SD.TimeV - SD.TimeP[0]
            LengthAtVR = np.empty(len(rTime))

            for i in range(len(SD.TimeV)):
                for k in range(j, len(fakeTimeline)):
                    if (fakeTimeline[k] >= (rTime)[i]):
                        #newVR_Timeline[i] = fakeTimeline[k]
                        LengthAtVR[i] = np.linalg.norm(FD.PositionArray[k])# FD.PositionArray[k][2] #
                        j = k
                        break

            PN_VR = SD.VR - getGravRedshift(FD.Mass, LengthAtVR)
            SD.VR = PN_VR
    

    return FD, SD, selIndex


def genMCD(SD:StarContainer, iter:int, kwargs:dict={}):
    """
    Calculates new Parameters after variating Star Data adn writes them to file
    Parameters
    ----------
    SD : DataContainer
        Star Data in question
    iter : int
        Total Fits to compute and use for Errorbar calculation
    """

    FileToWriteTo = outputFile
    if 'File' in kwargs.keys():
        FileToWriteTo = kwargs['File']

    if 'UseSGRA_Pos' in kwargs.keys():
        useSGRA_Pos = kwargs['UseSGRA_Pos']
    else:
        useSGRA_Pos = False


    #------------------------------------------------------------------------
    #   MAIN LOOP

    # main loop, generating new data and fitting to get a list for every parameter
    for curIt in range(iter):
        # generate new points within 1 sigma of error
        newRA = generateMCData(SD.RA, SD.eRA)
        newDE = generateMCData(SD.DE, SD.eDE)
        newVR = generateMCData(SD.VR, SD.eVR)

        print(SD.RA)
        print(newRA)
        print(SD.DE)
        print(newDE)

        # create copy of Star Data and overrite the points
        NewSD = SD.copy()
        NewSD.RA = newRA
        NewSD.DE = newDE
        NewSD.VR = newVR 

        # generate new position of sgr a* if needed, use global bounds
        if useSGRA_Pos:
            #local best fit 
            NewSGRA_Pos = np.random.normal(0,1,3) * GLOB_SGRA_Pos
        else:
            NewSGRA_Pos = np.array([0,0,0])

        # change the position of sgr a*
        kwargs['varSGRA'] = NewSGRA_Pos

        if DEBUG_MODE:
            print("New Position of SGR A*: ", NewSGRA_Pos)

        # Fit the new Star Data
        print("\n")
        print('-'*25 + "Starting new Fit (%s/%s)" % (curIt+1, iter))

        newFD, _ = FitDataInner(NewSD, kwargs=kwargs)
        _tParVec = newFD.returnParVec()
   
        # write data to file
        f = open(FileToWriteTo, "a")
        for j in range(len(_tParVec)):
            f.write(str(_tParVec[j]) + " ")
        f.write("\n")
        f.close()

    print("\nDone!\n")


def genMCD_MP(SD:StarContainer, pid:int, kwargs:dict={}):
    """
    Calculates new Parameters after variating Star Data adn writes them to file
    Parameters
    ----------
    SD : DataContainer
        Star Data in question
    iter : int
        Total Fits to compute and use for Errorbar calculation
    """

    FileToWriteTo = outputFile
    if 'File' in kwargs.keys():
        FileToWriteTo = kwargs['File']

    if 'UseSGRA_Pos' in kwargs.keys():
        useSGRA_Pos = kwargs['UseSGRA_Pos']
    else:
        useSGRA_Pos = False

    #------------------------------------------------------------------------
    #   MAIN LOOP

    # generate new points within 1 sigma of error
    newRA = generateMCData(SD.RA, SD.eRA)
    newDE = generateMCData(SD.DE, SD.eDE)
    newVR = generateMCData(SD.VR, SD.eVR)

    # create copy of Star Data and overrite the points
    NewSD = SD.copy()
    NewSD.RA = newRA
    NewSD.DE = newDE
    NewSD.VR = newVR 

    if useSGRA_Pos:
        NewSGRA_Pos = np.random.normal(0, 1, 3) * GLOB_SGRA_Pos
        #NewSGRA_Pos = PosRadToReal(NewSGRA_Pos, _tDist)
    else:
        NewSGRA_Pos = np.array([0,0,0])

    # still in mas
    kwargs['varSGRA'] = NewSGRA_Pos

    # Fit the new Star Data
    print('-'*25 + "Starting new Fit (%s)" % (pid))

    newFD, _ = FitDataInner(NewSD, kwargs=kwargs)
    _tParVec = newFD.returnParVec()

    # write data to file
    f = open(FileToWriteTo, "a")
    for j in range(len(_tParVec)):
        f.write(str(_tParVec[j]) + " ")
    f.write("\n")
    f.close()

    print("%s, Done!" % (pid))


def evaMCD(_fig, file:str):
    """
    Evaluates Parameters written to file and calculates mean and std values for every parameter and prints them out
    """

    print(file)

    f = open(file, 'r')

    lines = f.readlines()
    h= []

    for i in range(len(lines)):
        _t = lines[i].strip()
        _t = _t.split(" ")
        _t = [float(x) for x in _t]
        h.append(_t)

    mean = []
    std = []

    g = []
    histData = []
    histName = [r'$M$ [$10^6 M_\odot$]', r'$R$ [kpc]', r'$e$ [1]', r'$a$ [$10^{-3}$pc]', r'$i$ [$^\circ$]', r'$T$ [yr]']
    kartPos = []
    N = ["Mass", "R", "e", "a", "i", "LAN", "argPeri", "MeanM", "T", "True Anomaly"]

    print("-"*75)

    for i in range(len(h)):

        OE = getOrbitalElements(h[i])
        #            Mass,   Distance,  e,       a,          i,      Omega,  omega,        M,         T,   True Anomaly
        g.append( [ h[i][0], h[i][7], OE.Ecc, OE.MayAxis, OE.Incl, OE.LAN, OE.ArgPeri, OE.MeanM, OE.Period, OE.TAnom ] )
        histData.append([ h[i][0]/1E6, h[i][7]/1E3, OE.Ecc, OE.MayAxis, OE.Incl, OE.Period ])
        # position
        kartPos.append( [ h[i][1], h[i][2], h[i][3], h[i][4], h[i][5], h[i][6] ] )


    for j in range(len(g[0])):
        ParDat = [g[i][j] for i in range(len(g))]
        mean.append( np.mean( ParDat ) )
        std.append( np.std( ParDat ) )
        print("MCD: ", N[j], ", mean= ", mean[j], "; std= ", std[j])

    print("Length of data: ", len(g))

    
    '''
    mean = []
    std = []
    N = ["x", "y", "z", "vx", "vy", "vz"]

    for j in range(len(kartPos[0])):
        ParDat = [kartPos[i][j] for i in range(len(kartPos))]
        mean.append( np.mean( ParDat ) )
        std.append( np.std( ParDat ) )
        print("MCD: ", N[j], ", mean= ", mean[j], "; std= ", std[j])

    '''

    print("-"*75)

    _fig.clf()

    mean = []
    std = []

    for j in range(len(histData[0])):
        ParDat = [histData[i][j] for i in range(len(histData))]
        mean.append( np.mean( ParDat ) )
        std.append( np.std( ParDat ) )
    
    mean[3] *= 1E3
    std[3] *= 1E3
    for l in range(len(histData)):
        histData[l][3] *= 1E3

    for x in range(6):
        for y in range(6):
            if y <= x:
                _tf = _fig.add_subplot(6,6, 6*x + y + 1)
                _tf.grid(False)
                _tf.set_aspect('auto')
                #_tf.set_xlabel(histName[y])
                #_tf.set_ylabel(histName[x])

                if y != 0 or x == 0:
                    plt.yticks([])
                else:
                    pass
                    plt.yticks( [4.23,4.275,8.35,8.40,0.881,0.884,4.993,5.013,44.3,44.8,16.03,16.11],
                                [4.23,4.275,8.35,8.40,0.881,0.884,4.993,5.013,44.3,44.8,16.03,16.11], size=8)
                    plt.yticks(size=8)

                if x != 5 or y == 5:
                    plt.xticks([])
                else:
                    pass
                    plt.xticks( [4.23,4.275,8.35,8.40,0.881,0.884,4.993,5.013,44.3,44.8,16.03,16.11],
                                [4.23,4.275,8.35,8.40,0.881,0.884,4.993,5.013,44.3,44.8,16.03,16.11], rotation=90, size=8)
                    #plt.xticks(rotation=90, size=8)
                
                
                if y == x:
                    _t = [histData[i][x] for i in range(len(histData))]
                    plt.xlim(mean[x]-3*std[x],mean[x]+3*std[x])
                    _tf.hist(_t, bins=50)
                    plt.axvline(mean[x], color='black', linestyle='dashed')
                    plt.figtext(0.165 + 0.8/6 * x, 0.91 - 0.8/6 * x, round(mean[x], 3), ha='center', size=11 )

                else:
                    _x = [histData[i][y] for i in range(len(histData))]
                    _y = [histData[i][x] for i in range(len(histData))]
                    _t = _tf.hist2d(_x, _y, bins=(20,20), cmap=cm.jet)
    


    #_fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0, hspace=0)

    for i in range(len(histName)):
        # horizontal at bottom
        plt.figtext(0.15 + 0.825/6 * i, 0.02, histName[i], {'ha':'center', 'size':9})
        # vertical at left
        plt.figtext(0.01, 0.15 + 0.825/6 * (5-i), histName[i], {'ha':'left', 'size':9})


def DrawChi2Slice(_fig, SD:StarContainer, parVec:list, bounds:list, IndexList:list, _dim:int=50, kwargs:dict={}):
    """
    Draws chi2 distribution for M and R
    Parameters
    ----------
    SD : DataContainer
        Star Data
    parVec : list
        Current parameter vecotr, preferably from minimum
    varyIndex : list
        index of parameter to be varied, needs to be length 2
    bounds : list
        bounds for both variables
    """



    dim = _dim
    chi2 = np.zeros((dim, dim))

    Mvar = np.linspace(bounds[0][0], bounds[0][1], dim)
    Rvar = np.linspace(bounds[1][0], bounds[1][1], dim)

    iter = 0

    
    _chi2File = open("SMBH/Data/dump/chi2Slice.txt", 'r')

    lines = _chi2File.readlines()
    h= []

    for i in range(len(lines)):
        _t = lines[i].strip()
        _t = _t.split(" ")
        _t = [float(x) for x in _t]
        h.append(_t)

    chi2 = np.array(h)

    miny_i = []
    for i in range(100):
        _l = np.amin(chi2[i])
        _m = np.argwhere(chi2[i] == _l).flatten()
        miny_i.append(Mvar[_m])

    minx_i = []
    for j in range(100):
        _m = np.amin(chi2[:,j])
        _n = np.argwhere(chi2[:,j] == _m).flatten()
        minx_i.append(Rvar[_n])

    
    midy_i = []
    for i in range(100):
        _l = np.amin(chi2[i])
        _m = np.argwhere(chi2[i] <= _l+1).flatten()[0]
        midy_i.append(Mvar[_m])

    midx_i = []
    for j in range(100):
        _m = np.amin(chi2[:,j])
        _n = np.argwhere(chi2[:,j] <= _m+1).flatten()[0]
        midx_i.append(Rvar[_n])

    '''
    # Distance
    for i in range(dim):
        # Mass
        for j in range(dim):
            iter += 1
            newPar = parVec.copy()
            newPar[0]  = Mvar[j]*1E6 # mass
            newPar[-1] = Rvar[i]*1E3 # distance
            
            OrbEl = getOrbitalElements(newPar)
            if (OrbEl.Period > 0 and OrbEl.Ecc < 1 and OrbEl.Period <= 20):
                _FD = getOrbit(SD=SD, Orb=OrbEl, ParamVec=newPar, index=IndexList[0], max_iter=500000, stepsize=1E-8, kwargs=kwargs)
                x = returnCombinedError(SD, _FD, IndexList)
                chi2[i][j] = x
                ProgressBar(iter, dim**2, "chi2= " + str(x))
            else:
                ProgressBar(iter, dim**2, "chi2= 1E10")
                chi2[i][j] = 1E10

    print("\nDone!")

    '''

    '''
    _chi2File = open("SMBH/Data/dump/chi2Slice.txt", "w")

    
    for i in range(dim):
        for j in range(dim):
            _chi2File.write( str(chi2[i][j]) + " " )
        _chi2File.write("\n")
    _chi2File.close()
    '''

    _min = np.argwhere(chi2 == np.amin(chi2)).flatten()
    _minValue = [ Mvar[_min[1]], Rvar[_min[0]] ]

    maxval = np.amax(chi2)
    minval = np.amin(chi2)

    levels = np.geomspace(minval,maxval, 25)

    _fig.clf()
    _tf = _fig.add_subplot(1,1,1)
    _tf.grid(False)
    _tf.set_aspect('auto')

    _tf.set_xlabel(r'$R_0$ [kpc]', fontdict={'size':13})
    _tf.set_ylabel(r'$M$ [$10^6 M_\odot$]', fontdict={'size':13})
    
    xgit,ygit = np.meshgrid(Rvar, Mvar)
    ax = _tf.contourf(xgit.T, ygit.T, chi2, cmap=cm.get_cmap('viridis'), levels=levels)
    _fig.colorbar(ax)

    _label = r"Min: $M$ [$10^6 M_\odot$]="+str(np.round(_minValue[0],2)) + r", $R_0$ [kpc]=" + str(np.round(_minValue[1],2))
    _tf.scatter(_minValue[1], _minValue[0], label=_label, color='red', s=5)
    _tf.plot(Rvar,miny_i,color='blue', label='min line')
    print(_minValue)

    _tf.legend(loc='best')


def determineDeltaOmega(FD:FitContainer) -> list:
    
    startPos = ParVecToParams(FD.returnParVec())[1]

    StartEr = startPos / np.linalg.norm(startPos)

    #dotList = np.abs( np.dot( FD.PositionArray / np.linalg.norm(FD.PositionArray), StartEr ) )
    dotList = np.empty(len(FD.PositionArray))
    for i in range(len(FD.PositionArray)):
        dot = FD.PositionArray[i] / np.linalg.norm(FD.PositionArray[i])
        dotList[i] = ( np.abs( np.dot(dot, StartEr) ) )

    xmin = np.argwhere(dotList <= 1E-2).flatten()
    print(xmin)

    miniVal = 9999999999
    miniIndex = -1

    for i in range(len(xmin)):
        x = FD.PositionArray[i]
        #print(x)
        if  not np.array_equal(x,startPos):
            if np.dot(x,startPos) <= miniVal:
                miniVal = np.dot(x/np.linalg.norm(x),StartEr)
                miniIndex = i

    if miniIndex >= 0:
        OriginOE = getOrbitalElements(FD.returnParVec())

        newParVec = [FD.Mass, *FD.PositionArray[miniIndex], *FD.VelocityArray[miniIndex], FD.Distance]
        NewOE = getOrbitalElements(newParVec)

        GR_Const = 6*np.pi*GLOB_G*FD.Mass / GLOB_c**2
        GR_Predict = GR_Const / (OriginOE.MayAxis * (1 - OriginOE.Ecc**2))
        GR_Predict_Degree = GR_Predict * 180 / np.pi

        DeltaSim = np.abs(OriginOE.ArgPeri - NewOE.ArgPeri)
        
        print("GR predicted [degr]: ", GR_Predict_Degree)
        print("  Simulation [degr]: ", DeltaSim)
        print("  Difference [degr]: ", GR_Predict_Degree - DeltaSim)

        oldParVec = FD.returnParVec()
        
        oldParVec[1] += 2.587E-6
        oldParVec[2] += 4.625E-6
        oldParVec[3] += 6.79E-6
        oldParVec[4] += 0.612
        oldParVec[5] += 0.451
        oldParVec[6] += 0.528

        DeltaOld = getOrbitalElements(oldParVec)
        print("delta old plus: ", DeltaOld.ArgPeri)

        oldParVec = FD.returnParVec()
        
        oldParVec[1] -= 2.587E-6
        oldParVec[2] -= 4.625E-6
        oldParVec[3] -= 6.79E-6
        oldParVec[4] -= 0.612
        oldParVec[5] -= 0.451
        oldParVec[6] -= 0.528

        DeltaOld = getOrbitalElements(oldParVec)
        print("delta old minus: ", DeltaOld.ArgPeri)

        print("original: ", OriginOE.ArgPeri)
        print("new: ", NewOE.ArgPeri)

        #print("max err: ", )
        #R = np.array([2.587E-6, 4.625E-6, 6.79E-6])
        #V = np.array([0.612,0.451,0.528])
        #Ez = np.array([0,0,1])

        #h = np.cross(R, V)
        #e = np.cross(V, h) / (GLOB_G*4.42965544e+06) - R/np.linalg.norm(R)

        #n = np.cross(Ez, h)

        #omega = np.arccos( np.dot(n, e)/ (np.linalg.norm(n) * np.linalg.norm(e)) )
        #print(omega)

    else:
        print("found no minimal point!")


def FindChiError(_fig, FD:FitContainer, SD:StarContainer, selIn:list, OPT:dict, start:float, stop:float, step:int, index:int):
    """
    Plot a series of Chi^2 points for variation of Parameter [index] from [start] to [stop] in [step] steps.

    Used to find the 1-sigma error
    """
    x = np.linspace(start,stop,step)
    diffArr = []
    OverArr = []
    ParIndex = index

    # data of minimum
    Origin = FD.returnParVec()
    orChi2 = returnSpecificChi2Point(SD, Origin, selIn, OPT)

    for i in range(len(x)):
        dP = Origin.copy()
        dP[index] += x[i]
        # chi^2 of new point
        dPchi2 = returnSpecificChi2Point(SD, dP, selIndex, OPT)
        limit = np.sqrt(2*orChi2) # sqrt(2*334)
        diff = dPchi2 - orChi2
        if diff > limit:
            OverArr.append(diff)
        else:
            diffArr.append(diff)

        print(i+1, diff)


    _tf = _fig.add_subplot(1,1,1)
    _tf.grid(True)
    _tf.set_xlabel("Parameter")
    _tf.set_ylabel("chi^2")

    _tf.scatter(x[:len(diffArr)],diffArr, label='points under')
    _tf.scatter(x[len(diffArr):],OverArr, label='points over')
    _tf.plot([x[0], x[-1]], [limit, limit], color='red')


def getMinMaxOE(dM:float, dD:float, dR:np.ndarray, dV:np.ndarray, FD:FitContainer):
    """
    prints out the Error on the Orbital Elements given the Error on the ParVec and the FitData
    """

    Origin = FD.returnParVec()
    OriginOE = getOrbitalElements(Origin)

    n = [-1,1]
    OEList = []

    
    _Names = ["e", "a", "i", "LAN", "ArgPeri", "MeanM", "T", "TAnom"]
    _OriginPar = [ OriginOE.Ecc, OriginOE.MayAxis, OriginOE.Incl, OriginOE.LAN, OriginOE.ArgPeri, OriginOE.MeanM, OriginOE.Period, OriginOE.TAnom ]
    _max = []
    _min = []

    for m in n:
        for d in n:
            for r in n:
                for v in n:
                    _t = minFD.returnParVec().copy()
                    deltaParVec = [m*dM, *r*dR, *v*dV, d*dD]
                    for i in range(len(_t)):
                        deltaParVec[i] += _t[i]
                        _tOE = getOrbitalElements(deltaParVec)
                        _h = [ _tOE.Ecc, _tOE.MayAxis, _tOE.Incl, _tOE.LAN, _tOE.ArgPeri, _tOE.MeanM, _tOE.Period, _tOE.TAnom ]
                    OEList.append( _h )

    for i in range(len(OEList[0])):
        _list = [ OEList[j][i] for j in range(len(OEList)) ]
        _max.append( np.amax(_list) )
        _min.append( np.amin(_list) )
        #print(_Names[i] + " max: ", _OriginPar[i] - _max[i])
        #print(_Names[i] + " min: ", _OriginPar[i] - _min[i])
        print(_Names[i] + " mean: ", (np.abs(_OriginPar[i] - _min[i]) + np.abs(_OriginPar[i] - _max[i]))/2 )


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       PROGRAM START
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    mp.freeze_support()
    pool = mp.Pool()
    MAIN_fig = plt.figure(figsize=(8,8), dpi=100, clear=True)

    """
    ||------------|| OPTIONS ||------------||

    method          str         Newton, Schwarz, None, Random
    grav-red        bool        use gravitational redshift?
    Fit             str         None, Local, Full
    Stepsize        int         stepsize to use in integrator
    Pbar            bool        use progressbar? disabled for multi processing
    File            str         Output File for Ensemble Test
    UseSGRA_Pos     bool        variate position of Sgr A*?
    
    """

    #------------------------------------------------------------------------------------------

    # Options used for Initial Orbit Calculation
    OPTIONS = {
        'method': 'Newton', 
        'grav-red': False, 
        'Fit': 'None', 
        'Stepsize': 1E-8,
        'Pbar': True
        }

    #------------------------------------------------------------------------------------------

    # Initial Fit, return FitData at minimum, without fitting
    minFD, _SD, selIndex = FitDataStandalone(_starNr=2, kwargs=OPTIONS)

    print("min chi2 = ", minFD.getChi2(bReduced=False))

    #determineDeltaOmega(minFD)

    #------------------------------------------------------------------------------------------

    # Options used for MC and Chi2 Error Calculation
    OPTIONS = {
        'method': 'Newton', 
        'grav-red': False, 
        'Fit': 'Local', 
        'Stepsize': 1E-9,
        'File': "SMBH/Data/dump/OutputE8wSGRA.txt",
        'Pbar': False,
        'UseSGRA_Pos': True
        }

    #------------------------------------------------------------------------------------------
    
    # find Error for Parameter
    #FindChiError(MAIN_fig, minFD, _SD, selIndex, OPTIONS, start=10, stop=10000, step=10, index=0)

    #------------------------------------------------------------------------------------------

    '''
    # kepler: 11300
    dM = 11500
    # kepler: 10.56
    dD = 11
    # kepler: np.array([0.96E-5, 0.6E-5, 1.45E-5])
    dR = np.array([0.97E-5, 0.55E-5, 1.4E-5])
    # kepler: np.array([2.25, 1.67, 4.31])
    dV = np.array([1.9, 2.02, 4.1])

    # Get Error on Orbital Elements
    getMinMaxOE(dM, dD, dR, dV, minFD)
    '''
    
    #------------------------------------------------------------------------------------------

    # calculate Errorbars for every parameter via MC method
    #genMCD(_SD, 2, kwargs=OPTIONS)
    #evaMCD(MAIN_fig, "SMBH/Data/dump/OutputE8wSGRA.txt")

    # multi processing
    # x = [ pool.apply_async(genMCD_MP, args=(_SD, i, OPTIONS))  for i in range(9000) ]
    # pool.close()
    # pool.join()
    
    #------------------------------------------------------------------------------------------

    # Show Plots
    plot2Ways(_fig=MAIN_fig, SD=_SD, FD=minFD, _in=selIndex)           # plot the 2 views at once
    #plotDataAndFit(MAIN_fig, _SD, FD=minFD)                    # plot just the orbit in big


    #plt.savefig("SMBH/Data/dump/E8Hist.png", transparent=True)
    plt.show()