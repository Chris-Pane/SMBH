import sys
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from scipy import optimize
import multiprocessing as mp
import time


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       FUNCTIONS AND VARS
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------


# Display additional info for debugging
DEBUG_MODE = False
# File with all the data
fileName = "SMBH/Data/OrbitData2017.txt"
# Output file for MCD
outputFile = "SMBH/Data/Output.txt"
# store last orbit data in file
OrbitFileForewrd = "SMBH/Orbit/foreward.txt"
OrbitFileBackwrd = "SMBH/Orbit/backward.txt"
# Gravitational Constant in astronomy Units
GLOB_G = 4.30091E-3
# pc/yr to km/s conversion factor
GLOB_PcYrKmS = 997813.106
# mas to radians conversion factor
GLOB_masToRad = 4.8481368110954E-9
# 1 second to year conversion factor
GLOB_SecToYr = 3.17098E-8
# pc to km conversion factor
GLOB_PcToKm = 3.086E13
# speed of light in km/s
GLOB_c = 299792.458
#counter
GLOB_counter = 0
# Global Bounds of SGR A* Position in mas -- (del x, del y , del z)
GLOB_SGRA_Pos = np.array([0.2, 0.2, 0.2])


#   IMPORT AND DATA HANDLING

class FitContainer():
    '''
    Container for all the Data needed for a Fit
    '''
    success = False
    Mass = -1
    Distance = -1
    initR = None
    initV = None
    PosPath = None
    VPath = None
    ErrRA = None
    ErrDE = None
    ErrVR = None
    OrbitNumber = 0
    OrbElem = None
    PositionArray = None
    VelocityArray = None
    
    def __init__(self, ParVec, _oN = 1, success = True, _orb=None, _PosArr = None, _VelArr = None):
        Pars                = ParVecToParams(ParVec)
        self.Mass           = Pars[0]
        self.Distance       = Pars[3]
        self.initR          = Pars[1]
        self.initV          = Pars[2]
        self.OrbitNumber    = _oN # number of orbits infront of index data
        self.success        = success
        self.OrbElem        = _orb
        self.PositionArray  = _PosArr
        self.VelocityArray  = _VelArr

        if type(_PosArr) == np.ndarray:
            _tmp            = PosRealToRad(self.PositionArray, self.Distance)
            self.PosPath    = np.array( [ _tmp[:,0], _tmp[:,1] ] )
            self.VPath      = np.array( self.VelocityArray[:,2] )

    def initErrorData(self, _RA, _DE, _VR):
        self.ErrRA = _RA
        self.ErrDE = _DE
        self.ErrVR = _VR

    def getChi2(self, bReduced:bool=False) -> float:
        if type(self.ErrRA) == list:
            lenPos = len(self.ErrRA[3])
            lenVel = len(self.ErrVR[3])
            NPos = lenPos / (lenPos + lenVel)
            NVel = lenVel / (lenPos + lenVel)
            if bReduced:
                _t = (NPos*self.ErrRA[2] + NPos*self.ErrDE[2] + NVel*self.ErrVR[2])
                #_t = (self.ErrRA[2] + self.ErrDE[2] + self.ErrVR[2])
                return _t / (2*lenPos + lenVel)
            else:
                return (NPos*self.ErrRA[2] + NPos*self.ErrDE[2] + NVel*self.ErrVR[2])
                #return (self.ErrRA[2] + self.ErrDE[2] + self.ErrVR[2])
        else:
            return 1E10

    def returnParVec(self) -> list:
        return [self.Mass, *self.initR, *self.initV, self.Distance]


class DataContainer():
    '''
    Stores the Data from a Star
    '''
    TimeP   = None
    TimeV   = None
    RA      = None
    eRA     = None
    DE      = None
    eDE     = None
    VR      = None
    eVR     = None

    def __init__(self, _stNr, _SData):
        '''
        _stNr -- Star Number

        _SData -- Star Data, already extracted from Total Data (see readTable)
        '''
        if _SData:
            self.StarNr = _stNr
            scale = np.sqrt(5)/2.60335    # for weighted
            #scale = np.sqrt(5)/2.192    # for unweighted

            #positional data
            self.TimeP  = np.array([x["time"] for x in _SData["pos"]]   )
            self.RA     = np.array([x["RA"] for x in _SData["pos"]]     )
            self.DE     = np.array([x["DE"] for x in _SData["pos"]]     )
            self.eRA    = np.array([x["e_RA"] * scale for x in _SData["pos"]]   )
            self.eDE    = np.array([x["e_DE"] * scale for x in _SData["pos"]]   )

            #RVel data
            self.VR     = np.array([x["RVel"] for x in _SData["rad"]]   )
            self.eVR    = np.array([x["e_RVel"] * scale for x in _SData["rad"]] )
            self.TimeV  = np.array([x["time"] for x in _SData["rad"]]   )

            #print("len of Data N = ", len(self.RA) + len(self.DE) + len(self.VR))

    # create and return a copy of this object
    def copy(self):
        _t = DataContainer(self.StarNr, None)
        _t.TimeP    = self.TimeP
        _t.RA       = self.RA
        _t.DE       = self.DE
        _t.eRA      = self.eRA
        _t.eDE      = self.eDE
        _t.VR       = self.VR
        _t.eVR      = self.eVR
        _t.TimeV    = self.TimeV
        return _t


class OrbitElem():
    def __init__(self, _a, _e, _omega, _LAN, _i, _M, _T, _nu):
        """
        Container for saving orbital Elements
        Parameters
        ----------
        _a : semi mayor axis
        _e : eccentricity
        _omega : argument of periapsis
        _LAN : longitude of ascending node
        _i : inclination
        _M : Mean anomaly
        _T : Period
        """
        self.MayAxis = _a
        self.Ecc = _e
        self.ArgPeri = _omega
        self.LAN = _LAN
        self.Incl = _i
        self.MeanM = _M
        self.Period = _T
        self.TAnom = _nu


def readTable(_fName:str) -> tuple:
    '''
    reads the file. Returns (Data in rows, Header in the form [1st data, 2nd data,...])

    Format:    (Data[row][point], Header[point][more info])
    '''

    _file = open(_fName, 'r')
    _data = _file.readlines()
    _file.close()

    # this is where the header starts
    _index = 10
    _data_header = []


    #------------------------------------------------
    # read the header
    #------------------------------------------------
    while True:
        if _data[_index][0] == "-":
            break

        #this is a dataline
        _line = _data[_index]

        _byte = int(_line[0:4].strip())
        _byte_end = int(_line[5:8].strip())
        _format = _line[9:15].strip()
        _unit = _line[16:23].strip()
        _label = _line[24:33].strip()
        _desc = _line[34:].strip()
        
        _data_header.append([])
        _data_header[_index - 10].append(_byte)
        _data_header[_index - 10].append(_byte_end)
        _data_header[_index - 10].append(_format)
        _data_header[_index - 10].append(_unit)
        _data_header[_index - 10].append(_label)
        _data_header[_index - 10].append(_desc)

        _index += 1

    # this is where the data starts
    _index = 134
    _acData = []
    
    #------------------------------------------------
    # read the data
    #------------------------------------------------
    while True:

        #file end
        if not _data[_index]:
            break

        _line = _data[_index]
        _acData.append([])
        for i in range(len(_data_header)):
            _acData[_index-134].append(_line[ _data_header[i][0] - 1: _data_header[i][1] ].strip() )

        if _index+1 < len(_data):
            _index += 1
        else:
            break
        
    return (_acData, _data_header)


def EmptyCheck(_data:list, _index:int, _indexList:list) -> bool:
    """
    check if any element (index given by indexList) in Data is non zero

    return True if any element is non zero;    used for return_StarExistingData
    """

    for i in _indexList:
        if ( _data[0][_index][i] != '' ):
            return True
    return False


def return_StarExistingData(_data:list, StarNr:int) -> dict:
    """
    
    return data for specific Star

    IN: (raw_data, (int) Star Number)

    OUT: StarData

    FORMAT: [ Data["pos"], Data["rad"] ]

    Data["pos"]..."time", "RA", "e_RA", "DE", "e_DE"

    Data["rad"]..."time", "RVel", "e_RVel"

    """

    _header = _data[1]
    firstStarElem = "oRA-S" + str(StarNr)
    _index = -1
    for i in range(len(_header)):
        #is label the same
        if _header[i][4] == firstStarElem:
            _index = i
            break
    
    #wrong label => wrong star number
    if _index < 0:
        return []

    #_StarData = []

    #dictionary containing positional data and radial data seperately
    _StarData = dict( [ ("pos", []), ("rad", []) ] )
    #form a dict from the data
    #FORMAT:    time, RA, e_RA, DE, e_DE // RVel, e_RVel
    for i in range(len(_data[0])):
        #_data[0][i] is the i-th row of data; [1] is the flag position
        #check flag; a = position
        if (_data[0][i][1] == "a"):
            #is the star data not empty; _index is starting index of star data
            #check for all positional data
            if (EmptyCheck(_data, i, [ _index, _index+1, _index+2,_index+3 ] ) ):
                _StarData["pos"].append( dict( [ 
                    ("time", float(_data[0][i][0])),           #date
                    ("RA",   float(_data[0][i][_index])),      #Right ascention
                    ("e_RA", float(_data[0][i][_index+1])),    #Error RA
                    ("DE",   float(_data[0][i][_index+2])),    #Declination
                    ("e_DE", float(_data[0][i][_index+3]))     #Error DE
                     ] ) )


        #check if rad flag
        elif (_data[0][i][1] == "rv"):
            if (EmptyCheck(_data, i, [_index+4,_index+5] ) ):
                _StarData["rad"].append( dict( [
                    ("time",    float(_data[0][i][0])),        #date
                    ("RVel",    float(_data[0][i][_index+4])), #radial velocity
                    ("e_RVel",  float(_data[0][i][_index+5]))  #Error RVel

                ]))

    return _StarData

#   ORBITAL ELEMENTS

def getT(r0:np.ndarray, v0:np.ndarray, _M:float) -> float:
    '''
    returns the Period of one orbit for given state vector and Mass
    '''
    _a = 1/( 2/np.linalg.norm(r0) - np.linalg.norm(v0)**2/(GLOB_G * _M) )   # a in pc
    _t = 2*np.pi * np.sqrt( (_a**3)/(GLOB_G*_M) )                           # Units = sec * pc / km
    return _t * GLOB_SecToYr * GLOB_PcToKm                                  # convert to year plus additional length factor


def getOrbitalElements(_parVec:list) -> OrbitElem:
    """
    Returns all Orbital Elements and the Period, packed into a data class
    Parameters
    ----------
    _parVec : Parameter Vector for current orbit
    Returns
    -------
    OrbitalElem Object
    """

    Pars = ParVecToParams(_parVec)
    M = Pars[0]
    r0 = Pars[1]
    v0 = Pars[2]

    # momentum vector
    h = np.cross(r0, v0)
    # eccentricity vector
    e = np.cross(v0, h) / (GLOB_G*M) - r0/np.linalg.norm(r0)
    # eccentricity
    e_norm = np.linalg.norm(e)
    n = np.array( [-h[0], h[1], 0] )
    # true anomaly
    nu = np.arccos( np.dot(e, r0) / (e_norm * np.linalg.norm(r0)) )
    if np.dot(r0, v0) < 0:
        nu = 2*np.pi - nu
    # inclination
    i = np.arccos(h[2] / np.linalg.norm(h))
    # eccentric Anomaly
    E = 2* np.arctan( np.tan(nu/2) / ( np.sqrt( (1+e_norm)/(1-e_norm) ) ) )
    # LAN
    LAN = np.arccos( n[0] / np.linalg.norm(n) )
    if n[1] < 0:
        LAN = 2*np.pi - LAN
    # argument of periapsis
    omega = np.arccos( np.dot(n, e)/ (np.linalg.norm(n) * e_norm) )
    if e[2] < 0:
        omega = 2*np.pi - omega
    # mean anomaly
    MeanM = E - e_norm*np.sin(E)
    # semi mayor axis
    a = 1/( 2/np.linalg.norm(r0) - np.linalg.norm(v0)**2/(GLOB_G * M) )
    _t = 2*np.pi * np.sqrt( (np.clip(a, 0, a)**3)/(GLOB_G*M) )   # Units = sec * pc / km
    T = _t * GLOB_SecToYr * GLOB_PcToKm
    
    _OE = OrbitElem(a, e_norm, omega * 180 / np.pi, LAN * 180 / np.pi, i * 180 / np.pi, MeanM * 180 / np.pi, T, nu)

    return _OE


def OE_Essentials(_parVec:list) -> OrbitElem:
    """
    Only calculate e and T to be bounds checked for fit
    Parameters
    ----------
    _parVec : Parameter Vector for current orbit
    Returns
    -------
    OrbitalElem Object
    """

    Pars = ParVecToParams(_parVec)
    M = Pars[0]
    r0 = Pars[1]
    v0 = Pars[2]

    # momentum vector
    h = np.cross(r0, v0)
    # eccentricity vector
    e = np.cross(v0, h) / (GLOB_G*M) - r0/np.linalg.norm(r0)
    # eccentricity
    e_norm = np.linalg.norm(e)
    # semi mayor axis
    a = 1/( 2/np.linalg.norm(r0) - np.linalg.norm(v0)**2/(GLOB_G * M) )
    _t = 2*np.pi * np.sqrt( (np.clip(a, 0, a)**3)/(GLOB_G*M) )   # Units = sec * pc / km
    T = _t * GLOB_SecToYr * GLOB_PcToKm
    
    _OE = OrbitElem(a, e_norm, 0, 0, 0, 0, T, 0)

    return _OE

#   UTILITY

def PosRadToReal(_r:np.ndarray, _dist:float) -> np.ndarray:
    '''
    converts the first 2 radial elements to real distance, given the distance

    returns position vector in pc
    '''

    return _r*_dist*GLOB_masToRad


def RadToReal(_x:float, _dist:float) -> float:
    '''
    return distance in pc for one coordinate
    '''
    return _x*_dist*GLOB_masToRad


def PosRealToRad(_r:np.ndarray, _dist:float) -> np.ndarray:
    '''
    converts the real position to radial position in the first 2 elements.
    Used for plotting only

    returns postion vector with units ('','',pc)
    '''
    _t = np.array([ _dist*GLOB_masToRad, _dist*GLOB_masToRad, 1 ])
    return _r/_t


def potential(r:np.ndarray,v:np.ndarray,_M:float, r_SGRA:np.ndarray=np.array([0,0,0])) -> np.ndarray:
    """
    return Kepler acceleration
    Parameters
    ----------
    r : [vector]
        position of particle to evaluate potential at
    _M : [scalar]
        Mass of central object

    Returns
    -------
    Potential Strength
    """

    # true distance from star to srg a*
    dist = r - r_SGRA

    return -(GLOB_G*_M*dist) / (np.linalg.norm(dist)**3)


def potentialSchwarz(r:np.ndarray,v:np.ndarray,_M:float, r_SGRA:np.ndarray=np.array([0,0,0])) -> np.ndarray:
    """
    return the Schwarzschild acceleration
    Parameters
    ----------
    r : [vector]
        position of particle to evaluate potential at
    v : [vector]
        velocity of particle
    _M : [scalar]
        Mass of central object

    Returns
    -------
    Schwarzschild Potential Strength: Kepler Potential + a/r^3
    """

    h = np.cross(r,v)           # specific angular momentum
    kepl = potential(r,v,_M)
    Schw = (3 * GLOB_G * _M * np.dot(h,h) * r) / (GLOB_c**2 * np.linalg.norm(r)**5) 
    return kepl + Schw


def VerletStep(h:float,r0:np.ndarray,v0:np.ndarray,f0:np.ndarray,_M:float, r_SGRA:np.ndarray=np.array([0,0,0])) -> np.ndarray:
    """
    Orbital Integration using the Verlet Algorithm
    Parameters
    ----------
    h : [scalar]
        stepsize -> delta t
    r0 : [vector]
        position of particle from last evaluation
    v0 : [vector]
        velocity of particle from last evaluation
    f0 : [scalar]
        potential strength from last evaluation step
    _M : [scalar]
        Mass of central object
    func : [function]
        Potential function to evaluate

    Returns
    -------
    [r1, v1, f1]
        position, velocity and potential of new step
    """
    pp = np.add(v0, h/2*f0)                 # 1/2 Delta velocity
    r1 = np.add(r0, h*pp)                   # new position = r0 + del v*del t
    f1 = potential(r1,v0,_M, r_SGRA)                # new potential at new position
    v1 = np.add(pp, h/2*f1)                 # new velocity = v0 + 1/2 del a0*del t + 1/2 del a1*del t
    return np.array([r1,v1,f1])


def VerletStepSchwarz(h:float,r0:np.ndarray,v0:np.ndarray,f0:np.ndarray,_M:float, r_SGRA:np.ndarray=np.array([0,0,0])) -> np.ndarray:
    """
    Orbital Integration using the Verlet Algorithm
    Parameters
    ----------
    h : [scalar]
        stepsize -> delta t
    r0 : [vector]
        position of particle from last evaluation
    v0 : [vector]
        velocity of particle from last evaluation
    f0 : [scalar]
        potential strength from last evaluation step
    _M : [scalar]
        Mass of central object
    func : [function]
        Potential function to evaluate

    Returns
    -------
    [r1, v1, f1]
        position, velocity and potential of new step
    """
    pp = np.add(v0, h/2*f0)                 # 1/2 Delta velocity
    r1 = np.add(r0, h*pp)                   # new position = r0 + del v*del t
    f1 = potentialSchwarz(r1,pp,_M)             # new potential at new position
    v1 = np.add(pp, h/2*f1)                 # new velocity = v0 + 1/2 del a0*del t + 1/2 del a1*del t
    return np.array([r1,v1,f1])


def returnDataError(rData:np.ndarray, rDataErr:np.ndarray, rTime:np.ndarray, Fake:np.ndarray, fTimeEnd:float) -> list:
    """
    evaluates how much fake deviates from data

    data and fake must begin at the same point, for this to work
    Parameters
    ----------
    rData : np.ndarray
        real Data to compare Fake Data against
    rDataErr : np.ndarray
        Error for real Data, used in chi calculation
    rTime : np.ndarray
        Timestamps for all real Data points
    Fake : np.ndarray
        Fake Data points that will be compared to real Data
    fTimeEnd : float
        Total End Time of Fake Data, this function will create its own time array based on this value

    Returns
    -------
    [ x_time, y_UsedData, chi^2 value]
    """

    # create timing for fake data
    fakeTimeline = np.linspace(0,fTimeEnd, len(Fake))

    newTimeOfFake   = np.empty(len(rTime))
    newValues       = np.empty(len(rTime))
    j = 0

    # determine closest fakeTime for every measured timestamp
    # if fake orbit shorter than measured time => last measured points get ignored
    # if fake orbit longer than measured time => last fake points get ignored
    for i in range(len(rTime)):
        for k in range(j, len(fakeTimeline)):
            if (fakeTimeline[k] >= rTime[i]):
                newTimeOfFake[i] = fakeTimeline[k]
                newValues[i] = Fake[k]
                j = k
                break

    chi2 = ((rData - newValues)/rDataErr)**2
    
    return [newTimeOfFake, newValues, np.sum( chi2 ), chi2]


def returnCombinedError(StarData:DataContainer, FitData:FitContainer, _in, redshiftCorr:bool = False) -> float:
    """
    combines all measurement errors
    Parameters
    ----------
    StarData : DataContainer
        The Star Data
    FitData : FitContainer
        The Fit Data, prior to any Error Calculation, Error will be overwritten
    _in : [_index_R, _index_V]
        Index point of starting data
    redshiftCorr : bool
        use redshift correction in error calculation? Only for Schwarzschild potential

    Returns
    -------
    chi^2 value for current parameters
    """
    
    if FitData.success:
        # create timing for fake data
        _eT = StarData.TimeP[_in[0]] - StarData.TimeP[0] + FitData.OrbitNumber * FitData.OrbElem.Period

        # error on every measurement
        Err_RA = returnDataError(StarData.RA, StarData.eRA, StarData.TimeP - StarData.TimeP[0], FitData.PosPath[0], _eT)
        Err_DE = returnDataError(StarData.DE, StarData.eDE, StarData.TimeP - StarData.TimeP[0], FitData.PosPath[1], _eT)
        
        # rad vel points need to be shifted by the same amount as the position data for consistency
        if redshiftCorr:
            
            fakeTimeline = np.linspace(0,_eT, len(FitData.VPath))
            j = 0
            rTime = StarData.TimeV - StarData.TimeP[0]

            #newVR_Timeline = np.empty(len(rTime))
            LengthAtVR = np.empty(len(rTime))
            newFakeVR = np.empty(len(rTime))

            for i in range(len(rTime)):
                for k in range(j, len(fakeTimeline)):
                    if (fakeTimeline[k] >= rTime[i]):
                        #newVR_Timeline[i] = fakeTimeline[k]
                        LengthAtVR[i] = np.linalg.norm(FitData.PositionArray[k]) #FitData.PositionArray[k][2] #
                        newFakeVR[i] = FitData.VPath[k]
                        j = k
                        break


            PN_VR = StarData.VR - getGravRedshift(FitData.Mass, LengthAtVR)
            _chi2 = ((PN_VR - newFakeVR)/StarData.eVR)**2
            Err_Vz = [StarData.TimeV - StarData.TimeP[0], PN_VR, np.sum( _chi2 ), _chi2]

        else:
            Err_Vz = returnDataError(StarData.VR, StarData.eVR, StarData.TimeV - StarData.TimeP[0], FitData.VPath, _eT)


        lenPos = len(StarData.RA)
        lenVel = len(StarData.VR)
        NPos = lenPos / (lenPos + lenVel)
        NVel = lenVel / (lenPos + lenVel)
        
        #print("len: ", len(Err_RA[3]) + len(Err_DE[3]) + len(Err_Vz[3]))
        FitData.initErrorData(Err_RA, Err_DE, Err_Vz)   # save individual errors in FitData

        # chi^2 value
        #chi2 = (Err_RA[2] + Err_DE[2] + Err_Vz[2])
        chi2 = (NPos * Err_RA[2] + NPos * Err_DE[2] + NVel * Err_Vz[2])
        #Nlen = len(Err_RA[3]) + len(Err_DE[3]) + len(Err_Vz[3])

        #chi2 = chi2/Nlen

        return chi2
    
    else:
        return 1E10


def returnCombinedErrorFromFile(SD:DataContainer, FitData:FitContainer, _in) -> float:
    
    _OFile = open(OrbitFileForewrd, 'r')
    _line = _OFile.readline()

    NumberLines = -2
    StartBackwards = -1

    while _line:
        NumberLines += 1
        if _line[0] == '#' and NumberLines > 0:
            StartBackwards = NumberLines
        _line = _OFile.readline()

    _OFile.close()
    _OFile = open(OrbitFileForewrd, 'r')
    _line = _OFile.readline()

    chi2RA = 0
    chi2DE = 0
    chi2VR = 0
    fCount = -1

    PositionRealTime = SD.TimeP - SD.TimeP[0]
    VelocityRealTime = SD.TimeV - SD.TimeP[0]
    # end of time
    _eT = SD.TimeP[_in[0]] - SD.TimeP[0] + FitData.OrbitNumber * FitData.OrbElem.Period
    # time from index point to ent of time
    fakeTimeline = np.linspace(SD.TimeP[_in[0]] - SD.TimeP[0], _eT, StartBackwards - 1)

    fakeTimelineBack = np.linspace(0, SD.TimeP[_in[0]] - SD.TimeP[0], NumberLines - StartBackwards)
    fakeTimelineBack = np.flip(fakeTimelineBack)

    rUsedF = []
    vUsedF = []

    RAIndex = 0
    VRIndex = 0
    count = 1

    # forward
    while _line:
        if count > StartBackwards:
            break
        count += 1
        _t = _line.strip()
        _line = _OFile.readline()
        if _t[0] != '#':
            _t = _t.split(" ")
            _t = [float(x) for x in _t]
            r = np.array(_t[:3])
            v = np.array(_t[3:])

            if fakeTimeline[count-1] >= PositionRealTime[RAIndex]:
                rUsedF.append(r)
                RAIndex = count - 1
            
            if fakeTimeline[count - 1] >= VelocityRealTime[VRIndex]:
                vUsedF.append(v)
                VRIndex = count - 1


    _OFile.close()
    _OFile = open(OrbitFileForewrd, 'r')
    _line = _OFile.readline()
    count = 1
    rUsedB = []
    vUsedB = []

    while _line:
        if count < StartBackwards:
            _line = _OFile.readline()
        else:
            _t = _line.strip()
            _line = _OFile.readline()
            _t = _t.split(" ")
            _t = [float(x) for x in _t]
            r = np.array(_t[:3])
            v = np.array(_t[3:])

            if fakeTimeline[count-1] >= PositionRealTime[RAIndex]:
                rUsedB.append(r)
                RAIndex = count - 1
            
            if fakeTimeline[count - 1] >= VelocityRealTime[VRIndex]:
                vUsedB.append(v)
                VRIndex = count - 1


    if FitData.success:
        # create timing for fake data
        _eT = SD.TimeP[_in[0]] - SD.TimeP[0] + FitData.OrbitNumber * FitData.OrbElem.Period

        # error on every measurement
        Err_RA = returnDataError(SD.RA, SD.eRA, SD.TimeP - SD.TimeP[0], FitData.PosPath[0], _eT)
        Err_DE = returnDataError(SD.DE, SD.eDE, SD.TimeP - SD.TimeP[0], FitData.PosPath[1], _eT)
        
        Err_Vz = returnDataError(SD.VR, SD.eVR, SD.TimeV - SD.TimeP[0], FitData.VPath, _eT)

        FitData.initErrorData(Err_RA, Err_DE, Err_Vz)   # save individual errors in FitData

        # chi^2 value
        chi2 = (Err_RA[2] + Err_DE[2] + Err_Vz[2])

        return chi2
    
    else:
        return 1E10


def returnSpecificChi2Point(SD:DataContainer, ParVec:list, _in:list, kwargs:dict={}) -> float:
    
    OrbEl = getOrbitalElements(ParVec)
    NewFitData = getOrbit(SD=SD, Orb=OrbEl, ParamVec=ParVec, index=_in[0], max_iter=10E6, stepsize=kwargs['Stepsize'], kwargs=kwargs)
    x = returnCombinedError(SD, NewFitData, _in, kwargs['grav-red'])

    return x


def ParVecToParams(_list:list) -> list:
    """
    Returns parameter list for use in orbit parsing
    Parameters
    ----------
    _list : ParVec
    Returns
    -------
    [Mass, R vec, V vec, Distance]
    """
    _M = _list[0]
    _r = np.array( [_list[1], _list[2], _list[3]] )
    _v = np.array( [_list[4], _list[5], _list[6]] )
    _d = _list[7]
    return [_M, _r, _v, _d]


def generateMCData(OldData:np.ndarray, Error:np.ndarray) -> np.ndarray:
    """
    Generate a new set of points given the old data and the error bars.
    All points are within 1 sigma scaled with error
    Parameters
    ----------
    OldData : list
        Data points in given Coorinate
    Error : list
        coresponding errors

    Returns
    -------
    list
        new set of datapoints (of same length)
    """
    
    sig = np.random.normal(0,1,len(OldData))
    newData = OldData + sig * Error
    return newData


def ProgressBar(count:int, total:int, status:str=''):
    '''
    a simple progressbar to keep output clean
    '''
    barLen = 60
    fillLen = int(round(barLen*count/float(total)))
    percent = round(100*count/float(total),1)
    bar = '='*fillLen + '-'*(barLen-fillLen)

    sys.stdout.write('[%s] %s%s (%s) ... %s\r' % (bar, percent, '%', count, status))
    sys.stdout.flush()


def NoProgressBar(count:int, status:str=''):
    '''
    Display clean Message without progressbar
    '''
    sys.stdout.write('(%s) ... %s\r' % (count, status))
    sys.stdout.flush()


def getGravRedshift(M:float, r:np.ndarray) -> np.ndarray:
    """
    returns the velocity change due to gravitational redshift
    Parameters
    ----------
    M : float
        Mass of Black hole
    r : np.ndarray
        current position of star

    Returns
    -------
    Delta RVel : float
        radialvelocity correction
    """
    # Schwarzschild radius
    rs = 2 * GLOB_G * M / GLOB_c**2
    # redshift
    z = ( 1 / np.sqrt( 1 - rs/r ) ) - 1
    return GLOB_c * z

# PLOT FUNCTIONS

def plot4Ways(_fig, SD:DataContainer, FD:FitContainer = None, _in:list = [-1,-1], _fName:str = None):
    """
    plot 4 diagrams showing position and time dependency of data, plus FitData, if available
    Parameters
    ----------
    SD : DataContainer
        StarData
    FD : FitContainer
        FitData, can be None
    _fig : Reference to main Figure
    _fName : str, optional
        save plot as file, by default "frame0001"
    showGraph : bool, optional
        Show Figure, by default True
    _in: [_index_R, _index_V]
        if set, draw a point around the Index point
    """
    showFit = True
    if not FD:
        showFit = False

    #-------------------------------------------------------------
    # CONFIG
    StarColor = 'black'
    StarErr = 'blue'
    _ms=3                   # marker size
    chi2Color = 'tab:orange'

    _fig.clf()
    F = []

    for i in range(4):
        _tf = _fig.add_subplot(2,2,i+1)
        _tf.set_axisbelow(True)
        _tf.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        F.append(_tf)

    F[0].set_aspect('equal', 'box')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #-------------------------------------------------------------
    #x-y
    F[0].set_xlabel(r'RA [mas]', {'size':14})
    F[0].set_ylabel(r'DE [mas]', {'size':14})
    #vR-time
    F[1].set_xlabel(r'time - ' + str(SD.TimeP[0]) + ' [yr]', {'size':14})
    F[1].set_ylabel(r'RVel [km/s]', {'size':14})
    #x-time (RA)
    F[2].set_xlabel(r'time - ' + str(SD.TimeP[0]) + ' [yr]', {'size':14})
    F[2].set_ylabel(r'RA [mas]', {'size':14})
    #y-time (DE)
    F[3].set_xlabel(r'time - ' + str(SD.TimeP[0]) + ' [yr]', {'size':14})
    F[3].set_ylabel(r'DE [mas]', {'size':14})

    #-------------------------------------------------------------
    #Real Data
    #center
    F[0].scatter(0,0,c="red", marker='+', label='center', s=50)
    #position x-y
    F[0].errorbar(SD.RA, SD.DE, xerr=SD.eRA, yerr=SD.eDE, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) + ' Orbit', ms=_ms)
    #vR-time
    F[1].errorbar(SD.TimeV - SD.TimeP[0], SD.VR, yerr=SD.eVR, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) +' RVel', ms=_ms, zorder=2)
    #x-time
    F[2].errorbar(SD.TimeP - SD.TimeP[0], SD.RA, yerr=SD.eRA, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) + ' RA',ms=_ms, zorder=2)
    #y-time
    F[3].errorbar(SD.TimeP - SD.TimeP[0], SD.DE, yerr=SD.eDE, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) + ' DE',ms=_ms, zorder=2)

    #-------------------------------------------------------------
    #fake Data
    if showFit:

        # This is the length of the fake data. From Index point it extends some integer amount orbit to front
        # and backwards it extends to 0, because of float orbit number for back direction
        # subtract first time measure for relative data
        DataLenFromIndex = SD.TimeP[_in[0]] - SD.TimeP[0] + FD.OrbitNumber * FD.OrbElem.Period
        fake_time = np.linspace(0,DataLenFromIndex, len(FD.PosPath[0]))
        # end of real data, in relative units
        END_Data = SD.TimeP[-1] - SD.TimeP[0]
        # update time to only display relevant data
        fake_time = [x for x in fake_time if x < END_Data]
        #length of relevant data, used for truncating fake data
        FI = len(fake_time)
        # init points used for chi^2 and remove duplicates
        # 0 - time; 1 - values
        chi2RA = FD.ErrRA
        chi2DE = FD.ErrDE
        chi2RVel = FD.ErrVR

        SimRA = [x for x in FD.PosPath[0][:FI] if x not in chi2RA[1]]
        SimRA_Time = [x for x in fake_time if x not in chi2RA[0]]
        SimDE = [x for x in FD.PosPath[1][:FI] if x not in chi2DE[1]]
        SimDE_Time = [x for x in fake_time if x not in chi2DE[0]]
        SimRV = [x for x in FD.VPath[:FI] if x not in chi2RVel[1]]
        SimRV_Time = [x for x in fake_time if x not in chi2RVel[0]]

        #------------------------------------------------------------- 
        # Simulation data points
        #position x-y
        F[0].plot(FD.PosPath[0], FD.PosPath[1], c='tab:blue', label='Fit')
        #vR-time
        F[1].plot(fake_time, FD.VPath[:FI], label='sim RVel', zorder=1)
        #x-time
        F[2].plot(SimRA_Time, SimRA, label='sim RA', zorder=1)
        #y-time
        F[3].plot(SimDE_Time, SimDE, label='sim DE', zorder=1)
        #------------------------------------------------------------- 
        # simulation points used in chi^2
        #F[1].scatter(FD.ErrVR[0], FD.ErrVR[1], label=r'$\chi^2$ points', c=chi2Color, s=_ms, zorder=3) #vR - vz
        #F[2].scatter(FD.ErrRA[0], FD.ErrRA[1], label=r'$\chi^2$ points', c=chi2Color, s=_ms, zorder=3) #RA - x
        #F[3].scatter(FD.ErrDE[0], FD.ErrDE[1], label=r'$\chi^2$ points', c=chi2Color, s=_ms, zorder=3) #DE - y
        #------------------------------------------------------------- 
        # draw index point
        if (_in[0] > 0 and _in[1] > 0):
            F[1].scatter(SD.TimeV[_in[1]] - SD.TimeP[0], SD.VR[_in[1]], label=r'Index', s=20, color='red', zorder=99) #vR - vz
            F[2].scatter(SD.TimeP[_in[0]] - SD.TimeP[0], SD.RA[_in[0]], label=r'Index', s=20, color='red', zorder=99) #RA - x
            F[3].scatter(SD.TimeP[_in[0]] - SD.TimeP[0], SD.DE[_in[0]], label=r'Index', s=20, color='red', zorder=99) #DE - y
        #-------------------------------------------------------------

        # Print Orbit Elements left of screen
        OrbElem = FD.OrbElem
        RoundTo = 3
        # Mass
        plt.figtext(0.01,0.7, r"M [$10^6 M_\odot$] =" + str( np.round(FD.Mass/1E6, RoundTo) ) )
        print("M = ", FD.Mass/1E6)
        # Distance
        plt.figtext(0.01,0.65, "R [kpc] =" + str( np.round(FD.Distance/1E3, RoundTo) ) )
        print("D = ", FD.Distance/1E3)
        # Period
        plt.figtext(0.01,0.6, "T [yr] =" + str( np.round(OrbElem.Period, RoundTo) ) )
        print("T = ", OrbElem.Period)
        # Eccentricity
        plt.figtext(0.01,0.55, "e [1] =" + str( np.round(OrbElem.Ecc, RoundTo) ) )
        print("e = ", OrbElem.Ecc)
        # Semi Mayor Axis
        plt.figtext(0.01,0.45, "a [pc] =" + str( np.round(OrbElem.MayAxis, RoundTo) ) )
        print("a = ", OrbElem.MayAxis)
        # Inclination
        plt.figtext(0.01,0.4, r"i [$^\circ$] =" + str( np.round(OrbElem.Incl, RoundTo) ) )
        print("i = ", OrbElem.Incl)
        # Longitude of ascending node
        plt.figtext(0.01,0.35, r"$\Omega$ [$^\circ$] =" + str( np.round(OrbElem.LAN, RoundTo) ) )
        print("LAN = ", OrbElem.LAN)
        # argument of periapsis
        plt.figtext(0.01,0.3, r"$\omega$ [$^\circ$] =" + str( np.round(OrbElem.ArgPeri, RoundTo) ) )
        print("omega = ", OrbElem.ArgPeri)

    for i in range(len(F)):
        F[i].legend(loc='best', fontsize=12)
    
    if _fName:
        plt.savefig("SMBH/Data/dump/" + _fName)


def plot2Ways(_fig, SD:DataContainer, FD:FitContainer = None, _in:list = [-1,-1], _fName:str = None):
    """
    plot 2 diagrams showing position and Radial Velocity over Time
    Parameters
    ----------
    SD : DataContainer
        StarData
    FD : FitContainer
        FitData, can be None
    _fig : Reference to main Figure
    _fName : str, optional
        save plot as file, by default "frame0001"
    showGraph : bool, optional
        Show Figure, by default True
    _in: [_index_R, _index_V]
        if set, draw a point around the Index point
    """
    showFit = True
    if not FD:
        showFit = False

    #-------------------------------------------------------------
    # CONFIG
    StarColor = 'black'
    StarErr = 'gray'
    _ms=3                   # marker size
    chi2Color = 'tab:orange'

    _fig.clf()
    F = []

    for i in range(2):
        _tf = _fig.add_subplot(1,2,i+1)
        _tf.set_axisbelow(True)
        _tf.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        F.append(_tf)

    F[0].set_aspect('equal', 'box')
    #-------------------------------------------------------------
    #x-y
    F[0].set_xlabel(r'RA [mas]', {'size':14})
    F[0].set_ylabel(r'DE [mas]', {'size':14})
    #vR-time
    F[1].set_xlabel(r'time - ' + str(SD.TimeP[0]) + ' [yr]', {'size':14})
    F[1].set_ylabel(r'RVel [km/s]', {'size':14})
    #-------------------------------------------------------------
    #Real Data
    #center
    F[0].scatter(0,0,c="red", marker='+', label='center', s=50)
    #position x-y
    F[0].errorbar(SD.RA, SD.DE, xerr=SD.eRA, yerr=SD.eDE, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) + ' Orbit', ms=_ms)
    #vR-time
    F[1].errorbar(SD.TimeV - SD.TimeP[0], SD.VR, yerr=SD.eVR, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) +' RVel', ms=_ms, zorder=2)
    #-------------------------------------------------------------
    #fake Data
    if showFit:

        # This is the length of the fake data. From Index point it extends some integer amount orbit to front
        # and backwards it extends to 0, because of float orbit number for back direction
        # subtract first time measure for relative data
        DataLenFromIndex = SD.TimeP[_in[0]] - SD.TimeP[0] + FD.OrbitNumber * FD.OrbElem.Period
        fake_time = np.linspace(0,DataLenFromIndex, len(FD.PosPath[0]))
        # end of real data, in relative units
        END_Data = SD.TimeP[-1] - SD.TimeP[0]
        # update time to only display relevant data
        fake_time = [x for x in fake_time if x < END_Data]
        #length of relevant data, used for truncating fake data
        FI = len(fake_time)
        # init points used for chi^2 and remove duplicates
        # 0 - time; 1 - values
        chi2RVel = FD.ErrVR

        SimRV = [x for x in FD.VPath[:FI] if x not in chi2RVel[1]]
        SimRV_Time = [x for x in fake_time if x not in chi2RVel[0]]


        #------------------------------------------------------------- 
        # Simulation data points
        #position x-y
        F[0].plot(FD.PosPath[0], FD.PosPath[1], c='tab:blue', label='Fit')
        #vR-time
        F[1].plot(fake_time, FD.VPath[:FI], label='sim RVel', zorder=1)
        #------------------------------------------------------------- 
        # simulation points used in chi^2
        #F[1].scatter(FD.ErrVR[0], FD.ErrVR[1], label=r'$\chi^2$ points', c=chi2Color, s=_ms, zorder=3) #vR - vz
        #------------------------------------------------------------- 
        # draw index point
        if (_in[0] > 0 and _in[1] > 0):
            F[1].scatter(SD.TimeV[_in[1]] - SD.TimeP[0], SD.VR[_in[1]], label=r'Index', s=20, color='red', zorder=99) #vR - vz
        #-------------------------------------------------------------

        # Print Orbit Elements left of screen
        OrbElem = FD.OrbElem
        RoundTo = 3
        # Mass
        plt.figtext(0.01,0.7, r"M [$10^6 M_\odot$] =" + str( np.round(FD.Mass/1E6, RoundTo) ), {'size':16} )
        print("M = ", FD.Mass/1E6)
        # Distance
        plt.figtext(0.01,0.65, "R [kpc] =" + str( np.round(FD.Distance/1E3, RoundTo) ), {'size':16} )
        print("D = ", FD.Distance/1E3)
        # Period
        plt.figtext(0.01,0.6, "T [yr] =" + str( np.round(OrbElem.Period, RoundTo) ), {'size':16} )
        print("T = ", OrbElem.Period)
        # Eccentricity
        plt.figtext(0.01,0.55, "e [1] =" + str( np.round(OrbElem.Ecc, RoundTo) ), {'size':16} )
        print("e = ", OrbElem.Ecc)
        # Semi Mayor Axis
        plt.figtext(0.01,0.45, "a [pc] =" + str( np.round(OrbElem.MayAxis, RoundTo) ), {'size':16} )
        print("a = ", OrbElem.MayAxis)
        # Inclination
        plt.figtext(0.01,0.4, r"i [$^\circ$] =" + str( np.round(OrbElem.Incl, RoundTo) ), {'size':16} )
        print("i = ", OrbElem.Incl)
        # Longitude of ascending node
        plt.figtext(0.01,0.35, r"$\Omega$ [$^\circ$] =" + str( np.round(OrbElem.LAN, RoundTo) ), {'size':16} )
        print("LAN = ", OrbElem.LAN)
        # argument of periapsis
        plt.figtext(0.01,0.3, r"$\omega$ [$^\circ$] =" + str( np.round(OrbElem.ArgPeri, RoundTo) ), {'size':16} )
        print("omega = ", OrbElem.ArgPeri)

    for i in range(len(F)):
        F[i].legend(loc='best', fontsize=12)
    
    if _fName:
        plt.savefig("SMBH/Data/dump/" + _fName)


def plotDataAndFit(_fig, SD:DataContainer, FD:FitContainer, _fName:str = None):
    '''
    plots only the Positions of the Star with the Fit
    '''

    #-------------------------------------------------------------
    # CONFIG
    StarColor = 'black'
    StarErr = 'gray'
    _ms=3                   # marker size

    _fig.clf()
    _tf = _fig.add_subplot(1,1,1)
    _tf.set_aspect('equal', 'box')
    _tf.set_axisbelow(True)
    _tf.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    _tf.set_xlabel(r'RA [mas]', {'size':14})
    _tf.set_ylabel(r'DE [mas]', {'size':14})

    #-------------------------------------------------------------
    # center
    _tf.scatter(0,0,c="red", marker='+', label='center', s=50)
    # actual data
    _tf.errorbar(SD.RA, SD.DE, xerr=SD.eRA, yerr=SD.eDE, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) + ' Orbit', ms=_ms)
    # fake data
    _tf.plot(FD.PosPath[0], FD.PosPath[1], c='tab:blue', label='Fit')
    #-------------------------------------------------------------

    OrbElem = FD.OrbElem
    RoundTo = 3
    # Mass
    plt.figtext(0.01,0.7, r"M [$10^6 M_\odot$] =" + str( np.round(FD.Mass/1E6, RoundTo) ), {'size':16})
    print("M = ", FD.Mass/1E6)
    # Distance
    plt.figtext(0.01,0.65, "D [kpc] =" + str( np.round(FD.Distance/1E3, RoundTo) ), {'size':16} )
    print("R = ", FD.Distance/1E3)
    # Period
    plt.figtext(0.01,0.6, "T [yr] =" + str( np.round(OrbElem.Period, RoundTo) ), {'size':16} )
    print("T = ", OrbElem.Period)
    # Eccentricity
    plt.figtext(0.01,0.55, "e [1] =" + str( np.round(OrbElem.Ecc, RoundTo) ), {'size':16} )
    print("e = ", OrbElem.Ecc)
    # Semi Mayor Axis
    plt.figtext(0.01,0.45, "a [pc] =" + str( np.round(OrbElem.MayAxis, RoundTo) ), {'size':16} )
    print("a = ", OrbElem.MayAxis)
    # Inclination
    plt.figtext(0.01,0.4, r"i [$^\circ$] =" + str( np.round(OrbElem.Incl, RoundTo) ), {'size':16} )
    print("i = ", OrbElem.Incl)
    # Longitude of ascending node
    plt.figtext(0.01,0.35, r"$\Omega$ [$^\circ$] =" + str( np.round(OrbElem.LAN, RoundTo) ), {'size':16} )
    print("LAN = ", OrbElem.LAN)
    # argument of periapsis
    plt.figtext(0.01,0.3, r"$\omega$ [$^\circ$] =" + str( np.round(OrbElem.ArgPeri, RoundTo) ), {'size':16} )
    print("omega = ", OrbElem.ArgPeri)

    _tf.legend(loc='best', fontsize=12)

    if _fName:
        plt.savefig("SMBH/Data/dump/" + _fName)
      

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       ROUTINE
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def getOrbit(SD:DataContainer, Orb:OrbitElem, ParamVec:list, index:int, max_iter:float, stepsize:float, kwargs:dict={}) -> FitContainer:
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

    if 'useFile' in kwargs.keys():
        useExtFile = kwargs['useFile']
    else:
        useExtFile = False
    
    if useExtFile:
        _OFile = open(OrbitFileForewrd, "w")
    

    #------------------------------------------------------------------------
    #   ORBIT INTEGRATION

    # Integrate from Index Point to End point, forward time
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

        if useExtFile:
            _OFile.write(str(r_cur[0]) + " " + str(r_cur[1]) + " " + str(r_cur[2]) + " " + str(v_cur[0]) + " " + str(v_cur[1]) + " " + str(v_cur[2]) + "\n")
        else:
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


    # reset some data
    cur_timer = 0
    r_cur = _r
    v_cur = _v            
    f_cur = PotFunc(_r,_v,_M)

    if useExtFile:
        _OFile.close()
        _OFile = open(OrbitFileBackwrd, "w")

    # Integrate multiple orbits backwards depending on data beginning
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

        if useExtFile:
            _OFile.write(str(r_cur[0]) + " " + str(r_cur[1]) + " " + str(r_cur[2]) + " " + str(v_cur[0]) + " " + str(v_cur[1]) + " " + str(v_cur[2]) + "\n")
        else:
            PosTrackerBack.append(r_cur)
            VelTrackerBack.append(v_cur)


    if useExtFile:
        _OFile.close()

    if DEBUG_MODE:
        print("backward cutoff = ", cur_timer)

    #------------------------------------------------------------------------
    #   CONCAT DATA

    if not useExtFile:
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


def FitDataInner(SD:DataContainer, kwargs:dict={}) -> Tuple[FitContainer, list]:
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


def FitDataStandalone(_starNr:int, kwargs:dict={}) -> Tuple[FitContainer, DataContainer, list]:
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
    SD = DataContainer(_starNr, S_Data)

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


def genMCD(SD:DataContainer, iter:int, kwargs:dict={}):
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


def genMCD_MP(SD:DataContainer, pid:int, kwargs:dict={}):
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


def DrawChi2Slice(_fig, SD:DataContainer, parVec:list, bounds:list, IndexList:list, _dim:int=50, kwargs:dict={}):
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


def FindChiError(_fig, FD:FitContainer, SD:DataContainer, selIn:list, OPT:dict, start:float, stop:float, step:int, index:int):
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
    useFile         bool        output orbit to file? used for stepsizes under 1E-12
    File            str         Output File for Ensemble Test
    UseSGRA_Pos     bool        variate position of Sgr A*?
    
    """

    #------------------------------------------------------------------------------------------

    # Options used for Initial Orbit Calculation
    OPTIONS = {
        'method': 'Schwarz', 
        'grav-red': True, 
        'Fit': 'None', 
        'Stepsize': 1E-9,
        'Pbar': True,
        'useFile': False
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
    #plot4Ways(_fig=MAIN_fig, SD=_SD, FD=minFD, _in=selIndex)           # plot the 2 views at once
    plotDataAndFit(MAIN_fig, _SD, FD=minFD)                    # plot just the orbit in big


    #plt.savefig("SMBH/Data/dump/E8Hist.png", transparent=True)
    plt.show()