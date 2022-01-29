import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.index_tricks import r_

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       CONSTANTS
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

DEBUG_MODE = False

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

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       CLASSES
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------


class OrbitElem():
    """
    Data Object containing all Orbital Elements
    """
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


class FitContainer():
    '''
    Container for all the Data needed for a Fit
    '''
    success : bool = False
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
    OrbElem : OrbitElem = None
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


class StarContainer():
    """
    Stores the Data from a Star in a Container with additional functionality
    """
    TimeP   = None
    TimeV   = None
    RA      = None
    eRA     = None
    DE      = None
    eDE     = None
    VR      = None
    eVR     = None

    def __init__(self, _stNr:int, _SData:dict):
        """
        Create Star Object.
        Parameters
        ----------
        _stNr : int
            Star ID
        _SData : dict
            Original Data struct for the specified Star
        """

        if _SData:
            self.StarNr = _stNr
            scale = np.sqrt(5)/2.60335    # for weighted
            #scale = np.sqrt(5)/2.192    # for unweighted

            #positional data
            self.TimeP  = np.array( [x["time"]         for x in _SData["pos"]] )
            self.RA     = np.array( [x["RA"]           for x in _SData["pos"]] )
            self.DE     = np.array( [x["DE"]           for x in _SData["pos"]] )
            self.eRA    = np.array( [x["e_RA"] * scale for x in _SData["pos"]] )
            self.eDE    = np.array( [x["e_DE"] * scale for x in _SData["pos"]] )

            #RVel data
            self.VR     = np.array( [x["RVel"]           for x in _SData["rad"]] )
            self.eVR    = np.array( [x["e_RVel"] * scale for x in _SData["rad"]] )
            self.TimeV  = np.array( [x["time"]           for x in _SData["rad"]] )

    # create and return a copy of this object
    def copy(self):
        _t = StarContainer(self.StarNr, None)
        _t.TimeP    = self.TimeP
        _t.RA       = self.RA
        _t.DE       = self.DE
        _t.eRA      = self.eRA
        _t.eDE      = self.eDE
        _t.VR       = self.VR
        _t.eVR      = self.eVR
        _t.TimeV    = self.TimeV
        return _t


class Par():
    """
    docstring
    """
    
    Values : list
    Name : str

    def __init__(self, dim, name):
        self.Name = name
        self.Values = [0.]*dim

    @classmethod
    def Star(cls, name):
        x = cls(7,name)
        return x
    
    @classmethod
    def SMBH(cls):
        x = cls(8,"SMBH")
        return x


class PV():
    """
    Parameter Vector wrapper class
    """

    ParVec : np.ndarray = np.empty(0)
    ParVecTemplate : list[Par] = []
    bUseDynamicCenter : bool = False
    Dimension : int = 0

    def __init__(self, _list):
        _t = Par.SMBH()
        self.ParVecTemplate.append(_t)
        self.ParVec = np.append(self.ParVec, _t.Values)
        for i in _list:
            _t = Par.Star(i)
            self.ParVecTemplate.append(_t)
            self.ParVec = np.append(self.ParVec, _t.Values)
        
        self.Dimension = len(_list)

    def SetDynamicCenter(self, dynamicCenter:bool):
        """
        Sets if dynamic Center is allowed. Must be set at the very beginning.
        If enabled, the SMBH can move in the fit.
        Default: False
        """

        self.bUseDynamicCenter = dynamicCenter

    def updateParVec(self, ParVec:np.ndarray):
        """
        update the Parameter Vector with newly supplied vector
        Parameters
        ----------
        ParVec : np.ndarray
            new Parameter Vetor
        """
        self.ParVec = ParVec


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       ORBITAL ELEMENTS
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def getT(r0:np.ndarray, v0:np.ndarray, _M:float) -> float:
    """
    Get the Period of the Orbit based on initial Position and Velocity
    Parameters
    ----------
    r0 : np.ndarray
        Init Position
    v0 : np.ndarray
        Init Velocity
    _M : float
        Central Mass

    Returns
    -------
    float
        Period [yr]
    """

    _a = 1/( 2/np.linalg.norm(r0) - np.linalg.norm(v0)**2/(GLOB_G * _M) )   # a in pc
    _t = 2*np.pi * np.sqrt( (_a**3)/(GLOB_G*_M) )                           # Units = sec * pc / km
    return _t * GLOB_SecToYr * GLOB_PcToKm                                  # convert to year plus additional length factor


def getOrbitalElements(_parVec:list) -> OrbitElem:
    """
    Returns all Orbital Elements and the Period, packed into a data class
    Parameters
    ----------
    _parVec : list
        Parameter Vector for current orbit

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
    Only calculate e and T to be bound checked for fit
    Parameters
    ----------
    _parVec : list
        Parameter Vector for current orbit

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

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       UTILITY
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def ProgressBar(count:int, total:int, status:str=''):
    '''
    a simple progressbar to keep output clean
    Parameters
    ----------
    count : int
        current iteration
    total : int
        total iterations
    status : str
        additional output string, optional
    '''
    barLen = 60
    fillLen = int(round(barLen*count/float(total)))
    percent = round(100*count/float(total),1)
    bar = '='*fillLen + '-'*(barLen-fillLen)

    sys.stdout.write('[%s] %s%s (%s) ... %s\r' % (bar, percent, '%', count, status))
    sys.stdout.flush()


def NoProgressBar(count:int, status:str=''):
    '''
    Display clean Progress Message without progressbar
    Parameters
    ----------
    count : int
        current iteration
    total : int
        total iteration
    '''
    sys.stdout.write('(%s) ... %s\r' % (count, status))
    sys.stdout.flush()


def PosRadToReal(_r:np.ndarray, _dist:float) -> np.ndarray:
    """
    Converts the first 2 radial Elements to real distance
    Parameters
    ----------
    _r : np.ndarray
        Vector to convert
    _dist : float
        Distance of Angle vector

    Returns
    -------
    np.ndarray
        Position Vector [pc]
    """

    return _r*_dist*GLOB_masToRad


def RadToReal(_x:float, _dist:float) -> float:
    """
    Converts an angle scalar into a distance
    Parameters
    ----------
    _x : float
        angle
    _dist : float
        Distance of the angle

    Returns
    -------
    float
        Distance [pc]
    """

    return _x*_dist*GLOB_masToRad


def PosRealToRad(_r:np.ndarray, _dist:float) -> np.ndarray:
    """
    Converts the first 2 positional Elements into angles 
    Parameters
    ----------
    _r : np.ndarray
        position Vector [pc]
    _dist : float
        Distance to evaluate at

    Returns
    -------
    np.ndarray
        Radial Vector [[rad], [rad], [pc]]
    """

    _t = np.array([ _dist*GLOB_masToRad, _dist*GLOB_masToRad, 1 ])
    return _r/_t


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


def ParVecClsToArray(ParVec:PV) -> np.ndarray:
    """
    Converts Parameter Vector Class to an actual Vector for use in integration and fitting
    Parameters
    ----------
    ParVec : PV
        Parameter Vector Class

    Returns
    -------
    np.ndarray
        All Parameters stored in a vector
    """

    vec = np.empty(0)

    # if dynamic Center is enabled, position and velocity of SMBH needs to be fitted as well
    if ParVec.bUseDynamicCenter:
        for i in range(len(ParVec.ParVecTemplate)):
            np.append(vec, ParVec.ParVecTemplate[i].Values)

    # if disabled, position and velocity will be static and zero, no fitting required
    else:
        np.append(vec, ParVec.ParVecTemplate[0].Values[:-6])
        for i in range(len(ParVec.ParVecTemplate) - 1):
            np.append(vec, ParVec.ParVecTemplate[i+1].Values)

    return vec


def getParVecClsName(ParVec:PV) -> list[str]:
    """
    Converts the Parameter Vector Class into a list of Names for all parameters. Used in plotting
    Parameters
    ----------
    ParVec : PV
        Parameter Vector class

    Returns
    -------
    list[str]
        list of names for the parameters
    """

    NamList = []
    _tN = ["Mass", "Distance", "rx", "ry", "rz", "vx", "vy", "vz"]

    # TODO
    if ParVec.bUseDynamicCenter:
        pass
    

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       IMPORT AND READ DATA
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def readTable(_fName:str) -> tuple:
    """
    INT FX. reads the supplied file. Needs a specific format
    Parameters
    ----------
    _fName : str
        file to be read

    Returns
    -------
    tuple
        (Data[row][point], Header[point][more info])
    """

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
    INT FX. Check if any element in Data is non zero. Helper function
    Parameters
    ----------
    _data : list
        Data to be checked for valid entries
    _index : int
        row of the Data to be evaluated. Constant
    _indexList : list
        Elements within the row to check
    
    Returns
    -------
    True if any element is non zero, False otherwise
    """

    for i in _indexList:
        if ( _data[0][_index][i] != '' ):
            return True
    return False


def return_StarExistingData(_data:list, StarNr:int) -> dict:
    """
    return data for specific Star
    Parameters
    ----------
    _data : list
        Source data read out from file
    StarNr : int
        Specific Star ID
    
    Returns
    -------
    dict
        [ Data["pos"], Data["rad"] ] with\n
        Data["pos"] = "time", "RA", "e_RA", "DE", "e_DE"\n
        Data["rad"] = "time", "RVel", "e_RVel"
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


def GetStarData(_dataFile:str, _starNr:int) -> StarContainer:
    """
    get a Star Object for specified Star number.
    Parameters
    ----------
    _dataFile : str
        Location of source data
    _starNr : int
        Star ID for which the Data Container is created
    
    Returns
    -------
    StarContainer
        Data Object containing data of the specified Star
    """

    #read complete data from file
    Data = readTable(_dataFile)
    #return Data for Star S-NUMBER
    S_Data = return_StarExistingData(Data, _starNr)
    #Star Data Container
    SD = StarContainer(_starNr, S_Data)

    return SD

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       INTEGRATION AND POTENTIAL
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

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

    # Target - Begin => force is positive
    dist = r_SGRA - r

    return (GLOB_G*_M*dist) / (np.linalg.norm(dist)**3)


def ArbPotential():
    # TODO
    pass


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


def getGravRedshift(M:float, r:np.ndarray, r_SMBH:np.ndarray = np.zero(3)) -> np.ndarray:
    """
    returns the velocity change due to gravitational redshift for any object at distance r from the black hole
    Parameters
    ----------
    M : float
        Mass of Black hole
    r : np.ndarray
        current position of star
    r_SMBH : np.ndarray
        current position of SMBH

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

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       ERROR CALCULATION
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def returnDataError(RealData:np.ndarray, RealDataErr:np.ndarray, RealTime:np.ndarray, Simulation:np.ndarray, fTimeEnd:float) -> list:
    """
    evaluates how much simulated Data deviates from real Data. Both Datasets must begin at the same time for this to work.
    This works for every Dataset, Position and Radial velocity.
    Parameters
    ----------
    RealData : np.ndarray
        real Data. Consists of discrete points. Will be compared with the Simulations
    RealDataErr : np.ndarray
        Error List for every point in Real Data
    RealTime : np.ndarray
        Timestamps for all real Data points
    Simulation : np.ndarray
        Simulated Data points that will be compared to real Data. Density is usually much higher than Real Data
    fTimeEnd : float
        Total End Time of Fake Data, this function will create its own time array based on this value

    Returns
    -------
    [ x_time, y_UsedData, chi^2 value]
    """

    # create timing for Simulated Data. This begins from 0 to match the Real Timing supplied (must start from 0 also, handled elsewhere)
    fakeTimeline = np.linspace(0,fTimeEnd, len(Simulation))
    # Discrete Timestamps for Simulation Datapoints
    newTimeOfFake   = np.empty(len(RealTime))
    # Discrete Set of points from the Simulation, that are sufficiently near the real Datapoints (in time)
    newValues       = np.empty(len(RealTime))
    j = 0

    # determine closest fakeTime for every measured timestamp
    # if fake orbit is shorter than real time => last measured points get ignored
    # if fake orbit is longer than real time => last fake points get ignored
    for i in range(len(RealTime)):
        for k in range(j, len(fakeTimeline)):
            if (fakeTimeline[k] >= RealTime[i]):
                newTimeOfFake[i] = fakeTimeline[k]
                newValues[i] = Simulation[k]
                j = k
                break

    chi2 = ((RealData - newValues)/RealDataErr)**2
    
    return [newTimeOfFake, newValues, np.sum( chi2 ), chi2]


def returnCombinedError(StarData:StarContainer, FitData:FitContainer, _in, redshiftCorr:bool = False) -> float:
    """
    combines all errors from deviation from the data (2 Positional + 1 Radial Velocity). Done on a per Star level.
    Parameters
    ----------
    StarData : StarContainer
        The Star Data
    FitData : FitContainer
        The Fit Data, prior to any Error Calculation, Error will be overwritten
    _in : [_index_R, _index_V]
        Index point of starting data
    redshiftCorr : bool
        use redshift correction in error calculation? Only for Schwarzschild potential

    Returns
    -------
    final chi^2 value for current Parameter Vector and supplied Star
    """
    
    # True if Period of orbit is defined (T>0)
    if FitData.success:

        #TODO remove index use here
        # create timing for fake data
        _eT = StarData.TimeP[_in[0]] - StarData.TimeP[0] + FitData.OrbitNumber * FitData.OrbElem.Period

        # error on every measurement
        Err_RA = returnDataError(StarData.RA, StarData.eRA, StarData.TimeP - StarData.TimeP[0], FitData.PosPath[0], _eT)
        Err_DE = returnDataError(StarData.DE, StarData.eDE, StarData.TimeP - StarData.TimeP[0], FitData.PosPath[1], _eT)
        
        '''
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
        
        else:'''

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


def returnSpecificChi2Point(SD:StarContainer, ParVec:list, _in:list, kwargs:dict={}) -> float:
    
    #OrbEl = getOrbitalElements(ParVec)
    #NewFitData = getOrbit(SD=SD, Orb=OrbEl, ParamVec=ParVec, index=_in[0], max_iter=10E6, stepsize=kwargs['Stepsize'], kwargs=kwargs)
    #x = returnCombinedError(SD, NewFitData, _in, kwargs['grav-red'])
    pass
    #return x


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

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#       PLOTTING
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def plot4Ways(_fig, SD:StarContainer, FD:FitContainer = None, _in:list = [-1,-1], _fName:str = None):
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


def plot2Ways(_fig, SD:StarContainer, FD:FitContainer = None, _in:list = [-1,-1], _fName:str = None):
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


def plotDataAndFit(_fig, SD:StarContainer, FD:FitContainer, _fName:str = None):
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
#       ROUTINES
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def getOrbit(SD:StarContainer, Orb:OrbitElem, ParVec:PV, index:int, max_iter:float, stepsize:float, kwargs:dict={}) -> FitContainer:
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
    ParVec : PV
        Parameter Vector Class. Can be any dimension
    index : int
        Index point from which the Algorithm starts to integrate in both directions. Index of Position data at specific time used
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
