import numpy as np
import sys
from numpy.core.arrayprint import dtype_short_repr
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import logging
from numba import jit

# ------------------------------------------------------------------

# Gravitational Constant in astronomy Units (pc M_sun^-1 (km/s)^2)
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

class ParVec():
    """
    docstring
    """
    
    ParameterVector : np.ndarray = None
    ParNames : np.ndarray = None
    bCenterMove : bool = False

    def __init__(self, ObjNames:list = None, bCenterCanMove:bool = False) -> None:
        """
        comment
        """
        
        self.bCenterMove : bool = bCenterCanMove
        self.dim         : int = len(ObjNames)
        _t = np.zeros(3)

        if ObjNames == None:
            print("Warning: Parameter Vector contains only one Object!")
            #                                  D M  R   V
            self.ParameterVector = np.array( [ 0,0, _t, _t ], dtype=object )
        else:
            _masses = np.ones(self.dim)
            _f = [1, _masses]
            for _ in range(self.dim):
                _f.append(_t)
                _f.append(_t)

            self.ParameterVector = np.array(_f, dtype=object)
            self.ParNames = np.asarray(ObjNames)

            # FOR DEBUG
            for i in range(self.dim*2):
                self.ParameterVector[i+2] = np.random.random(3)

    def GetDistance(self) -> float:
        """
        comment
        """
        return self.ParameterVector[0]

    def GetMasses(self) -> np.ndarray:
        """
        """
        return self.ParameterVector[1]#.copy()

    def GetPosition(self) -> np.ndarray:
        """
        """
        return self.ParameterVector[2:self.dim+2]#.copy()

    def GetVelocity(self) -> np.ndarray:
        """
        """
        return self.ParameterVector[self.dim+2:]#.copy()


class ParVecObj(ParVec):
    def __init__(self, param) -> None:
        self.ParameterVector = param


class OrbitElem():
    """
    Data Object containing all Orbital Elements
    """
    def __init__(self, a:float = -1, e:float = -1, omega:float = -1, LAN:float = -1, i:float = -1, M:float = -1, T:float = -1, nu:float = -1):
        """
        Container for saving orbital Elements
        Parameters
        ----------
        a : semi mayor axis
        e : eccentricity
        omega : argument of periapsis
        LAN : longitude of ascending node
        i : inclination
        M : Mean anomaly
        T : Period
        """
        self.MayAxis         : float = a
        self.Ecc             : float = e
        self.ArgPeri         : float = omega
        self.LAN             : float = LAN
        self.Incl            : float = i
        self.MeanM           : float = M
        self.Period          : float = T
        self.TAnom           : float = nu


class FitContainer():
    '''
    TODO
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
    
    def __init__(self, Param:ParVec, _oN = 1, success = True, _orb=None, _PosArr = None, _VelArr = None):

        self.success = success

        if success:
            self.Param          = Param
            self.OrbitNumber    = _oN # number of orbits infront of index data
            self.OrbElem        = _orb
            self.PositionArray  = _PosArr
            self.VelocityArray  = _VelArr
        else:
            logging.info("Fitcontainer is not valid")


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

    def __init__(self, _stNr:int, Pos_Data:DataFrame, Rad_Data:DataFrame):
        """
        comment
        """

        self.StarNr = _stNr
        scale = np.sqrt(5)/2.60335    # for weighted
        #scale = np.sqrt(5)/2.192    # for unweighted

        self.PosData = Pos_Data
        self.RadData = Rad_Data

    # create and return a copy of this object
    def copy(self):

        _a = self.PosData.copy(deep=True)
        _b = self.RadData.copy(deep=True)

        _t = StarContainer(self.StarNr, _a, _b)
        return _t


# TODO
class OPTIONS():
    def __init__(self) -> None:
        pass

# ------------------------------------------------------------------

class SMBH():
    """
    Main Class: comment
    """

    def __init__(self) -> None:
        pass


# ------------------------------------------------------------------

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

# TIME, RA, DE, E_RA, E_DE, RVEL, E_RVEL
def return_StarExistingData(_data:list, StarNr:int) -> DataFrame:
    """
    comment
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

    _time = []
    _RA = []
    _e_RA = []
    _DE = []
    _e_DE = []
    _timeR = []
    _RVel = []
    _e_RVel = []

    #form a dict from the data
    #FORMAT:    time, RA, e_RA, DE, e_DE // RVel, e_RVel
    for i in range(len(_data[0])):
        #_data[0][i] is the i-th row of data; [1] is the flag position
        #check flag; a = position
        if (_data[0][i][1] == "a"):
            #is the star data not empty; _index is starting index of star data
            #check for all positional data
            if (EmptyCheck(_data, i, [ _index, _index+1, _index+2,_index+3 ] ) ):
                _time.append(float(_data[0][i][0]))
                _RA.append(float(_data[0][i][_index]))
                _e_RA.append(float(_data[0][i][_index+1]))
                _DE.append(float(_data[0][i][_index+2]))
                _e_DE.append(float(_data[0][i][_index+3]))

        #check if rad flag
        elif (_data[0][i][1] == "rv"):
            if (EmptyCheck(_data, i, [_index+4,_index+5] ) ):
                _timeR.append(float(_data[0][i][0]))
                _RVel.append(float(_data[0][i][_index+4]))
                _e_RVel.append(float(_data[0][i][_index+5]))


    df = {"TIME":_time, "RA":_RA, "E_RA":_e_RA, "DE":_DE, "E_DE":_e_DE}
    dg = {"TIME":_timeR, "RVEL":_RVel, "E_RVEL":_e_RVel}
    df = pd.DataFrame(df)
    dg = pd.DataFrame(dg)

    return df, dg


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
    SD = StarContainer(_starNr, *S_Data)

    return SD

# ------------------------------------------------------------------

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
    F[1].set_xlabel(r'time - ' + str(SD.PosData.TIME[0]) + ' [yr]', {'size':14})
    F[1].set_ylabel(r'RVel [km/s]', {'size':14})
    #x-time (RA)
    F[2].set_xlabel(r'time - ' + str(SD.PosData.TIME[0]) + ' [yr]', {'size':14})
    F[2].set_ylabel(r'RA [mas]', {'size':14})
    #y-time (DE)
    F[3].set_xlabel(r'time - ' + str(SD.PosData.TIME[0]) + ' [yr]', {'size':14})
    F[3].set_ylabel(r'DE [mas]', {'size':14})

    #-------------------------------------------------------------
    #Real Data
    #center
    F[0].scatter(0,0,c="red", marker='+', label='center', s=50)
    #position x-y
    F[0].errorbar(SD.PosData.RA, SD.PosData.DE, xerr=SD.PosData.E_RA, yerr=SD.PosData.E_DE, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) + ' Orbit', ms=_ms)
    #vR-time
    F[1].errorbar(SD.RadData.TIME - SD.PosData.TIME[0], SD.RadData.RVEL, yerr=SD.RadData.E_RVEL, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) +' RVel', ms=_ms, zorder=2)
    #x-time
    F[2].errorbar(SD.PosData.TIME - SD.PosData.TIME[0], SD.PosData.RA, yerr=SD.PosData.E_RA, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) + ' RA',ms=_ms, zorder=2)
    #y-time
    F[3].errorbar(SD.PosData.TIME - SD.PosData.TIME[0], SD.PosData.DE, yerr=SD.PosData.E_DE, fmt='o', ecolor=StarErr, color=StarColor, label='S'+str(SD.StarNr) + ' DE',ms=_ms, zorder=2)

    #-------------------------------------------------------------
    #fake Data
    if showFit:

        # This is the length of the fake data. From Index point it extends some integer amount orbit to front
        # and backwards it extends to 0, because of float orbit number for back direction
        # subtract first time measure for relative data
        DataLenFromIndex = SD.PosData.TIME[_in[0]] - SD.PosData.TIME[0] + FD.OrbitNumber * FD.OrbElem.Period
        fake_time = np.linspace(0,DataLenFromIndex, len(FD.PosPath[0]))
        # end of real data, in relative units
        END_Data = SD.PosData.TIME[-1] - SD.PosData.TIME[0]
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
            F[1].scatter(SD.RadData.TIME[_in[1]] - SD.PosData.TIME[0], SD.RadData.RVEL[_in[1]], label=r'Index', s=20, color='red', zorder=99) #vR - vz
            F[2].scatter(SD.PosData.TIME[_in[0]] - SD.PosData.TIME[0], SD.PosData.RA[_in[0]], label=r'Index', s=20, color='red', zorder=99) #RA - x
            F[3].scatter(SD.PosData.TIME[_in[0]] - SD.PosData.TIME[0], SD.PosData.DE[_in[0]], label=r'Index', s=20, color='red', zorder=99) #DE - y
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

# ------------------------------------------------------------------

# ARRAY STRUCTURE
# D:1 -- M:dim -- (X,Y,Z):3dim -- (VX,VY,VZ):3dim

def constructParVec(dim:int = 1):

    return np.array( [i for i in range(7*dim + 1)], dtype=np.float64 )

#@jit(nopython=True, cache=True)
def ParGetMass(Par:np.ndarray, dim:int):
    return Par[1:dim+1]
#@jit(nopython=True, cache=True)
def ParGetPos(Par:np.ndarray, dim:int):
    return Par[dim+1:4*dim+1]
#@jit(nopython=True, cache=True)
def ParGetVel(Par:np.ndarray, dim:int):
    return Par[4*dim+1:]


# ------------------------------------------------------------------


#@jit(nopython=True, cache=True)
def GetAccNewton(Mass:np.ndarray, Pos:np.ndarray, Vel:np.ndarray=None, Index:int = 0) -> np.ndarray:
    """
    """

    ownP = Pos[ 3*Index : 3*(Index+1) ]
    a = np.zeros(3)

    for i in range(len(Mass)): #enumerate( zip( Pos, Mass ) ):
        if i != Index:
            p = Pos[ 3*i : 3*(i+1) ]
            m = Mass[i]
            dist = p - ownP
            a += GLOB_G * m * dist / la.norm(dist)**3

    return a

#@jit(nopython=True, cache=True)
def GetAccVec(Mass:np.ndarray, Pos:np.ndarray, Vel:np.ndarray=None, dim:int=2) -> np.ndarray:
    """
    """

    av = np.empty(3*dim, dtype=np.float64)

    for i in range(dim):
        av[3*i:3*(i+1)] = GetAccNewton(Mass, Pos, Vel, i)

    return av

#@jit(nopython=True, cache=True)
def StormerVerlet(Par:np.ndarray, h:float, dim:int):
    """
    Numerical Integrator for n Bodies, based on Verlet algorithm. 

    Param
    ---
    Par : array
        Array with initial parameters of all Bodies (previous step of integration)

    h : float
        timestep delta t

    dim : int
        Number of Bodies that Par consists of. Used to cut Par array into slices

    Return
    ---
    Par : array
        new Array with coordinate and velocity updates to all bodies. Distance and Mass unchanged
    """

    Mass = Par[1:dim+1]
    pos = Par[dim+1:4*dim+1]
    vel = Par[4*dim+1:]

    newPos = pos + h * vel
    newPot = GetAccVec(Mass, newPos, vel, dim)
    newVel = vel + h * newPot

    Par[dim+1:4*dim+1] = newPos
    Par[4*dim+1:] = newVel

    return Par


def testIntegration(Par:np.ndarray, _fig):
    
    MainList = [Par]
    h = 10E-6
    curPar = np.copy(Par)

    for i in range(50000):
        curPar = StormerVerlet(curPar, h, 2)
        MainList.append(np.copy(curPar))

        ProgressBar(i, 50000, " F")

    
    _fig.clf()
    _tf = _fig.add_subplot(1,1,1)
    _tf.set_axisbelow(True)
    _tf.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    _tf.set_xlabel('x', {'size':14})
    _tf.set_ylabel('y', {'size':14})

    _tf.scatter([x[3] for x in MainList][0], [x[4] for x in MainList][0], label="obj1")
    _tf.plot([x[3] for x in MainList], [x[4] for x in MainList], label="obj1-line")
    _tf.plot([x[6] for x in MainList], [x[7] for x in MainList], label="obj2")
    _tf.legend(loc='best', fontsize=12)

    plt.show()

    #dfArr = np.array( [[x[3] for x in MainList], [x[4] for x in MainList], [x[5] for x in MainList],
    #                   [x[6] for x in MainList], [x[7] for x in MainList], [x[8] for x in MainList]] ).T

    #df = pd.DataFrame(dfArr, columns=["obj1X", "obj1Y", "obj1Z", "obj2X", "obj2Y", "obj2Z"])

    #import plotly.express as px
    #fig = px.line_3d(df, x="obj2X", y="obj2Y", z="obj2Z")
    #fig.show()



def getOrbitalElements(Par:np.ndarray, index:int, dim:int) -> OrbitElem:
    """
    Returns all Orbital Elements and the Period, packed into a data class
    Parameters
    ----------
    _parVec : Parameter Vector for current orbit
    Returns
    -------
    OrbitalElem Object
    """

    M = ParGetMass(Par, dim)[0]
    r0 = ParGetPos(Par, dim)[3*index:3*(index+1)]
    v0 = ParGetVel(Par, dim)[3*index:3*(index+1)]

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

def getOrbit(SD:StarContainer, Orb:OrbitElem, ParVec:ParVec, index:int, max_iter:float, stepsize:float, kwargs:dict={}) -> FitContainer:
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

    # exit when Stardata or Orbital Elements are not present
    if not Orb or not SD:
        return FitContainer(Param=None, success=False)

    # convert parameter vector to parameters
    Par = ParVec.ParameterVector
    
    #_M, _r, _v, _d = ParVecToParams(ParamVec)

    # switch the Integrator method
    # if 'method' in kwargs.keys():
    #     if kwargs['method'] == 'Schwarz':
    #         IntegratorToUse = VerletStepSchwarz
    #         PotFunc = potentialSchwarz
    #     else:
    #         IntegratorToUse = VerletStep
    #         PotFunc = potential
    # else:
    #     IntegratorToUse = VerletStep
    #     PotFunc = potential

    # try different positions of sgr a*
    # if 'varSGRA' in kwargs.keys():
    #     # convert to pc using the current distance while fitting
    #     SGRA_Pos = PosRadToReal(kwargs['varSGRA'], _d)

    # else:
    #     # default to (0,0,0)
    #     SGRA_Pos = np.array([0,0,0])
    
    #------------------------------------------------------------------------
    #   CURRENT VALUES

    PeriodFront = SD.PosData.TIME.iloc[-1] - SD.PosData.TIME.iloc[index]            # Time to render from index point to end
    PeriodBack  = SD.PosData.TIME.iloc[index] - SD.PosData.TIME.iloc[0]             # Time to render from beginning to index point
    CurPar = Par.copy()     # create copy of Parameter Vector
    CurF = potential()
    #r_cur = _r                                              # Position
    #v_cur = _v                                              # Velocity
    #f_cur = PotFunc(_r,_v,_M, SGRA_Pos)                     # init the potential
    #init_Er = _r / np.linalg.norm(_r)                       # Unit vector of initial position
    

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



# ------------------------------------------------------------------


#logging.basicConfig(level=logging.DEBUG, format='-- %(levelname)s - %(asctime)s - %(message)s')

x = ["a", "b", "c", "d"]
s = ParVec(x, True)
f = SMBH()

fileName = "SMBH/Data/OrbitData2017.txt"

#sd = GetStarData(fileName, 2)

#h = StormerVerlet(s, 1)
#print(h)
#print(s.ParameterVector)

dim = 2
arr = constructParVec(dim)
#print("arr",arr)

#print( ParGetPos(arr, dim) )
#print( ParGetVel(arr, dim) )

#StormerVerlet(arr, 1, dim)

#print("pos", ParGetPos(arr, dim) )
#print("vel", ParGetVel(arr, dim) )

ParamVec = np.array( [

0,
1E10, 1E0,
0, 0, 0,
5, 0, 0,
0, 0, 0,
0, 2000, 0

] )

#OE = getOrbitalElements(ParamVec, 1, 2)
#T = OE.Period

#print("Period", T)

#MAIN_fig = plt.figure(figsize=(8,8), dpi=100, clear=True)

#testIntegration(ParamVec, MAIN_fig)

#print( type(np.array_split(x[2], dim) ))

#gt = ParVecObj("lol")
#print(gt.ParameterVector)

#getOrbit(None,None,ParVec(["S2"],True), 0, 0,0)
