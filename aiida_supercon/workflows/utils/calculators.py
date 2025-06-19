from aiida.engine import calcfunction
from aiida.orm import ArrayData, XyData, Float

import numpy
from scipy.interpolate import interp1d

@calcfunction
def calculate_Allen_Dynes_tc(a2f: ArrayData, mustar = 0.13) -> Float:
    w        = a2f.get_array('frequency')
    # Here we preassume that there are 10 smearing values for a2f calculation
    spectral = a2f.get_array('a2f')[:, 9]   
    mev2K    = 11.604525006157

    _lambda  = 2*numpy.trapz(numpy.divide(spectral, w), x=w)

    # wlog =  np.exp(np.average(np.divide(alpha, w), weights=np.log(w)))
    wlog     =  numpy.exp(2/_lambda*numpy.trapz(numpy.multiply(numpy.divide(spectral, w), numpy.log(w)), x=w))

    Tc = wlog/1.2*numpy.exp(-1.04*(1+_lambda)/(_lambda-mustar*(1+0.62*_lambda))) * mev2K


    return Float(Tc)

@calcfunction
def calculate_iso_tc(max_eigenvalue: XyData) -> Float:
    me_array = max_eigenvalue.get_array('max_eigenvalue')
    if me_array[:, 1].max() < 1.0:
        return Float(0.0)
    else:
        return Float(float(interp1d(me_array[:, 1], me_array[:, 0])(1.0)))
