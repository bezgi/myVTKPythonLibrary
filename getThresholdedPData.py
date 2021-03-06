#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2012-2017                               ###
###                                                                  ###
### University of California at San Francisco (UCSF), USA            ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

########################################################################

def getThresholdedPData(
        pdata,
        field_support,
        field_name,
        threshold_value,
        threshold_by_upper_or_lower,
        verbose=0):

    mypy.my_print(verbose, "*** getThresholdedPData ***")

    thresholded_ugrid = getThresholdedUGrid(
        pdata,
        field_support,
        field_name,
        threshold_value,
        threshold_by_upper_or_lower,
        False)
    thresholded_pdata = ugrid2pdata(thresholded_ugrid, False)

    return thresholded_pdata
