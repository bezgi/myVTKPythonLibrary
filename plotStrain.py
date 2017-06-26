import numpy
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk
import dolfin_dic as ddic
import os

#read the original data - in World Coordinates
x = os.listdir("../DispInWorldCoords")
mesh2 = myvtk.readUGrid("../MeVisLab/Mesh_CspammCoords_ED_LV.vtk", 10)

farray_name = 'displacement'
S_in = numpy.diag(list(numpy.loadtxt("../MeVisLab/Image_Cspamm_Scaling.dat"))+[1])
W_in = numpy.loadtxt("../MeVisLab/Image_Cspamm_WorldMatrix.dat")
W_in_new = W_in[0:3,0:3]
M = numpy.dot(S_in, numpy.linalg.inv(W_in))

images_basename = "projected_p0_31012017_PHASE"
mesh_folder = "."
working_folder = "."
working_basename = images_basename
mesh_basename = "Mesh_CspammCoords_ED_LV"

for files in x:
   print ("Converting file: "+ files)
   data = myvtk.readSGrid("../DispInWorldCoords/"+files, 10)
   dataset = data.GetPointData()
   n_points = data.GetNumberOfPoints()
   print ("number of points: "+str(n_points))
   assert (dataset.HasArray(farray_name)), "mesh has no array named"+displacement
   farray_type = dataset.GetArray(farray_name).GetDataTypeAsString()
   farray_n_components = dataset.GetArray(farray_name).GetNumberOfComponents()
   myvtk.moveMeshWithWorldMatrix(data, M)

   for k_point in xrange(n_points):
       n_components = len(dataset.GetArray(farray_name).GetTuple(k_point))
       if (n_components == 3):
           rotated = numpy.dot(numpy.linalg.inv(W_in_new), dataset.GetArray(farray_name).GetTuple(k_point))
           dataset.GetArray(farray_name).SetTuple(k_point,tuple(rotated))

   myvtk.writeSGrid(data, "./"+"converted_"+files, 10)
   mesh1 = myvtk.readSGrid("./"+"converted_"+files, 10)

   myvtk.addMappingToPointData(mesh1, "point", mesh2, ["displacement"],"./"+"projected_"+files )


for files in x:
   ddic.compute_strains(working_folder=working_folder,working_basename=working_basename,working_ext="vtk",disp_array_name="displacement_avg",mesh_w_local_basis_folder=mesh_folder,mesh_w_local_basis_basename=mesh_basename,CYL_or_PPS="PPS",write_strains=1,plot_strains=1,verbose=1)


