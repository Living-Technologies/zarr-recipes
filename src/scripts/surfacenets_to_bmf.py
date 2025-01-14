#!/usr/bin/env python

import ltzarr

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
    
import sys, pathlib, numpy
import binarymeshformat as bmf

"""
This script will load a version 0.4 zarr file of labeled images, find the meshes using vtkSurfaceNets3D,
then save the resulting mesh as a bmf file. The requrements:

Requirements

- vtk 9.4.0 (installed from vedo)
- ome-zarr 0.10.2
- binarymeshformat 1.0

-----------------------------------------------------------

Surface Nets references from vtk documentation.

S. Frisken (Gibson), “Constrained Elastic SurfaceNets: Generating Smooth Surfaces from Binary Segmented Data”, Proc. MICCAI, 1998, pp. 888-898.
S. Frisken, “SurfaceNets for Multi-Label Segmentations with Preservation of Sharp Boundaries”, J. Computer Graphics Techniques, 2022.
"""

class Transformer:
    """
        This transform pixel coordinate based coordinates to normalized points.
        The longest real unit axis goes from -0.5 to 0.5. The other axis are 
        scaled and centered to the same value.
        
    """
    def __init__(self, metadata, shape):
        scale = metadata["coordinateTransformations"][0][0]["scale"][-3:]
        lx = scale[-1]*shape[-1]
        ly = scale[-2]*shape[-2]
        lz = scale[-3]*shape[-3]
        lengths = numpy.array((lz, ly, lx))
        long = max( lengths )
        half_pixel = 0.5*numpy.array(scale)/long
        self.factors = numpy.array(( scale[0]/long, scale[1]/long, scale[2]/long ))
        self.offsets = numpy.array( (-lz/long/2, -ly/long/2, -lx/long/2) ) + half_pixel
        self.scale = scale
    def transform( self, pt ):
        """
            transforms img coordinates pt from z, y, x to normalized coordinates x, y, z
        """
        pt = self.factors * pt + self.offsets
        return numpy.array((pt[2], pt[1], pt[0]))
        
def loadImageStack( inpth ):
    """
        #TODO move to common function
        Loads a zarr
    """
    url = ome_zarr.io.parse_url(inpth)
    reader = ome_zarr.reader.Reader(url)
    img = list(reader())[0]
    dask_stack = img.data[0]
    transformer = Transformer( img.metadata, dask_stack.shape[-3:] )
    return dask_stack, transformer


def toBmf(triangles, points):
    connections = set()
    n = triangles.shape[0]//4
    
    ptt = triangles.reshape((n, 4))
    print(n, "triangles converted")
    for i in range(n):
        t = ( triangles[4*i + 1], triangles[4*i + 2], triangles[4*i + 3] )
        for i in range(3):
            i0 = t[i]
            i1 = t[ ( i + 1 ) % 3]
            if i0 > i1:
                connections.add( (i1, i0) )
            elif i1 > i0:
                connections.add( (i0, i1) )
            else:
                raise Exception("duplicate indexes! %s"%( t, ))
                #print(t)

    flat_points = points.reshape( ( points.shape[0]*points.shape[1], ) )
    flat_connections = numpy.array( [con for con in connections] ).reshape( len(connections)*2 )
    
    flat_triangles = ptt[:, 1:].reshape( (n*3, ) )
    
    return bmf.Mesh(flat_points, flat_connections, flat_triangles)


def getMeshes( stack ):
    """
        Uses VTK SurfaceNets3D to generate pixel coordinate meshes.
        
        stack: 3 dimensional labeled array. Expecting z, y, x dimensions, but
               shouldn't matter.
    """
    #VTK seems to use x as the first index.
    d0 = stack.shape[0]
    d1 = stack.shape[1]
    d2 = stack.shape[2]

    img = vtk.vtkImageData();
    img.SetDimensions( stack.shape )
    img.AllocateScalars(vtk.VTK_SHORT, 1)
    scalars2 = img.GetPointData().GetScalars()

    scalars2.Fill(0)
    
    # Region 1
    for i in range(d0):
        for j in range(d1):
            for k in range(d2):
                index = k*d1*d0 + j*d0 + i
                scalars2.SetTuple1( index,stack[i,j,k] )
    
    snets = vtk.vtkSurfaceNets3D()
    snets.SetInputData(img)
    #snets.SmoothingOff()
    snets.SetOutputMeshTypeToTriangles()
    snets.Update()
    
    #The normals are not all the same direction.
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetConsistency(True)
    nrm.SetSplitting(False)
    nrm.SetInputDataObject( snets.GetOutputDataObject(0) )
    nrm.Update()
    polys = nrm.GetOutput()
    #polys = snets.GetOutput()
    pda = dsa.WrapDataObject(polys)
    points = numpy.array(pda.GetPoints())
    
    return pda.GetPolygons(), points
    
    
if __name__=="__main__":
    
    inpath = pathlib.Path(sys.argv[1])
    time_stack, transformer = loadImageStack(inpath)
    time_stack = time_stack[0]
    tracks = {}
    print("image shape: ", time_stack.shape)
    for tp in range(time_stack.shape[0]):
        stack = numpy.array( time_stack[tp] )
        #VTK seems to use x as the first index.     
        labels = numpy.unique(stack)
        for lbl in labels[1:]:
            
            z, y, x = numpy.where(stack==lbl)
            lx = max(0, numpy.min(x) - 1)
            hx = numpy.max(x) + 2
            ly = max(0, numpy.min(y) - 1)
            hy = numpy.max(y) + 2
            lz = max(0, numpy.min(z) - 1)
            hz = numpy.max(z) + 2
            lbl_stack = stack[lz:hz, ly:hy, lx:hx]
            lbl_stack = (lbl_stack==lbl)*1
            print(lx, hx, ly, hy, lz, hz, "no mesh!", lbl)
            polygons, points = getMeshes(lbl_stack)
            
            
            print("points", points.shape)
            if not points.shape:
                print(lx, hx, ly, hy, lz, hz, "no mesh!", lbl)
                continue
            ky = "#999999-%s"%lbl
            if ky in tracks:
                trk = tracks[ky]
            else:
                trk = bmf.Track(ky)
                tracks[ky] = trk
            points[:, 0] = points[:, 0] + lz
            points[:, 1] = points[:, 1] + ly
            points[:, 2] = points[:, 2] + lx
            for i in range(points.shape[0]):
                points[i] = transformer.transform(points[i])
            indexes = numpy.array(polygons)
            bmfMesh = toBmf( indexes, points )
            trk.addMesh( tp, bmfMesh )
         
    fin = [ tracks[ky] for ky in tracks]
    op = pathlib.Path( inpath.parent, "".join( [inpath.name.replace(".zarr", ""), ".bmf"] ) )
    bmf.saveMeshTracks( fin, op)
                    
