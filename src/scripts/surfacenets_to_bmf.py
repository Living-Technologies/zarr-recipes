#!/usr/bin/env python

import ngff_zarr

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
    
import vedo

import sys, pathlib, numpy
import binarymeshformat as bmf

"""
S. Frisken (Gibson), “Constrained Elastic SurfaceNets: Generating Smooth Surfaces from Binary Segmented Data”, Proc. MICCAI, 1998, pp. 888-898.
S. Frisken, “SurfaceNets for Multi-Label Segmentations with Preservation of Sharp Boundaries”, J. Computer Graphics Techniques, 2022.
"""

class Transformer:
    def __init__(self, metadata, shape):
        scale = metadata.scale
        self.dx = scale['x']
        self.dy = scale['y']
        self.dz = scale['z']
        lx = scale['x']*shape[-1]
        ly = scale['y']*shape[-2]
        lz = scale['z']*shape[-3]
        lengths = numpy.array((lz, ly, lx))
        longest = max( lengths )
        print(lengths, longest)

        self.factors = numpy.array(( scale['z']/longest, scale['y']/longest, scale['x']/longest ))
        half_pixel = self.factors * 0.5
        self.offsets = numpy.array( (-lz/longest/2, -ly/longest/2, -lx/longest/2) ) + half_pixel
        print(self.offsets, self.factors)
    def transform( self, pt ):
        """
            transforms img coordinates pt from z, y, x to normalized coordinates x, y, z
        """
        pt = self.factors * pt + self.offsets
        return numpy.array((pt[2], pt[1], pt[0]))

class ScaleTranslate:
    def __init__(self, shift, scale):
        self.factors = scale
        self.offsets = shift

def loadImageStack( inpth ):
    """
        Loads a zarr
    
    """
    ms = ngff_zarr.from_ngff_zarr(inpth)
    img = ms.images[0]

    #(t, z, y, x)
    np_stack = numpy.array(img.data[:, 0])
    transformer = Transformer( img, np_stack.shape[1:] )
    return np_stack, transformer
def toBmf(triangles, points):
    connections = set()
    n = triangles.shape[0]//4

    ptt = triangles.reshape((n, 4))
    #print(n, "triangles converted")
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
    points = numpy.flip(points, axis=1)
    flat_points = points.reshape( ( points.shape[0]*points.shape[1], ) )
    flat_connections = numpy.array( [con for con in connections] ).reshape( len(connections)*2 )

    flat_triangles = ptt[:, 1:].reshape( (n*3, ) )
    return bmf.Mesh(flat_points, flat_connections, flat_triangles)
def surfacNetsOutputToMesh(snets, transform):
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

    polygons = pda.GetPolygons()
    points = points*transform.factors + transform.offsets
    indexes = numpy.array(polygons)
    bmfMesh = toBmf( indexes, points )
    return bmfMesh

if True:
    
    inpath = pathlib.Path(sys.argv[1])
    time_stack, transformer = loadImageStack(inpath)
    print(transformer.offsets, "initilized")
    org_stack = None
    ot = None
    if len(sys.argv) > 2:
        org_stack, ot = loadImageStack(pathlib.Path(sys.argv[2]))

    for tp in range(time_stack.shape[0]):
        stack = time_stack[tp]
        print(stack.shape)
        #VTK seems to use x as the first index.     
        d0 = stack.shape[0]
        d1 = stack.shape[1]
        d2 = stack.shape[2]
        #VTK seems to use x as the first index.     
        keys = numpy.unique(stack)

        shapes = []
        if org_stack is not None:
            vol = vedo.Volume(org_stack[0], spacing=[ot.dz, ot.dy, ot.dx])
            shapes.append(vol)
        tracks = []
        for key in keys:
            if key == 0:
                continue

            z, y, x = numpy.where(stack==key)
            xlow = numpy.min(x) - 1
            xhigh = numpy.max(x) + 1
            ylow = numpy.min(y) - 1
            yhigh = numpy.max(y) + 1
            zlow = numpy.min(z) - 1
            zhigh = numpy.max(z) + 1
            w = xhigh - xlow + 1
            h = yhigh - ylow + 1
            d = zhigh - zlow + 1

            img = vtk.vtkImageData();
            img.SetDimensions( d, h, w )
            img.AllocateScalars(vtk.VTK_SHORT, 1)
            img.SetSpacing([transformer.dz, transformer.dy, transformer.dx]);
            scalars2 = img.GetPointData().GetScalars()
            scalars2.Fill(0)
            ntuples = scalars2.GetNumberOfTuples()
            for xi, yi, zi in zip(x, y, z):
                index = (xi - xlow)*d*h + (yi - ylow)*d + zi - zlow
                scalars2.SetTuple1( index, 1 )

            snets = vtk.vtkSurfaceNets3D()
            snets.SetInputData(img)
            snets.Update()

            op = snets.GetOutput()
            scale = transformer.factors
            low = numpy.array([zlow, ylow, xlow])
            offset = transformer.offsets + low*scale
            mesh = surfacNetsOutputToMesh(snets, ScaleTranslate(offset, scale))

            trk = bmf.Track("#999999_%s"%key)
            trk.addMesh(tp, mesh)
            tracks.append(trk)
            #polys = vedo.mesh.Mesh(op)
            #polys.shift(zlow, ylow, xlow);
            #polys.color(numpy.random.random((3,)))
            #shapes.append(polys)

        #cells = vedo.show(shapes)
        op = pathlib.Path( inpath.parent, "".join( [inpath.name.replace(".zarr", ""), "_%s_sn.bmf"%tp] ) )
        bmf.saveMeshTracks( tracks, op)
