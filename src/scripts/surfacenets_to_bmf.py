#!/usr/bin/env python

import ngff_zarr

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import skimage

DISPLAY=False
try:
    import vedo
except:
    DISPLAY=False

import sys, pathlib, numpy
import binarymeshformat as bmf

"""
S. Frisken (Gibson), “Constrained Elastic SurfaceNets: Generating Smooth Surfaces from Binary Segmented Data”, Proc. MICCAI, 1998, pp. 888-898.
S. Frisken, “SurfaceNets for Multi-Label Segmentations with Preservation of Sharp Boundaries”, J. Computer Graphics Techniques, 2022.
"""

class Transformer:
    """
        Builds a transform from the provided metadata. The default bmf format is normalized cordinates. Where the longest
        axis goes from -0.5 to 0.5 and the other axes go from -a/2l, a/2l where l is the length of the longest axis.

    """
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

        self.factors = numpy.array(( self.dz/longest, self.dy/longest, self.dx/longest ))
        half_pixel = self.factors * 0.5
        self.offsets = numpy.array( (-lz/longest/2, -ly/longest/2, -lx/longest/2) ) + half_pixel
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

def paddedCrop(img, low, high):
    cl = [0]*3
    ch = [0]*3
    pl = [0]*3
    ph = [0]*3
    for i in range(3):
        if low[i] < 0:
            cl[i] = 0
            pl[i] = 1
        else:
            cl[i] = low[i]

        if high[i] == img.shape[i]:
            ch[i] = img.shape[i] - 1
            ph[i] = 1
        else:
            ch[i] = high[i]

    arr = img[cl[0]:ch[0] + 1, cl[1]:ch[1] + 1, cl[2]:ch[2] + 1]
    pads = [(l, h) for l, h in zip(pl, ph)]
    return numpy.pad(arr, pads)

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

def normalizeTriangles(snets):
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetConsistency(True)
    nrm.SetSplitting(False)
    nrm.SetInputDataObject( snets.GetOutputDataObject(0) )
    nrm.NonManifoldTraversalOff()
    nrm.Update()
    polys = nrm.GetOutput()
    return polys

def imageOutputToMesh(polys, transform):

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
    print(transformer.offsets, "offsets")
    print(transformer.factors, "scaling")

    org_stack = None
    ot = None
    if len(sys.argv) > 2:
        org_stack, ot = loadImageStack(pathlib.Path(sys.argv[2]))
    flying_edges=True
    frames = time_stack.shape[0]

    mesh_folder = pathlib.Path( inpath.parent, "%s_sn"%inpath.name.replace(".zarr", "") )

    if not mesh_folder.exists():
        mesh_folder.mkdir()

    for tp in range(frames):
        stack = time_stack[tp]
        print("image shape: ", stack.shape)
        keys = numpy.unique(stack)

        shapes = []


        if DISPLAY is not None and org_stack is not None:
            vol = vedo.Volume(org_stack[0], spacing=(transformer.dz, transformer.dy, transformer.dx))
            shapes.append(vol)
        tracks = []

        spacing = numpy.array((transformer.dz, transformer.dy, transformer.dx))
        smax = numpy.max(spacing)
        spacing = spacing/smax

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

            zc = z - zlow
            yc = y - ylow
            xc = x - xlow

            if len(zc) == 0:
                continue
            img = vtk.vtkImageData();
            img.SetDimensions( d, h, w )
            img.AllocateScalars(vtk.VTK_SHORT, 1)
            img.SetSpacing(spacing);
            scalars2 = img.GetPointData().GetScalars()
            scalars2.Fill(0)
            ntuples = scalars2.GetNumberOfTuples()
            for xi, yi, zi in zip(xc, yc, zc):
                index = (xi)*d*h + (yi)*d + zi
                scalars2.SetTuple1( index, 1 )
            snets = None
            if flying_edges:
                snets = vtk.vtkDiscreteFlyingEdges3D()
            else:
                snets = vtk.vtkSurfaceNets3D()
                snets.SetSmoothing(False)
                snets.SetTriangulationStrategyToMinArea()
                snets.SetOutputMeshTypeToTriangles()

            snets.SetInputData(img)
            snets.Update()

            triangles = normalizeTriangles(snets)

            low = numpy.array([zlow, ylow, xlow])
            low_scaled = low*spacing

            low_normal = low*transformer.factors
            offset = transformer.offsets + low_normal
            factors = transformer.factors/spacing
            mesh = imageOutputToMesh(triangles, ScaleTranslate(offset, factors ))

            trk = bmf.Track("#999999_%s"%key)
            trk.addMesh(tp, mesh)
            tracks.append(trk)

            if DISPLAY:
                polys = vedo.mesh.Mesh(triangles)
                polys.shift(low_scaled[0], low_scaled[1], low_scaled[2]);
                polys.color(numpy.random.random((3,)))
                shapes.append(polys)

        if DISPLAY:
            cells = vedo.show(shapes)

        op = pathlib.Path( mesh_folder, "frame-%d.bmf"%tp )
        bmf.saveMeshTracks( tracks, op)
