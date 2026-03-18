#!/usr/bin/env python

import ngff_zarr
import cellpose
import numpy
from cellpose import models
import dask
import sys, pathlib
import skimage

import tifffile

class ScaleImage:
    def __init__(self,shape, scale):
        w = shape[-1]
        h = shape[-2]
        d = shape[-3]

        self.out_shape = (int(scale[0]*d), int(scale[1]*h), int(scale[2]*w))
    def getOutputShape( self ):
        return self.out_shape
    def scale(self, image):
        n_dims = image.ndim
        leave = n_dims - 3
        complete = (*image.shape[0:leave], *self.out_shape)
        return skimage.transform.resize(image, complete)

inline = False

def loadTiffData(fname):
    inpth = pathlib.Path(fname)
    oname = inpth.stem.replace(" ", "_")
    opth = pathlib.Path( inpth.parent, "%s_cp-masks.zarr"%oname )

    tf = tifffile.TiffFile(fname)
    data = tf.asarray()
    data = numpy.rollaxis(data, 2, 1)
    md = tf.imagej_metadata
    first = 0
    last = data.shape[0]

    indexes = [i for i in range(first, last)]

    res_tag = tf.pages.get(0).tags["XResolution"]
    xy = res_tag.value[1]/res_tag.value[0]
    z = md["spacing"]

    scaler = ScaleImage( data.shape, (z, xy, xy) ) #scale to 1um
    print("scales: ", xy, z)
    new_scales = {"t":1, "c":1, "z":1.0, "y":1.0, "x":1.0}
    translation = { x : 0 for x in new_scales }
    model = models.CellposeModel( gpu=True )

    anisotropy = new_scales["z"]/new_scales["x"]

    def torchit( block_id, model=model, data=data, anisotropy=anisotropy, scaler = scaler):
        #get scale version of image.
        simg = scaler.scale(numpy.array(data[block_id[0]]))
        y = model.eval( simg, z_axis=1,channel_axis=0, do_3D = True, anisotropy=anisotropy )
        #recover the time axis
        pred = numpy.expand_dims(y[0], (0, 1))
        return pred

    print("preparing")
    chunks = ( (1, )*len(indexes), 1,  *scaler.getOutputShape() )

    print(chunks)

    out = dask.array.map_blocks( torchit, dtype="uint16", chunks = chunks )
    print( "out stack: ", out.shape )

    oi = ngff_zarr.to_ngff_image(out, dims=[d for d in new_scales], translation=translation, scale=new_scales)
    print("multiscales")
    next_ms = ngff_zarr.to_multiscales( oi, cache=False, chunks=(1, 1, *data.shape[2:]) )
    print("writing")
    ngff_zarr.to_ngff_zarr( opth, next_ms, overwrite=False )
def loadZarrData(fname):
    #dask.config.set(workers=2)
    inpth = pathlib.Path(fname)
    opth = pathlib.Path( inpth.parent, "%s_cp-masks.zarr"%inpth.stem )

    ms = ngff_zarr.from_ngff_zarr(inpth)
    img = ms.images[0]

    if inline:
        data = dask.array.from_zarr(pathlib.Path(inpth, ms.metadata.datasets[0].path ), inline = True)
    else:
        data = img.data
    first = 0
    last = data.shape[0]

    indexes = [i for i in range(first, last)]

    xy = img.scale["x"]
    z = img.scale["z"]

    scaler = ScaleImage( img.data.shape, (z, xy, xy) ) #scale to 1um
    print("scales: ", xy, z)
    new_scales = {"t":1, "c":1, "z":1.0, "y":1.0, "x":1.0}

    model = models.CellposeModel( gpu=True )

    anisotropy = new_scales["z"]/new_scales["x"]

    def torchit( block_id, model=model, data=data, anisotropy=anisotropy, scaler = scaler):
        #get scale version of image.
        simg = scaler.scale(numpy.array(data[block_id[0]]))
        y = model.eval( simg, z_axis=1,channel_axis=0, do_3D = True, anisotropy=anisotropy )
        #recover the time axis
        pred = numpy.expand_dims(y[0], (0, 1))
        return pred

    print("preparing")
    chunks = ( (1, )*len(indexes), 1,  *scaler.getOutputShape() )

    print(chunks)

    out = dask.array.map_blocks( torchit, dtype="uint16", chunks = chunks )
    print( "out stack: ", out.shape )

    oi = ngff_zarr.to_ngff_image(out, dims=img.dims, translation=img.translation, scale=new_scales)
    print("multiscales")
    next_ms = ngff_zarr.to_multiscales( oi, cache=False, chunks=(1, 1, *img.data.shape[2:]) )
    print("writing")
    ngff_zarr.to_ngff_zarr( opth, next_ms, overwrite=False )

if __name__=="__main__":
    #dask.config.set(workers=2)
    if sys.argv[1].endswith('.zarr'):
        loadZarrData(sys.argv[1])
    else:
        loadTiffData(sys.argv[1])
