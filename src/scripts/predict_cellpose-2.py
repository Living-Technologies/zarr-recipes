#!/usr/bin/env python

import ngff_zarr
import cellpose
import numpy
from cellpose import models
import dask
import sys, pathlib
import skimage

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

if __name__=="__main__":
    #dask.config.set(workers=2)
    inpth = pathlib.Path(sys.argv[1])
    opth = pathlib.Path( inpth.parent, inpth.name.replace(".zarr", "_cp-masks.zarr") )

    ms = ngff_zarr.from_ngff_zarr(inpth)
    img = ms.images[0]

    first = 0
    last = img.data.shape[0]

    indexes = [i for i in range(first, last)]

    xy = img.scale["x"]
    z = img.scale["z"]

    scaler = ScaleImage( img.data.shape, (z, xy, xy) ) #scale to 1um

    new_scales = {"t":1, "c":1, "z":1.0, "y":1.0, "x":1.0}

    model = models.CellposeModel( gpu=True )

    anisotropy = new_scales["z"]/new_scales["x"]
    def torchit( block_id, model=model, img=img, anisotropy=anisotropy, scaler = scaler):
        simg = scaler.scale(numpy.array(img.data[block_id[0],0:2]))
        y = model.eval( simg, z_axis=1,channel_axis=0, do_3D = True, anisotropy=anisotropy )
        pred = numpy.expand_dims(y[0], (0, 1))
        print("shape after prediction: ", pred.shape)
        return pred

    print("preparing")
    chunks = ( (1, )*len(indexes), 1,  *scaler.getOutputShape() )

    print(chunks)

    out = dask.array.map_blocks( torchit, dtype="uint16", chunks = chunks, meta=numpy.array((), dtype="int16"))
    print( "out stack: ", out.shape )

    oi = ngff_zarr.to_ngff_image(out, dims=img.dims, translation=img.translation, scale=new_scales)
    print("multiscales")
    next_ms = ngff_zarr.to_multiscales( oi, cache=False, chunks=(1, 1, *img.data.shape[2:]) )
    print("writing")
    ngff_zarr.to_ngff_zarr( opth, next_ms, overwrite=False )
