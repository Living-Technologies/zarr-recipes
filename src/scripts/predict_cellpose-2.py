#!/usr/bin/env python

import ngff_zarr
import cellpose
import numpy
from cellpose import models
import dask
import sys, pathlib

if __name__=="__main__":
    dask.config.set(workers=2)
    inpth = pathlib.Path(sys.argv[1])
    opth = pathlib.Path( inpth.parent, inpth.name.replace(".zarr", "_cp-masks.zarr") )

    ms = ngff_zarr.from_ngff_zarr(inpth)
    img = ms.images[0]

    indexes = [i for i in range(img.data.shape[0])]

    xy = img.scale["x"]
    z = img.scale["z"]

    model = models.CellposeModel( gpu=True )


    def torchit( block_id, model=model, img=img, anisotropy=z/xy):
        y = model.eval(numpy.array(img.data[block_id[0],0]), z_axis=0, do_3D = True, anisotropy=anisotropy )
        pred = numpy.expand_dims(y[0], (0, 1))
        print("shape after prediction: ", pred.shape)
        return pred

    print("preparing")
    chunks = ( (1, )*len(indexes), 1,  *img.data.shape[2:] )
    print(chunks)
    out = dask.array.map_blocks( torchit, dtype="uint16", chunks = chunks )
    print( "out stack: ", out.shape )

    oi = ngff_zarr.to_ngff_image(out, dims=img.dims, translation=img.translation, scale=img.scale)
    print("multiscales")
    next_ms = ngff_zarr.to_multiscales( oi, cache=False, chunks=(1, 1, *img.data.shape[2:]) )
    print("writing")
    ngff_zarr.to_ngff_zarr( opth, next_ms, overwrite=False )
