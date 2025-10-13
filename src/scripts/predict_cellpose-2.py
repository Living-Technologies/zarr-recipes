#!/usr/bin/env python

import ngff_zarr
import cellpose
import numpy
from cellpose import models
import dask
import sys, pathlib

if __name__=="__main__":
    dask.config.set(workers=1)
    inpth = pathlib.Path(sys.argv[1])
    opth = pathlib.Path( inpth.parent, inpth.name.replace(".zarr", "_cp-masks.zarr") )

    ms = ngff_zarr.from_ngff_zarr(inpth)
    img = ms.images[0]

    indexes = [0, 1, 2]

    xy = img.scale["x"]
    z = img.scale["z"]

    model = models.CellposeModel( gpu=False )


    def torchit( block_id, model=model, img=img, anisotropy=z/xy):
        y = model.eval(numpy.array(img.data[block_id[0],0]), z_axis=0, do_3D = True, anisotropy=anisotropy )
        return y

    print("preparing")
    out = dask.array.map_blocks( torchit, dtype="int8", chunks = ((1,)*len(indexes) ,1, *img.data.shape[2:]) )
    print( "out stack: ", out.shape )

    oi = ngff_zarr.to_ngff_image(out, dims=img.dims, translation=img.translation, scale=img.scale)
    print("multiscales")
    next_ms = ngff_zarr.to_multiscales( oi, cache=False, chunks=(1, 1, *img.data.shape[2:]) )
    print("writing")
    ngff_zarr.to_ngff_zarr( opth, next_ms, overwrite=False )
