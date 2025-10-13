#!/usr/bin/env python3

import sys
import ngff_zarr


scale_factors = [{'x':2, 'y':2, 'z':1}, {'x':4, 'y':4, 'z':1},{'x':8, 'y':8, 'z':1},{'x':16, 'y':16, 'z':2},{'x':32, 'y':32, 'z':2}]

location = sys.argv[1]
ms = ngff_zarr.from_ngff_zarr(location)
location2 = location.replace(".zarr", "-mr.zarr")
multiscales = ngff_zarr.to_multiscales(ms.images[0], scale_factors=scale_factors)
for img in multiscales.images:
    print(img.scale, img.data.shape)
ngff_zarr.to_ngff_zarr(location2, multiscales, overwrite=True, version="0.4")