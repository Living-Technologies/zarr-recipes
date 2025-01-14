#!/usr/bin/env python 


#import ome_zarr, zarr, ome_zarr.io, ome_zarr.writer, ome_zarr.reader
import ltzarr

import numpy
import sys

"""
Opens a zarr file and prints things.

Should use a common library command.
"""

print("Checking zarr file: ", sys.argv[1])
mses = ltzarr.loadZarr(sys.argv[1])
for ms in mses:
    ms.summary()