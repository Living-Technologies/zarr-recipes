#!/usr/bin/env python 


import ome_zarr, zarr, ome_zarr.io, ome_zarr.writer, ome_zarr.reader
import numpy
import sys

"""
Opens a zarr file and prints things.

Should use a common library command.
"""
url = ome_zarr.io.parse_url(sys.argv[1])
reader = ome_zarr.reader.Reader(url)
img = list(reader())[0]
print(img)
dask_stack = img.data[0]
print(dask_stack.shape)
np_stack = numpy.array(dask_stack)
print(numpy.sum(np_stack))