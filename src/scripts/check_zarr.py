#!/usr/bin/env python 


#import ome_zarr, zarr, ome_zarr.io, ome_zarr.writer, ome_zarr.reader
import zarr

import numpy
import sys

"""
Opens a zarr file and prints things.

Should use a common library command.
"""

def loadZarr( location ):

    start = zarr.open_group(
       store=sys.argv[1],
       mode='r',
    )
    multiscales = start.attrs["multiscales"]
    arrays = [ start[ key ] for key in start.array_keys() ]
    return multiscales, arrays

def printMetadata( metadata ):
    axes = metadata["axes"]
    for axis in axes:
        unit = "NA"
        if "unit" in axis:
            unit = axis["unit"]
        print("    > %s, %s, %s"%(axis["name"], axis["type"], unit) )
    ds = metadata["datasets"]
    for dsmd in ds:
        for tf in dsmd["coordinateTransformations"]:
            tp = tf["type"]
            print( "  ", tp )
            if tp in tf:
                print( "    *", tf[ tp ] ) 

print("Checking zarr file: ", sys.argv[1])
metadata, array_data = loadZarr( sys.argv[1] )
for md, arrays in zip(metadata, array_data):
    print("Dataset: \"%s\""%arrays.name, arrays.shape )
    printMetadata(md)
print("finished checking!")
#url = ome_zarr.io.parse_url(sys.argv[1])
#reader = ome_zarr.reader.Reader(url)
#img = list(reader())[0]
#print(img)
#dask_stack = img.data[0]
#print(dask_stack.shape)
#np_stack = numpy.array(dask_stack)
#print(numpy.sum(np_stack))