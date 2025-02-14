#!/usr/bin/env python3 

import ome_zarr, zarr, ome_zarr.io, ome_zarr.writer, ome_zarr.reader
import cellpose.models
import numpy
import sys, pathlib

def saveZarrPrediction( listOfMasks, metadata, path ):
    
    out = numpy.array(listOfMasks)
    shaped = out.reshape( (out.shape[0], 1, *out.shape[1:]))
    print(shaped.shape)
    
    path.mkdir()
    store = ome_zarr.io.parse_url(path, mode="w").store
    root = zarr.group(store=store)
    ome_zarr.writer.write_multiscale(pyramid=[shaped], group=root, coordinate_transformations=metadata['coordinateTransformations'],
            axes=metadata['axes'], storage_options=dict( chunks = ( 1, 1, *shaped.shape[2:] ) ), compression=None )

def loadImageStack( inpth ):
    """
        Attempts to load the provided path and coerce the data into a 
        (t, c, z, y, x) shape
        
        #TODO migrate to library common.
        #TODO get data from xarray instead of manually depending on 0.4 zarr
    
    """
    url = ome_zarr.io.parse_url(inpth)
    reader = ome_zarr.reader.Reader(url)
    img = list(reader())[0]
    dask_stack = img.data[0]
    np_stack = numpy.array(dask_stack)
    #both forms of saving that I use are either (time, channel, slice, y, x) or (time, slice, y, x) 
    ndim = len(np_stack.shape)
    print(img.metadata)
    if ndim == 3:
        #one channel, one frame.
        img.metadata["axes"].insert( 0,  )
        img.metadata["axes"].insert( 0, {"type":"time", "name":"t"} )
        img.metadata['coordinateTransformations'][0][0]['scale'].insert(0, 1)
        img.metadata['coordinateTransformations'][0][0]['scale'].insert(0, 1)
        img.metadata['coordinateTransformations'][0][0]['translation'].insert(0, 0)
        img.metadata['coordinateTransformations'][0][0]['translation'].insert(0, 0)
        add_axis(img.metadata, {"type":"channel", "name":"c"}, 0)
        add_axis(img.metadata, {"type":"time", "name":"t"}, 0)
        np_stack = np_stack.reshape((1, 1, *np_stack.shape))        
    elif ndim == 4:
        #Either time or channel.
        first_axis = img.metadata["axes"][0]
        if first_axis["name"] == "c":
            #channel defined
            add_axis( img.metadata, {"type":"time", "name":"t"}, 0)
            np_stack = np_stack.reshape((1, *np_stack.shape[0:]))
        else:
            shp = np_stack.shape
            np_stack = np_stack.reshape((shp[0], 1, *shp[1:]))
            add_axis(img.metadata, {"type":"channel", "name":"c"}, 1 )
    print(img.metadata)
    stack = [y for y in np_stack]
    return stack, img.metadata
def add_axis( metadata, axis, index):
    for dst in metadata['coordinateTransformations']:
        for trans in dst:
            if 'scale' in trans:
                trans['scale'].insert(index, 1)
            if 'translation' in trans:
                trans['translation'].insert(index, 0)
    metadata['axes'].insert( index, axis )

if __name__=="__main__":
    inpth = pathlib.Path(sys.argv[2])
    mdlpth = pathlib.Path(sys.argv[1])
    opth = pathlib.Path( inpth.parent, inpth.name.replace(".zarr", "_cp-masks.zarr") )

    stack, metadata = loadImageStack( inpth )

    transformations = metadata['coordinateTransformations']

    level_1 = transformations[0]

    resolution = level_1[0]

    xy = resolution['scale'][-1]
    z = resolution['scale'][-3]


    
    model = cellpose.models.CellposeModel( gpu=True, pretrained_model="cyto3")

    print("z ", "xy ", z, xy, stack[0].shape)
    stack = [ s[0] for s in stack ]
    y = model.eval(stack, z_axis=0, do_3D = True,dP_smooth=3, anisotropy= z/xy, diameter=30)

    saveZarrPrediction(y[0], metadata, opth)
