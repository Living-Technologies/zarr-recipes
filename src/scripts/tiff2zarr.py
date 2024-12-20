#!/usr/bin/env python3


import ome_zarr, zarr, ome_zarr.io, ome_zarr.writer
from tifffile import TiffFile
import numpy
import os, sys

"""
Opens a tiff file using tifffile and writes a .zarr folder. 

It is assumed that the tiff was created using imagej. As such that is
how the spatial data is written.

The output will have time, channel, slices, y, x shape.
"""

def getGenericCalibrations( image, tags):
    """
        FIXME: the xy resolution tags are not in this version of tifffile!?
    """
    ttags= image.pages.get(0).tags
    for tag in ttags:
        if tag.name == "XResolution":
            tags["x_resolution"] = tag.value[1]/tag.value[0]
        if tag.name == "YResolution":
            tags["y_resolution"] = tag.value[1]/tag.value[0]

def getImageJCalibration(img, tags = None):
    """
        Pulls apart the "info" string based on imagej style tiff stacks, and
        populates tags. 
        
        @param img Tifffile loaded data
        @param tags target dictionary for output.
        
        Return:
            returns tags, or a new dictionary if tags omitted or None.
    """
    
    
    
    if tags is None:
        tags = img.imagej_metadata
    else:
        for key in img.imagej_metadata:
            tags[key] = img.imagej_metadata[key]
    
    return tags

def loadImage(imageFile):
    """
        Loads the image file and returns it as 'frames, channels, z, y, x' data. 

        Requires a tiff file created by imagej, the format is assumed to be (frames, slices, channels, ...)
        
        Args:
            imageFile: path to file to be loaded. converts to str.
        
    """
    imageFile = str(imageFile)
    tags={}
    with TiffFile(imageFile) as tiff:
        data = []
        for p in tiff.pages:
            data.append(p.asarray())
        data = numpy.array(data)
        if tiff.is_imagej:
            getGenericCalibrations(tiff, tags)
            getImageJCalibration(tiff, tags)
            frames = tags.get("frames", 1)
            slices = tags.get("slices", 1)
            channels = tags.get("channels", 1)
            data = data.reshape((frames, slices, channels, data.shape[-2], data.shape[-1]))
            data = numpy.rollaxis(data, 2, 1)
            if "spacing" not in tags:
                if "Step" in tags:
                    print("step found")
                    tags["spacing"] = tags["Step"].split()[0]
                else:
                    tags["spacing"] = 1.0
                    print("step not found!");
                    for key in tags:
                        print(key)
                    print(tags["Info"]["Color"])
        return data, tags

def writeZarr(img, tags, path):
    """
        Creates a zarr file with spacial information derived from the provided tags.
    """
    # prepare metadata
    #print(tags)
    pix_size_x = tags['x_resolution']
    pix_size_y = tags['y_resolution']
    pix_size_z = tags['spacing']
    time_interval = tags['finterval']
    
    xorigin = tags.get('xorigin', 0)
    yorigin = tags.get('yorigin', 0)
    zorigin = tags.get('zorigin', 0)
        
    unit = tags['unit']
    
    coordinate_transformations = [
        [{"scale": [time_interval, 1, pix_size_z, pix_size_y, pix_size_x], "type": "scale"}, {'translation': [0.0, 0.0, zorigin, yorigin, xorigin], 'type': 'translation'}]
    ]
    
    axes = [
        {
            "name": "t",
            "type": "time",
            "unit": "second"
        },
        {
            "name": "c",
            "type": "channel"
        },
        {
            "name": "z",
            "type": "space"
        },
        {
            "unit": "micrometer",
            "name": "y",
            "type": "space"
        },
        {
            "unit": "micrometer",
            "name": "x",
            "type": "space"
        }
    ]
    os.mkdir(path)
    store = ome_zarr.io.parse_url(path, mode="w").store
    root = zarr.group(store=store)
    ome_zarr.writer.write_multiscale(pyramid=[img], group=root, coordinate_transformations=coordinate_transformations,
            axes=axes, storage_options=dict( chunks = ( 1, 1, *img.shape[2:] ) ), compression=None )
      
if __name__=="__main__":
    inpath = sys.argv[1]
    outpath = inpath.replace(".tif", "-py.zarr")
    
    image, tags = loadImage( inpath )
    writeZarr(image, tags, outpath)
