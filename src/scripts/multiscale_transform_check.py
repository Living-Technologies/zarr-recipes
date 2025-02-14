#!/usr/bin/env python3

"""
This is a test to make sure that the transforms are correct
for multiscales.

The idea is that the transform will transform a coordinate to "real space".
Any coordinate will have the same "real space" value in each of the multiscales.

 r = ( p + t )*s
 
 p, pixel location. t is the translate portion of the transform. s is the scale.
 
 For a different multiscale there will be the same transforms available.
 
 r = ( p' + t' )*s'
 
 The p and p' represent pixel coordinates of the same volumes,  
 so when p = 0 then p' = 0. 

    t's' = ts
 
The volume cannot change so p'=p/f.

    (p' + t') = 0 when p + t = 0
    
    p/f + t' = 0 and combine to get t' = t/f

There will be a t, s and a p', t', s' corresponding to the scale and the offset of the 
scaled transform.

    (p' + t')*s' = (p + t)*s

The transformation must work for multiple points so s` = s*f

When p' = t', then p = t. 
"""

import ngff_zarr, numpy
import ltzarr


def createImage():
    arr = numpy.zeros( (3, 2, 32, 64, 64), dtype="uint8")
    dims = ["t", "c", "z", "y", "x"]
    scale = { "t":60, "c":1, "z":2, "y":0.35, "x":0.35 }
    translate = { "t":0, "c":0, "z":10, "y":20, "x":30 }
    img = ngff_zarr.ngff_image.NgffImage( arr, dims, scale=scale, translation=translate)
    return img


image = createImage();
scale_factors = [{'x':2, 'y':2, 'z':1}]
multiscales = ngff_zarr.to_multiscales(image, scale_factors=scale_factors)

os = multiscales.images[0].scale
ot = multiscales.images[0].translation

ns = multiscales.images[1].scale
nt = multiscales.images[1].translation 
print(os, ns)

print( os["x"] == ns["x"]/2, os["y"] == ns["y"]/2, os["z"] == ns["z"] )
print( ot["x"] == nt["x"]*2, ot["y"] == nt["y"]*2, os["z"] == ns["z"] )