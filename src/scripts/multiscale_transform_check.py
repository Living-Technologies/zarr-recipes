#!/usr/bin/env python3

"""
# Multiscale transformation checks.

This is a test to make sure that the transforms are correct
for multiscales.

There are two version 

  r = ( p + t ) * s
  r = p*s + t

## Transform is in scaled units.

The coordnates represent the center of a pixel.
The transform will transform a coordinate to "real space".

  r = s*p + t

p, pixel location. t is the translate portion of the transform. s is the scale.

Each scale will have it's own set of pixels, transform, and scale. pi, ti, si respectivly.
s0 is the original scale factor, it is often used to describe the size of a voxel.

The downsampling factor is the ration or original pixels to scaled pixels.

    f = ( D0/D1, H0/H1, W0/W1 )


 

The volume will cover the same real volume, so the ratio of the scale factors will be 
the downsampling used.

    s1 = f * s0

To find the translation use the begining of the volume:

  pi = (-0.5, -0.5, -0.5) 

Then we can solve.

 0.5s0 + t0 = 0.5*s1 + t1

  t1 = t0 + 0.5*(s0 - s1)
  t1 = t0 + 0.5*(1 - f)s0

## Transform in scaled pixels. **NOT USED**

The coordnates represent the center of a pixel.
The transform will transform a coordinate to "real space".

 r = ( p + t )*s

 p, pixel location. t is the translate portion of the transform. s is the scale.

Each scale will have it's own set of pixels, transform, and scale. pi, ti, si respectivly.
s0 is the original scale factor, it is often used to describe the size of a voxel.

The downsampling factor is the ration or original pixels to scaled pixels.

    f = ( D0/D1, H0/H1, W0/W1 )

The volume will cover the same real volume, so the ratio of the scale factors will be 
the downsampling used.

    s1 = f * s0
        
To find the translation use the begining of the volume:

  pi = (-0.5, -0.5, -0.5) 

Then we can solve.

  s0*( p0 + t0) = s1*(p1 + t1)
  
  0.5s0 + s0t0 = 0.5s1 + t1s1
  t1s1 = 0.5*(s0 - s1) + s0t0
  t1 = 0.5*(s0/s1 - 1) + t0 s0/s1
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

for key in scale_factors[0]:
    t1 = ot[key] + 0.5*(1 - scale_factors[0][key])*os[key]
    print(t1, nt[key], ot[key])
    s1 = os[key]*scale_factors[0][key]
    print(key, nt[key] == t1, ns[key] == s1)
