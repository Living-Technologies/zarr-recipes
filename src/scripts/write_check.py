#!/usr/bin/env python3
import ngff_zarr, numpy
import ltzarr

def createImage():
    arr = numpy.zeros( (3, 2, 32, 64, 64), dtype="uint8")
    dims = ["t", "c", "z", "y", "x"]
    scale = { "t":60, "c":1, "z":2, "y":0.35, "x":0.35 }
    translate = { "t":0, "c":0, "z":-10, "y":-20, "x":-30 }
    img = ngff_zarr.ngff_image.NgffImage( arr, dims, scale=scale, translation=translate)
    return img

if __name__=="__main__":
    image = createImage()
    #ltzarr.saveZarr("testing.zarr", image, scale_factors = [{"z":1, "x":2, "y":2}])
    multiscales = ngff_zarr.from_ngff_zarr("testing.zarr")
    
    original = multiscales.images[0]
    scaled = multiscales.images[1]
    print(scaled)
    #print(multiscales)