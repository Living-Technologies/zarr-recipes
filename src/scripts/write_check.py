#!/usr/bin/env python3
import ngff_zarr, numpy

def createImage():
    arr = numpy.zeros( (3, 2, 32, 64, 64), dtype="int8")
    dims = ["t", "c", "z", "y", "x"]
    scale = { "t":60, "c":1, "z":2, "y":0.35, "x":0.35 }
    translate = { "t":0, "c":0, "z":-10, "y":-20, "x":-30 }
    img = ngff_zarr.ngff_image.NgffImage( arr, dims, scale=scale, translation=translate)
    return img
def writeImage(location, image2):
    multiscales = ngff_zarr.to_multiscales(image2)
    ngff_zarr.to_ngff_zarr(location, multiscales)

if __name__=="__main__":
    image = createImage()
    writeImage("testing.zarr", image)
    multiscales = ngff_zarr.from_ngff_zarr("testing.zarr")
    print(multiscales)