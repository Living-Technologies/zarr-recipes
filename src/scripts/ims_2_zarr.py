import sys, pathlib
import re
from imaris_ims_file_reader.ims import ims
import numpy

import dask

from dask import array as da
import ngff_zarr

def getKey( pth ):
    name = pth.name
    pt = re.compile(".*_F(\\d+)\\.ims")
    k = int( pt.findall(name)[0] )
    return k

def sortPaths( all_files ):
    out = [ ( getKey(f), f) for f in all_files ]
    out.sort()
    return [i[1] for i in out]

class Loader:
    def __init__(self, paths):
        self.paths = paths
        self.n = len(paths)
        fp = ims(paths[0])
        self.dims = ["t", "c", "z", "y", "x"]
        self._prepareMetaData( fp )
        fp.close()

    def loadChunk(self, block_id):
        fp = ims(self.paths[block_id[0]])
        ret = numpy.expand_dims( numpy.array(fp[0]), 0 )
        fp.close()
        return ret

    def _prepareMetaData(self, fp):
        self.shape = fp[0].shape
        scales = (1, 1, *fp.resolution)
        self.scale = { d:v for d, v in zip(self.dims, scales) }
        self.translation = { d : 0 for d in self.dims}
        self.dtype = fp.dtype



if __name__ == "__main__":
    #dask.config.set(scheduler='synchronous')
    dask.config.set(num_workers=4)
    base = pathlib.Path(sys.argv[1])
    all_files = base.glob("*.ims")
    sfs = sortPaths(all_files)
    loader = Loader(sfs)

    out = da.map_blocks( loader.loadChunk, dtype=loader.dtype, chunks = ((1, )*loader.n, *loader.shape ) )
    print(out.shape)
    oi = ngff_zarr.to_ngff_image(out, dims=loader.dims, translation=loader.translation, scale=loader.scale)
    ms = ngff_zarr.to_multiscales( oi, cache=False, chunks=(1, 1, *loader.shape[1:]))
    print("writing")
    ngff_zarr.to_ngff_zarr( sys.argv[2], ms, overwrite=False )
