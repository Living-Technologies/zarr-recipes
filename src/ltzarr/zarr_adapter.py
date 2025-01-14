import zarr
import ngff_zarr
class MultiScale:
    """
        A single multiscale image pyramid (or just image.)
    """
    def __init__(self, metadata, arrays):
        self.axes = []
        self.data = []
        self.transforms = []
        self._setMetadata( metadata )
        self._setData( arrays )
        self._validate()
    def _setMetadata(self, metadata):
        
        self.axes = metadata["axes"]
        ds = metadata["datasets"]
        for dsmd in ds:
            transforms0 = dsmd["coordinateTransformations"]
            self.transforms.append(transforms0)
    def _setData( self, arrays):
        self.data.append(arrays)
    
    def _validate(self):
        t_ps = len(self.transforms)
        d_ps = len(self.data)
        if t_ps != d_ps:
            raise Exception(
                "transformations do not match data %s, %s"%( len(self.transformations), len(self.data) )
                )
        ax_l = len(self.axes)
        for arr in self.data:
            shp = arr.shape
            dims = len(shp)
            if ax_l != dims:
                an = "".join( ax["name"] for ax in self.axes )
                raise Exception(
                    "number of axes do not correspond to number of dimensions. %s, %s"%( an, shp) 
                )
    def summary(self):
        for axis in self.axes:
            unit = "NA"
            if "unit" in axis:
                unit = axis["unit"]
            print("    > %s, %s, %s"%(axis["name"], axis["type"], unit) )
        for tfs, arr in zip(self.transforms, self.data):
            print("  data: ", arr.shape)
            print("  transforms: ")
            for tf in tfs:
                tp = tf["type"]
                print( "    ", tp )
                if tp in tf:
                    print( "      *", tf[ tp ] ) 
    def getVolume(self, scale, time):
        """
            
        """
        pass
    def getTransformations( self, scale ):
        """
        
        """
        pass
def loadZarr(location):
    """
      Loads the location zarr file.
        
        Params:
          location - Path to zarr folder
        
        Returns:
          metadata, array_data
    """
    start = zarr.open_group(
       store=location,
       mode='r',
    )
    multiscales = start.attrs["multiscales"]
    
    arrays = [ start[ key ] for key in start.array_keys() ]
    out = []
    for md, array in zip(multiscales, arrays):
        out.append( MultiScale( md, array) )
        
    return out
    
