from fastai.vision.augment import RandTransform
from tsai.all import *

class TSNormalize(RandTransform):
    # normalize by dividing each sample by its max value
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        output = o.clone()
        for i in range(len(o)):
            output[i] = output[i]/output[i].max()
        return output
    
class Resample(RandTransform):
    # resampling (probably want downsampling)
    def __init__(self, size=None, scale_factor=None,**kwargs):
        self.size = size
        self.scale_factor = scale_factor
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        output = F.interpolate(o,self.size,self.scale_factor)
        return output