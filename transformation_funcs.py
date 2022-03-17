from fastai.vision.augment import RandTransform
from tsai.all import *
from scipy.interpolate import CubicSpline
    
class Scale(Transform):
    # resampling (probably want downsampling)
    def __init__(self, size=None, scale_factor=None,**kwargs):
        self.size = size
        self.scale_factor = scale_factor
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        output = F.interpolate(o,self.size,self.scale_factor)
        
        return output
    
class Normalize(Transform):
    # normalize by dividing each sample by its max value
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        output = o.clone()
        for i in range(len(o)):
            output[i] = output[i]/output[i].max()
        return output
    
class MulNoise(Transform):
    "Applies multiplicative noise on the y-axis for each step of a `TSTensor` batch"
    order = 90
    def __init__(self, magnitude=1, ex=None, **kwargs):
        self.magnitude, self.ex = magnitude, ex
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        if not self.magnitude or self.magnitude <= 0: return o
        noise = torch.normal(1, self.magnitude * .025, o.shape, dtype=o.dtype, device=o.device)
        output = o * noise
        if self.ex is not None: output[...,self.ex,:] = o[...,self.ex,:]
        return output
    
class RandomShift(Transform):
    "Shifts and splits a sequence"
    order = 90
    def __init__(self, magnitude=0.02, ex=None, **kwargs):
        self.magnitude, self.ex = magnitude, ex
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        if not self.magnitude or self.magnitude <= 0: return o
        pos = int(round(np.random.randint(0, o.shape[-1]) * self.magnitude)) * (random.randint(0, 1)*2-1)
        output = torch.cat((o[..., pos:], o[..., :pos]), dim=-1)
        if self.ex is not None: output[...,self.ex,:] = o[...,self.ex,:]
        return output
    
class WindowWarping(Transform):
    """Applies window slicing to the x-axis of a `TSTensor` batch based on a random linear curve based on
    https://halshs.archives-ouvertes.fr/halshs-01357973/document"""
    order = 90
    def __init__(self, magnitude=0.1, ex=None, **kwargs):
        self.magnitude, self.ex = magnitude, ex
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        if not self.magnitude or self.magnitude <= 0 or self.magnitude >= 1: return o
        f = CubicSpline(np.arange(o.shape[-1]), o.cpu(), axis=-1)
        output = o.new(f(random_cum_linear_generator(o, magnitude=self.magnitude)))
        if self.ex is not None: output[...,self.ex,:] = o[...,self.ex,:]
        return output
    
class WindowSlicing(Transform):
    "Randomly extracts and resize a ts slice based on https://halshs.archives-ouvertes.fr/halshs-01357973/document"
    order = 90
    def __init__(self, magnitude=0.1, ex=None, mode='linear', **kwargs):
        "mode:  'nearest' | 'linear' | 'area'"
        self.magnitude, self.ex, self.mode = magnitude, ex, mode
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        if not self.magnitude or self.magnitude <= 0 or self.magnitude >= 1: return o
        seq_len = o.shape[-1]
        win_len = int(round(seq_len * (1 - self.magnitude)))
        if win_len == seq_len: return o
        start = np.random.randint(0, seq_len - win_len)
        return F.interpolate(o[..., start : start + win_len], size=seq_len, mode=self.mode, align_corners=None if self.mode in ['nearest', 'area'] else False)

class CutOut(Transform):
    "Sets a random section of the sequence to zero"
    order = 90
    def __init__(self, alpha=5,beta=40, ex=None, **kwargs):
        self.alpha,self.beta, self.ex = alpha,beta,ex
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        seq_len = o.shape[-1]
        lambd = np.random.beta(self.alpha,self.beta) 
        win_len = int(round(seq_len * lambd))
        start = np.random.randint(-win_len + 1, seq_len)
        end = start + win_len
        start = max(0, start)
        end = min(end, seq_len)
        output = o.clone()
        output[..., start:end] = 0
        if self.ex is not None: output[...,self.ex,:] = o[...,self.ex,:]
        
        return output