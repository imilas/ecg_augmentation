from fastai.vision.augment import RandTransform
from tsai.all import *
from scipy.interpolate import CubicSpline
import torchaudio

class Scale(Transform):
    # resampling (probably want downsampling)
    def __init__(self, size=None, scale_factor=None,mode="nearest",**kwargs):
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        output = F.interpolate(o,self.size,self.scale_factor,mode = self.mode)
        
        return output
# normlization methods
class Normalize(Transform):
    # normalize by dividing each ecg by its max value (each lead divided by max value of entire ecg)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        output = o.clone()
        
        for i in range(len(output)):
            output[i] = output[i]/output[i].max() # this should probably be absolute value
        return output
    
class NormMinMax(Transform):
    # 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encodes(self, output: TSTensor):
        for i in range(len(output)):
            output[i] = (output[i]-output[i].min())/(output[i].max()-output[i].min())
        return output

class NormMaxDiv(Transform):
    # normalize by dividing each ecg by its max value (each lead divided by max value of entire ecg)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encodes(self, output: TSTensor):
        for i in range(len(output)):
            output[i] = output[i]/output[i].max() # this should probably be absolute value
        return output

class NormZScore(Transform):
    #
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encodes(self, output: TSTensor):
        for i in range(len(output)):
            output[i] = (output[i]-output[i].mean())/(output[i].std()) 
        return output

class NormMedian(Transform):
    # normalize by y = (x-median)/median(abs(x-median))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encodes(self, output: TSTensor):
        for i in range(len(output)):
            output[i] = (output[i]-output[i].median()) / torch.abs(output[i]-output[i].median()).median()
        return output
    
class NormDecimalScaling(Transform):
    # normalize by y = x/(10**log(max))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encodes(self, output: TSTensor):
        for i in range(len(output)):
            output[i] = (output[i])/10**torch.floor(torch.log10(output[i].max()))
        return output
    
class MulNoise(RandTransform):
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
    
class RandomShift(RandTransform):
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
    
class WindowWarping(RandTransform):
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
    
class WindowSlicing(RandTransform):
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

class CutOutWhenTraining(RandTransform):
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

class BandPass(Transform):
    # resampling (probably want downsampling)
    def __init__(self, sample_rate =500, low_cut=45, high_cut=3,leads=12, **kwargs):
        self.sr = sample_rate
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.leads = leads
        super().__init__(**kwargs)
    def encodes(self, o: TSTensor):
        signal_len = o.shape[-1]
        o = o.reshape([-1,signal_len])
        o = torchaudio.functional.highpass_biquad(o,sample_rate=self.sr,cutoff_freq = self.high_cut)
#         o = torchaudio.functional.lowpass_biquad(o,sample_rate=self.sr,cutoff_freq = self.low_cut)
        o = o.reshape([-1,self.leads,signal_len])
        return o