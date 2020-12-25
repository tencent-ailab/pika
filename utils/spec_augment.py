import torch
import numpy as np
from torch.distributions.uniform import Uniform
class SpecAugment(object):
    def __init__(self, max_freq_span, max_time_span, batch_first=True):
        self.freq_span_sampler = Uniform(0.0, float(max_freq_span+1))
        self.time_span_sampler = Uniform(0.0, float(max_time_span+1)) 
        self.batch_first = batch_first

    def apply(self, inp):
        #batch, frame, freq if batch_first = True
        #TODO: frame, batch, freq otherwise
        freq_span = int(self.freq_span_sampler.sample().item()) 
        time_span  = int(self.time_span_sampler.sample().item())
        if freq_span > 0:
            freq_start = np.random.randint(0, inp.size()[-1] - freq_span) 
            inp[:, :, freq_start: freq_start + freq_span] = 0.0
        if time_span > 0:
            time_start = np.random.randint(0, inp.size()[1] - time_span) 
            inp[:, time_start: time_start + time_span, :] = 0.0
