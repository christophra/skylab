import ps_model
PowerLawLLH = ps_model.PowerLawLLH
import numpy as np
from scipy.stats import norm

class TimePDF(object):
    r""" time pdf, top level class, for given time returns value
    can be either binned pdf or a function (Gaussian)
    """
    def __init__(self,tstart,tend):
        self.tstart=tstart
        self.tend=tend
        
    def tPDFvals(self,ev_times):
        return self.tPDFval_vect(self,ev_times)

class TimePDFBinned(TimePDF):
    r""" the time pdf for the case it is binned, i.e. Fermi lightcurve or a Box.
    takes as imput time_bin_series , which is list of [binstart,value]
    everything before the first bin will be evaluated as zero, the last point should be defining the end of the time PDF 
    and should be of the form [binstart,-1]
    """
    
    def __init__(self,tstart,tend,time_bin_series,threshold=0):
        super(TimePDFBinned, self).__init__(tstart,tend)
        self.TBS=time_bin_series
        self.TBS_to_PDF(threshold)
    
    def TBS_to_PDF(self,threshold=0):
        self.threshold=threshold
        self.tPDF=np.recarray((len(self.TBS)+1,), dtype=[('t', float), ('v', float)])
        self.tPDF.t=np.asarray((self.tstart,)+zip(*self.TBS)[0])
        self.tPDF.v=np.asarray((0.,)+zip(*self.TBS)[1][:-1]+(0.,))
        self.tPDF.v[self.tPDF.v < self.threshold] = 0.
        norm=0.
        for i in range(len(self.tPDF)-1):
            norm+=self.tPDF.v[i]*(self.tPDF.t[i+1]-self.tPDF.t[i])
        self.tPDF.v=self.tPDF.v/norm
        
    @staticmethod
    @np.vectorize
    def tPDFval_vect(self,ev_time):
        return self.tPDF.v[max(0,np.searchsorted(self.tPDF.t,ev_time)-1)]
        
class TimePDFGauss(TimePDF):
    r""" the time pdf for the case of a gaussuan shaped flare """
    
    def __init__(self,tstart,tend):
        super(TimePDFGauss, self).__init__(tstart,tend)        
        self.mean=(tend-tstart)/2.
        self.sigma=self.mean/100.
        
    def setMeanAndSigma(self,mean,sigma):
        self.mean=mean
        self.sigma=sigma
        
    def tPDFvals(self,ev_times):
        gfcn=norm(loc = self.mean, scale = self.sigma)
        return gfcn.pdf(ev_times)
        
class TimeBoxLLH(PowerLawLLH):
    r""" Simplest time dep. search, a time PDF shaped as a box"""
    
    def __init__(self, twodim_bins=ps_model._2dim_bins, twodim_range=None, **kwargs):
        super(TimeBoxLLH, self).__init__(["logE", "sinDec"], twodim_bins, range=twodim_range, **kwargs)
