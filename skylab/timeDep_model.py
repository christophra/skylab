import ps_model
PowerLawLLH = ps_model.PowerLawLLH
import numpy as np


class TimePDF(object):
    r""" time pdf, top level class, for given time returns value
    can be either binned pdf or a function (Gaussian)
    """
    def __init__(self,tstart,tend):
        self.tstart=tstart
        self.tend=tend
        
    def tPDFval(ev_time):
        return self.__raise__()

class TimePDFBinned(TimePDF):
    r""" the time pdf for the case it is binned, i.e. Fermi lightcurve or a Box.
    takes as imput time_bin_series , which is list of [binstart,value]
    everything before the first bin will be evaluated as zero, the last point should be defining the end of the time PDF 
    and should be of the form [binstart,-1]
    """
    
    def __init__(self,tstart,tend,time_bin_series):
        super(TimePDFBinned, self).__init__(tstart,tend)
        
        self.tPDF=np.recarray((len(time_bin_series)+1,), dtype=[('t', float), ('v', float)])
        self.tPDF.t=np.asarray((tstart,)+zip(*time_bin_series)[0])
        self.tPDF.v=np.asarray((0.,)+zip(*time_bin_series)[1][:-1]+(0.,))
        norm=0.
        for i in range(len(self.tPDF)-1):
            norm+=self.tPDF.v[i]*(self.tPDF.t[i+1]-self.tPDF.t[i])
        self.tPDF.v=self.tPDF.v/norm
        
    def tPDFval(self,ev_time):
        return self.tPDF.v[max(0,np.searchsorted(self.tPDF.t,ev_time)-1)]
        

class TimeBoxLLH(PowerLawLLH):
    r""" Simplest time dep. search, a time PDF shaped as a box"""
    
    def __init__(self, twodim_bins=ps_model._2dim_bins, twodim_range=None, **kwargs):
        super(TimeBoxLLH, self).__init__(["logE", "sinDec"], twodim_bins, range=twodim_range, **kwargs)
