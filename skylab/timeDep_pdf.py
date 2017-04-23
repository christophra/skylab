import numpy as np
from scipy.stats import norm

class TimePDF(object):
    r""" time pdf, top level class, for given time returns value
    can be either binned pdf or a function (Gaussian)
    """
    def __init__(self,tstart,tend):
        self.tstart=tstart
        self.tend=tend
        self.changed=True # does this have a point (?) is the name appropriate (?)
        
    def tPDFvals(self,ev_times): # what is the point of another name for  the method (?)
        return self.tPDFval_vect(self,ev_times)

class TimePDFBinned(TimePDF):
    r""" the time pdf for the case it is binned, i.e. Fermi lightcurve or a Box.
    takes as imput time_bin_series , which is list of [binstart,value]
    everything before the first bin will be evaluated as zero, the last point should be defining the end of the time PDF 
    and should be of the form [binstart,-1]
    """
    
    def __init__(self,tstart,tend,time_bin_series,threshold=0):
        """Constructor which creates a TimePDFBinned within
            [tstart,tend] : float, float
        by applying
            threshold : float
                the cutoff which the lightcurve has to exceed
        to
            time_bin_series : float, ndarray-like, (N+1) x 2
                [ [lower edge_1, level_1] , ... , [upper edge_N, irrelevant value]]
                
        """
        super(TimePDFBinned, self).__init__(tstart,tend)
        self.TBS=time_bin_series
        self.threshold = threshold
        
    def __str__(self):
        r"""String representation of TimePDFBinned.

        """
        out_str = "{0:s}\n".format(self.__repr__())
        out_str += 67*"~"+"\n"
        out_str += "start time  : {0:d}\n".format(int(self.tstart))
        out_str += "end time  : {0:d}\n".format(int(self.tend))
        out_str += "threshold  : {0:4e}\n".format(self.threshold)
        out_str += 67*"~"+"\n"

        return out_str
    
    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, val):
        self._threshold = val
        self.TBS_to_PDF(val)
    
    def TBS_to_PDF(self,threshold=0):
        r"""Turns list of tuples self.TBS to recarray self.tPDF
        by filling bins and values into the latter as
                    tPDF.t = [tstart,bins]
                    tPDF.v = [0,fluxes,0]
        then selecting tPDF.v above threshold and normalizing."""
        self.tPDF=np.recarray((len(self.TBS)+1,), dtype=[('t', float), ('v', float), ('d', float)])
        self.tPDF.t=np.asarray((self.tstart,)+zip(*self.TBS)[0])
        self.tPDF.v=np.asarray((0.,)+zip(*self.TBS)[1][:-1]+(0.,))
        self.tPDF.d=np.hstack((1.,self.tPDF.t[2:]-self.tPDF.t[1:-1],1.)) # outer values only used when v[0]=v[N]=0
        # self.tPDF.v[self.tPDF.v < self.threshold] = 0. # BlockTimePdf behaviour
        # is same behaviour as BlockTimePdf.
        # BUT used in current Track: BlockTimePdf1
        # which DOES subtract!
        # ergo:
        self.tPDF.v = self.tPDF.v - self.threshold # BlockTimePdf1 behaviour
        self.tPDF.v[self.tPDF.v < 0.] = 0.
        norm=(self.tPDF.v * self.tPDF.d)[:len(self.tPDF)-1].sum()
        #for i in range(len(self.tPDF)-1):
        #    norm+=self.tPDF.v[i]*self.tPDF.d[i]
        self.tPDF.v=self.tPDF.v/norm
        
   
    # try replacing with smoothed tPDF
    @staticmethod
    #@np.vectorize # Vectorizing multiplies overhead (!)
    def tPDFval_vect(self,ev_time):
        r"""Method which evaluates the PDF at ev_time, vectorized."""
        #return self.tPDF.v[max(0,np.searchsorted(self.tPDF.t,ev_time)-1)] # max doesn't handle NumPy arrays (!)
        return self.tPDF.v[np.maximum(0,np.searchsorted(self.tPDF.t,ev_time)-1)]


    @staticmethod
    def tPDFval_smooth(self,ev_time):
        r"""Method which evaluates the PDF at ev_time, vectorized, smoothed with parabola window.
        This however entirely misses the point of getting a smoother behaviour w.r.t. varying thres.
        For that, one would need to include this shape into the LC normalisation instead of replacing each
        block by the same window function that does not change shape."""
        index_i = np.maximum(0,np.searchsorted(self.tPDF.t,ev_time)-1)
        v_i = self.tPDF.v[index_i] # block pdf value
        res_i = (ev_time - self.tPDF.t[index_i])/self.tPDF.d[index_i] # time residual in window
        return parabola_window(res_i)*v_i
        
        
class TimePDFGauss(TimePDF):
    r""" the time pdf for the case of a gaussuan shaped flare """
    
    def __init__(self,tstart,tend):
        super(TimePDFGauss, self).__init__(tstart,tend)        
        self.mean=(tend-tstart)/2. # what is the point of this definition (?)
        self.sigma=self.mean/100. # dito (?)
        
    def setMeanAndSigma(self,mean,sigma):
        self.mean=mean
        self.sigma=sigma
        
    def tPDFvals(self,ev_times):
        gfcn=norm(loc = self.mean, scale = self.sigma)
        return gfcn.pdf(ev_times)