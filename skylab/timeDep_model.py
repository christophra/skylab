import ps_model
WeightLLH = ps_model.WeightLLH
import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from copy import deepcopy

class TimePDF(object):
    r""" time pdf, top level class, for given time returns value
    can be either binned pdf or a function (Gaussian)
    """
    def __init__(self,tstart,tend):
        self.tstart=tstart
        self.tend=tend
        self.changed=True
        
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
        
class TimeBoxLLH(WeightLLH):
    r""" Simplest time dep. search, a time PDF shaped as a box"""
    
    def __init__(self, twodim_bins=ps_model._2dim_bins,timePDF=None, twodim_range=None, **kwargs):
        
        self.timePDF=timePDF
        params = dict(gamma=(kwargs.pop("seed", ps_model._gamma_params["gamma"][0]), deepcopy(kwargs.pop("bounds", deepcopy(ps_model._gamma_params["gamma"][1])))))

        super(TimeBoxLLH, self).__init__(params, ["logE", "sinDec"], twodim_bins, range=twodim_range, normed=1, **kwargs)

        r"""this part is similar to the time integrated, but the important difference is that it is in local coodinates
        so that on short time scales the detector special directions are evaluated correctly
        """
        self.Azimuth_bins_n=90
        self.cosZenith_bins_n=12
        
    def __call__(self, exp, mc,livetime):
        super(TimeBoxLLH, self).__call__(exp, mc,livetime)
        self.hist, binsa, binsz=np.histogram2d(exp["Azimuth"],exp["cozZenith"], bins=[self.Azimuth_bins_n,self.cosZenith_bins_n], range=None, normed=False )
        for zen in range(self.cosZenith_bins_n):
            sumZband=sum(self.hist[:,zen])
            for az in range(self.Azimuth_bins_n):
                self.hist[az][zen]=self.hist[az][zen]*self.Azimuth_bins_n/sumZband

        # overwrite range and bins to actual bin edges
        self.Azimuth_bins = binsa
        self.Azimuth_range = (binsa[0], binsa[-1])
        self.cosZenith_bins= binsz
        self.cosZenith_range = (binsz[0], binsz[-1])

        if np.any(self.hist <= 0.):
            estr = ("Local coord. hist bins empty, this must not happen! "
                    +"Empty bins: {0}".format(bmids[self.hist <= 0.]))
            raise ValueError(estr)

        self._bckg_spline2D = RectBivariateSpline( (binsa[1:] + binsa[:-1]) / 2.,(binsz[1:] + binsz[:-1]) / 2.,  np.log(self.hist))
        
    def drawSpacBkgHisto(self):
        import matplotlib.pyplot as plt
        fig=plt.figure()
        H = np.rot90(self.hist)
        H = np.flipud(H)
        plt.imshow(H, extent=[self.Azimuth_bins[0],self.Azimuth_bins[-1],self.cosZenith_bins[0],self.cosZenith_bins[-1]], interpolation='nearest',aspect="auto")
        cbar = plt.colorbar()
        return fig
    
    def drawSpacBkgPDF(self):
        import matplotlib.pyplot as plt
        fig=plt.figure()
        XB = np.linspace(0.,2.*np.pi,100)
        YB = np.linspace(-1.,1.,100)
        X,Y = np.meshgrid(XB,YB)
        Z = self.background_vect(self,X,Y,None)
        Z = np.flipud(Z)
        plt.imshow(Z,interpolation='none')
        cbar = plt.colorbar()
        return fig
        
    
    def background(self, ev):
        flatTimePDF=1./(self.timePDF.tend*self.timePDF.tstart)
        return self.background_vect(self, ev["Azimuth"],ev["cozZenith"],ev["sinDec"])*flatTimePDF
        
    @staticmethod
    @np.vectorize
    def background_vect(self, az,zen,sinDec=None):
        r"""Spatial background distribution. local coordinates for time dep.

        For IceCube is only declination dependent, in a more general scenario,
        it is dependent on zenith and
        azimuth, e.g. in ANTARES, KM3NET, or using time dependent information.

        Parameters
        -----------
        ev : structured array
            Event array, importand information *sinDec* for this calculation

        Returns
        --------
        P : array-like

        """
        azbin =int((az-self.Azimuth_range[0])/(self.Azimuth_range[1]-self.Azimuth_range[0])*self.Azimuth_bins_n)
        cosbin=int((zen-self.cosZenith_range[0])/(self.cosZenith_range[1]-self.cosZenith_range[0])*self.cosZenith_bins_n)
        if azbin==self.Azimuth_bins_n:
            azbin=self.Azimuth_bins_n-1
        if cosbin==self.cosZenith_bins_n:
            cosbin=self.cosZenith_bins_n-1

        if sinDec:
            return self.hist[azbin][cosbin]*self.bckg_spline(sinDec) 
        else:
            return self.hist[azbin][cosbin]

        #return np.exp(self.bckg_spline(az,zen))
        
    def _effA(self, mc, livetime, **pars):
        r"""Calculate two dimensional spline of effective Area versus
        declination and spectral index for Monte Carlo.

        """

        gamma_vals = pars["gamma"]

        x = np.sin(mc["trueDec"])
        hist = np.vstack([np.histogram(x,
                                       weights=self._get_weights(mc, gamma=gm)
                                                * livetime * 86400.,
                                       bins=self.sinDec_bins)[0]
                          for gm in gamma_vals]).T

        self._spl_effA = RectBivariateSpline(
                (self.sinDec_bins[1:] + self.sinDec_bins[:-1]), gamma_vals,
                np.log(hist), kx=2, ky=2, s=0)

        return
    
    def effA(self, dec, **params):
        r"""Evaluate effective Area at declination and spectral index.

        Parameters
        -----------
        dec : float
            Declination.

        gamma : float
            Spectral index.

        Returns
        --------
        effA : float
            Effective area at given point(s).
        grad_effA : float
            Gradient at given point(s).

        """

        if (np.sin(dec) < self.sinDec_bins[0]
                or np.sin(dec) > self.sinDec_bins[-1]):
            return 0., None

        gamma = params["gamma"]

        val = np.exp(self._spl_effA(np.sin(dec), gamma, grid=False, dy=0.))
        grad = val * self._spl_effA(np.sin(dec), gamma, grid=False, dy=1.)

        return val, dict(gamma=grad)
        
    
    
    def _get_weights(self, mc, **params):
        return mc["ow"] * mc["trueE"]**(-params["gamma"])
    
    
    def weight(self, ev, **params):
        print params
        val, grad_w=super(TimeBoxLLH, self).weight(ev, **params)
        if self.timePDF.changed:
            self.time_w=self.timePDF.tPDFvals(ev["mjd"])
        
        val=val*self.time_w
        
        return val, grad_w
    
    
    