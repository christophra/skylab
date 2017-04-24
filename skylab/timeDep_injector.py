# -*-coding:utf8-*-

from __future__ import print_function

# python packages
import logging
from copy import deepcopy

# scipy-project imports
import numpy as np
from numpy.lib.recfunctions import drop_fields
import scipy.interpolate


# local package imports
from . import set_pars
from .utils import rotate, times, rotate_2d
from .ps_injector import PointSourceInjector
from .timeDep_model import TimePDFBinned



class FlareInjector(PointSourceInjector):
    r"""Injector that injects events proportional to a lightcurve.
    """
    
    
    def __init__(self, gamma, blocks, fluxes, threshold, timegen, *args, **kwargs):
        r"""Constructor, setting up the weighting function.

        Parameters
        -----------
        gamma : float
            Spectral index in spectrum E^(-gamma).
        blocks : array
            Edges of lightcurve blocks in MJD.
        fluxes : array
            Lightcurve flux in units 1 / cm^2 s.
        threshold : float
            Lightcurve threshold applied to make PDF in units 1 / cm^2 s.
        timegen : utils.times, dict of utils.times
            Time scrambler based on list of times loaded from disk.

        Other Parameters
        -----------------
        args, kwargs
            Passed to PointSourceInjector

        """

        
        # Store spectral index
        self.gamma = gamma

        # Store the lightcurve as a model
        self.lightcurve_pdf = TimePDFBinned(blocks[0],blocks[-1],
                                                          zip(blocks, np.append(fluxes,0)),
                                                          threshold=threshold)
                                                          
        # FlareInjector modifies the sampling probabilities of its time
        # generators, but they are mutable - so we need to deepcopy so that we
        # don't change this timegen in other places where it is used, such as
        #  other FlareInjectors.
        timegen = deepcopy(timegen)
        if not isinstance(timegen, dict):
            timegen = {-1: timegen}

        # set the lightcurve as sampling probability (and time range) for times
        for key, gen_i in timegen.iteritems():
            gen_i.tstart = self.lightcurve_pdf.TBS[0][0]
            gen_i.tend = self.lightcurve_pdf.TBS[-1][0]
            gen_i.pdf = self.lightcurve_pdf.tPDFvals
            timegen[key] = gen_i
        self.timegen = timegen
                                                          
        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return
        
    def __str__(self):
        r"""String representation showing some basic parameters of the injector.

        """
        # Most are taken from PointSourceInjector.__str__
        sout = super(FlareInjector,self).__str__()
        # Add additional ones before last line of dashes
        sout_split = sout.split('\n')
        sout_split.insert(-1,'\tLightcurve threshold : {0:2.2e} deg\n'.format(self.threshold))
        # Join into one string again
        sout = '\n'.join(sout_split)
        
        return sout
        
        
    def _weights(self):
        r"""Setup weights for given models.

        """
        # weights given in days, weighted to the point source flux
        # self.mc_arr["ow"] is multiplied by lifetime in fill(...)
        # include (time pdf x detector up)>0 into livetime (!)
        # (current solution: only detector uptime)
        self.mc_arr["ow"] *= self.mc_arr["trueE"]**(-self.gamma) / self._omega
        

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)

        # normalized weights for probability
        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        # double-check if no weight is dominating the sample
        if self._norm_w.max() > 0.1:
            logger.warn("Warning: Maximal weight exceeds 10%: {0:7.2%}".format(
                            self._norm_w.max()))

        return

    def fill(self, src_dec, mc, livetime):
        r"""Fill the Injector with MonteCarlo events selecting events around
        the source position(s), and store the (dictionary of) time generator(s).

        Parameters
        -----------
        src_dec : float, array-like
            Source location(s)
        mc : recarray, dict of recarrays with sample enum as key (MultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample
             
        """
        # most tasks are same as with the default PointSourceInjector
        super(FlareInjector, self).fill(src_dec, mc, livetime) # this breaks with IPython autoreload
        #self.fill(src_dec, mc, livetime)
        
        # Check whether we can generate times for all the mc samples that are given
        # If there is only one of each, they are both {-1 : value} and this cannot be verified!
        if not set(self.mc.keys()) <= set(self.timegen.keys()):
            raise ValueError("There are mc samples without matching timegen. "
                            "mc keys: %s, "
                            "timegen keys: %s."%(str(self.mc.keys()),str(self.timegen.keys())))
 
        
    def sample(self,src_ra, mean_mu, poisson=True):
        r""" Generator to get sampled events for a flaring point source.

        Parameters
        -----------
        mean_mu : float
            Mean number of events to sample

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_mu*

        """
        # generate event numbers using poissonian events
        while True:
            # The basic rotation should be the same as for any other point source
            num, sam_ev = super(FlareInjector, self).sample(src_ra, mean_mu, poisson).next() # this breaks with IPython autoreload
            
            if num<1:
                yield num, sam_ev
                continue
            # only one sample, add times for that one and return recarray
            if not isinstance(sam_ev, dict):
                # draw times
                enums = self.timegen.keys() # is this safe (?) no it's not (!)
                sam_times = self.timegen[enums[0]].sample(num).next()
                # store time in sample
                sam_ev["time"] = sam_times
                yield num, sam_ev
                continue
            
            # else: we have several samples
            for enum in enums:
                sam_ev_i = sam_ev[enum]
                sam_times_i = self.timegen[enum].sample(num)
                # store times in sam_ev_i, they already have the field
                sam_ev_i["time"] = sam_times_i
                sam_ev_i["Azimuth"] = rotate_2d(sam_ev_i["ra"], sam_times_i) # THIS NEEDS TO CHANGE TO PRESERVE CORRELATION (!)
                sam_ev[enum] = sam_ev_i
            yield num, sam_ev