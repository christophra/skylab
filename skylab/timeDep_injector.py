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
from .ps_injector import PointSourceInjector, StackingPointSourceInjector
from .ps_injector import logger, _deg, _ext
from .timeDep_pdf import TimePDFBinned



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
        
        # After this, self.mc is filled and we can check whether
        # we can generate times for all the mc samples that are given.
        # If there is only one of each, they are both {-1 : value}
        # and this cannot be verified!
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

            
class FlareStackingInjector(StackingPointSourceInjector): # (or base it on FlareInjector?)
    r"""Injector that injects events for several sources proportional to their lightcurves.
    """
    
    
    def __init__(self, gamma, blocks, fluxes, threshold, timegen, *args, **kwargs):
        r"""Constructor, setting up the weighting function.

        Parameters
        -----------
        gamma : float
            Spectral index in spectrum E^(-gamma).
        blocks : list(array)
            List of edges of lightcurve blocks in MJD.
        fluxes : list(array)
            List of lightcurve flux arrays in units 1 / cm^2 s.
        threshold : float
            List of lightcurve thresholds applied to make PDF in units 1 / cm^2 s.
        timegen : utils.times, dict of utils.times
            Single time scrambler based on list of times loaded from disk.

        Other Parameters
        -----------------
        args, kwargs
            Passed to PointSourceInjector

        """
        
        # StackingPointSourceInjector:
        # runs PointSourceInjector.__init__(*args, **kwargs)
        # i.e. self.gamma = gamma and set_pars, already have that.

        if not isinstance(threshold,list):
            threshold = list(threshold)
        
        # Store spectral index
        self.gamma = gamma

        # Store the lightcurve as a model
        self.lightcurve_pdf = [TimePDFBinned(b[0],b[-1],
                                             zip(b, np.append(f,0)),
                                             threshold=t)\
                                             for b,f,t in zip(blocks,fluxes,threshold)]
                                                          
        # FlareInjector modifies the sampling probabilities of its time
        # generators, but they are mutable - so we need to deepcopy so that we
        # don't change this timegen in other places where it is used, such as
        #  other FlareInjectors.
        if not isinstance(timegen, dict):
            timegen = {-1: timegen}

        # set the lightcurve as sampling probability (and time range) for times
        for key, gen_i in timegen.iteritems():
            timegen_list = []
            for lc in self.lightcurve_pdf:
                gen_lc = deepcopy(gen_i)
                #gen_lc.tstart = lc.TBS[0][0]
                #gen_lc.tend = lc.TBS[-1][0]
                gen_lc.pdf = lc.tPDFvals
                timegen_list.append(gen_lc)
            timegen[key] = timegen_list
        

        self.timegen = timegen
                                                          
        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return
        
    def __str__(self):
        r"""String representation showing some basic parameters of the injector.

        """
        # Most are taken from PointSourceInjector.__str__
        sout = super(FlareStackingInjector,self).__str__()
        # Add additional ones before last line of dashes
        sout_split = sout.split('\n')
        sout_split.insert(-1,''.join(['\tLightcurve threshold : {0:2.2e} deg\n'.format(th) for th in self.threshold])
        # Join into one string again
        sout = '\n'.join(sout_split)
        return sout
        
    def _weights(self, ow, omega):
        r"""Setup weights for given models.

        """
        # weights given in days, weighted to the point source flux
        # self.mc_arr["ow"] is multiplied by lifetime in fill(...)
        # include (time pdf x detector up)>0 into livetime (!)
        # (current solution: only detector uptime)
        
        # Mods for stacking: forget about self.mc_arr (don't know why)
        # pass ow = arr['ow']*E**-gamma as arg, also omega=omega/w_theo
        # but then same result: ow /= omega
        
        # Reason for passing omega: has to include theo weights (via fill)
        # and is already in fill expanded into an array like mc_arr
        # Reason for passing ow: includes E^-g x livetime (from fill)
        # (also an array like mc_arr)
        
        # Tentatively: same as Stacking (so could inherit this)
        
        ow /= omega
        #self.mc_arr["ow"] *= self.mc_arr["trueE"]**(-self.gamma) / self._omega
        

        #self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)
        self._raw_flux = np.sum(ow, dtype=np.float)

        # normalized weights for probability
        #self._norm_w = self.mc_arr["ow"] / self._raw_flux
        self._norm_w = ow / self._raw_flux

        # double-check if no weight is dominating the sample
        if self._norm_w.max() > 0.1:
            logger.warn("Warning: Maximal weight exceeds 10%: {0:7.2%}".format(
                            self._norm_w.max()))

        return

    def fill(self, src_dec, mc, livetime):
        r"""Fill the Injector with MonteCarlo events selecting events around
        the source position(s).

        Parameters
        -----------
        src_dec : float, array-like
            Source location(s)
        mc : recarray, dict of recarrays with sample enum as key (MultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample
        Optional:
        -----------
        w_theo : float.
            Theoretical weights for the sources.
            Default: array([1. for s in sources])
        model : Use the ModelInjector class. (?)
             Default: False.
        """
        # most tasks are same as with the StackingPointSourceInjector
        # (at least the code looks totally appropriate)
        super(FlareStackingInjector, self).fill(src_dec, mc, livetime)
        
        
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
        src_ra : float, array-like
            (array of) source right ascension(s).

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
        src_ra = src_ra
        # generate event numbers using poissonian events
        while True:

            num = (self.random.poisson(mean_mu)
                        if poisson else int(np.around(mean_mu)))

            logger.debug(("Generated number of sources: {0:3d} "+
                          "of mean {1:5.1f} sources").format(num, mean_mu))

            # if no events should be sampled, return nothing
            if num < 1:
                yield num, None
                continue

            # get random choice of Monte Carlo events across sources
            # (_norm_w includes the omega and weight for each)
            # in terms of an array of (enum, src_idx, idx)
            # which are indices for the sample, the source, and the event in self.mc[key]
            # (the latter, idx, can appear multiple times for different sources)
            sam_idx = self.random.choice(self.mc_arr, size=num, p=self._norm_w)


            # which samples do we have events from?
            enums = np.unique(sam_idx["enum"])

            # only one sample, add times for that one and return recarray
            if len(enums) == 1 and enums[0] < 0:
                # only one sample, just return recarray
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]]) # why need copy (?)
                src_ind = sam_idx['src_idx']
                sam_ev_rot = rotate_struct(sam_ev, src_ra[src_ind], self.src_dec[src_ind]])
                # insert times here
                # for each source that is involved in sam_ev:
                for i_src in range(src_ind.max()+1):
                    # where the events belonging to that source are
                    src_mask = (sam_idx['src_idx'])==i_src
                    # number of injected events for that source
                    src_num = np.count_nonzero(src_mask)
                    sam_times = self.timegen[-1][i_src].sample(src_num).next()
                    sam_ev_rot['time'][src_mask] = sam_times
                # Does Azimuth need to change to preserve correlation (?)
                #sam_ev_rot["Azimuth"] = rotate_2d(sam_ev_rot["ra"], sam_ev_rot['time'])
                yield num, sam_ev_rot 
                continue

            # else: we have several samples
            sam_ev = dict()
            for enum in enums:
                idx = sam_idx[sam_idx["enum"] == enum]["idx"]
                sam_ev_i = np.copy(self.mc[enum][idx])
                src_ind = sam_idx[sam_idx["enum"] == enum]["src_idx"]
                sam_ev_rot_i = rotate_struct(sam_ev_i, src_ra[src_ind], self.src_dec[src_ind])
                # insert times here
                # for each source that is involved in sam_ev_i:
                for i_src in range(src_ind.max()+1):
                    # where the events belonging to that source are
                    src_mask = (sam_idx['src_idx'])==i_src
                    # number of injected events for that source & sample
                    src_num = np.count_nonzero(src_mask)
                    sam_times_i = self.timegen[enum][i_src].sample(src_num).next()
                    sam_ev_rot_i['time'][src_mask] = sam_times_i
                # Does Azimuth need to change to preserve correlation (?)
                #sam_ev_rot_i["Azimuth"] = rotate_2d(sam_ev_rot_i["ra"], sam_ev_rot_i['time'])
                sam_ev[enum] = sam_ev_rot_i
            yield num, sam_ev
