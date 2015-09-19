# -*-coding:utf8-*-

from data import init

from scipy.stats import chi2
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from skylab.ps_injector import PointSourceInjector

if __name__=="__main__":

    # init likelihood class
    llh = init(10000, 1000000, ncpu=4)

    print(llh)

    # init a injector class sampling events at a point source
    inj = PointSourceInjector(2.)

    # start calculation for dec = 0
    result = llh.weighted_sensitivity(0., [0.5, 2.87e-7], [0.9, 0.5],
                                      inj,
                                      n_bckg=1000,
                                      n_iter=1000,
                                      eps=5e-2)
