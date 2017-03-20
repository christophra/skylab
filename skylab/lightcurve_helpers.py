# An embarassing "file of helper functions"
# TO DO: integrate these as class methods of TimePDFBinned
import numpy as np

def histogram_centers(edges):
    """Returns the centers from an array of bin edges."""
    return (edges[1:]+edges[:-1])/2.

def histogram_percentile(hist,q):
    """Takes a numpy-style histogram and percentile number, outputs percentile(s)
    (along the last axis if there are more than 1)."""
    if len(hist)>2:
        return np.array([histogram_percentile((h,)+hist[2:],q) for h in hist[0]])
    else:
        temp = 0
        total_entries = hist[0].sum()
        for iBin in range(hist[0].size):
            temp += hist[0][iBin]
            if temp > total_entries*(q/100.0):
                gradient = hist[0][iBin]/(hist[1][iBin+1]-hist[1][iBin])
                difference_to_upper_bin_edge = temp - total_entries*(q/100.0)
                correction = difference_to_upper_bin_edge/gradient
                return hist[1][iBin+1] - correction

def variability(flux):
    flux_sorted = np.sort(flux)
    nbins = len(flux_sorted)
    return (flux_sorted[-1] - flux_sorted[0]) / np.mean(flux_sorted[nbins/10:-nbins/10])
def central_variability(fluxwindow,sigma=.2):
    """Variability inside `fluxwindow`, weighted with gaussian centered middle of window.
    `sigma` = std.dev in units of window width."""
    flux_sorted = np.sort(fluxwindow)
    nbins = len(flux_sorted)
    index = np.arange(nbins)
    
    gaussian_weights = np.exp(-(index-(nbins/2.))**2 /2 /(sigma*nbins)**2)

    mask_lower = fluxwindow<np.mean(flux_sorted[:nbins/20])
    mask_upper = fluxwindow>np.mean(flux_sorted[-nbins/20:])
    mask_central = np.invert(mask_lower) * np.invert(mask_upper)
    
    weights_lower = (gaussian_weights*mask_lower)/(gaussian_weights*mask_lower).sum()
    weights_upper = (gaussian_weights*mask_upper)/(gaussian_weights*mask_upper).sum()
    weights_central = (gaussian_weights*mask_central)/(gaussian_weights*mask_central).sum()
    
    weighted_vari = ((fluxwindow*weights_upper).sum() - (fluxwindow*weights_lower).sum()) / (fluxwindow*weights_central).sum()
    return weighted_vari

def moving_avg(y,nhalf):
    y_avg = []
    for i_ctr in range(len(y)-2*nhalf):
        y_avg.append(np.mean(y[i_ctr:i_ctr+2*nhalf+1]))
    return np.array(y_avg)
def moving_func(y,nhalf,func):
    y_avg = []
    for i_ctr in range(len(y)-2*nhalf):
        y_avg.append(func(y[i_ctr:i_ctr+2*nhalf+1]))
    return np.array(y_avg)
def old_variability(flux):
    """Computes Asen's variability from a time series of fluxes, using 11-day average."""
    flux_avg = moving_avg(flux,5)
    return (np.max(flux_avg) - np.min(flux_avg))/np.mean(flux)
    #flux_avg = np.array(flux_avg)
    #flux_sorted = np.sort(flux_avg)
    #nbins = len(flux_sorted)
    #return (flux_sorted[-1] - flux_sorted[0]) / np.mean(flux_sorted[nbins/10:-nbins/10]) 
def jumpcurve(bins,blocks):
    dflux = blocks[1:]-blocks[:-1]
    time_ctr = (bins[1:] + bins[:-1])/2.
    dtime = time_ctr[1:] - time_ctr[:-1]
    flux_avg = (blocks[1:] + blocks[:-1])/2.
    return bins[1:-1],np.abs(dflux)/dtime

    
def auto_threshold(bins, blocks):
    # using jumpcurve on blocks
    time_jump, jumps = jumpcurve(bins,blocks)
    binwidth = bins[1:] - bins[:-1]
    binmid = histogram_centers(bins)
    binwidth_jump = (binwidth[1:] + binwidth[:-1])/2.
    jump_thres = histogram_percentile(np.histogram(jumps,weights=binwidth_jump,bins=1000),50)
    quiesc = jumps<jump_thres

    # determine quiescent flux and rms with blocks
    blocks_avg = (blocks[1:] + blocks[:-1])/2.
    blocks_quiescent = np.average(blocks_avg,weights=quiesc * binwidth_jump) # average flux under jump threshold
    blocks_quiescent_squared = np.average(blocks_avg**2,weights=quiesc * binwidth_jump)
    blocks_quiescent_rms = np.sqrt(blocks_quiescent_squared - blocks_quiescent**2)
        
    #define threshold
    threshold = blocks_quiescent + 5*blocks_quiescent_rms
    return threshold
    
def moving_window_threshold(time, flux, bins, blocks):
    # using gaussian variability window on flux
    avg_width = 50
    time_vari = moving_avg(time,avg_width)
    vari = moving_func(flux,avg_width,lambda f:central_variability(f,sigma=0.1))
    vari_thres = np.percentile(vari,50)
    
    quiesc = vari<vari_thres
    
    flux_avg = moving_avg(flux,avg_width)
    t_quiesc = time_vari[quiesc]
    flux_quiesc = np.interp(t_quiesc,histogram_centers(bins),blocks)
    avg_quiesc = np.sum(flux_quiesc / vari[quiesc])/np.sum(1./vari[quiesc])
    rms_quiesc = np.sqrt(flux_avg[quiesc].var())
    threshold = avg_quiesc + 3*rms_quiesc
    #threshold = flux_quiesc.max()
    return threshold
