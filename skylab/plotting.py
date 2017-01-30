import numpy as np
import matplotlib.pyplot as plt
def drawSpacBkgHisto(lc_llh):
    r"""Draw a LightcurveLLH model's self.hist as an interpolated colour plot, return figure."""
    import matplotlib.pyplot as plt
    fig=plt.figure()
    H = lc_llh.hist
    H = np.flipud(H)
    plt.pcolormesh(lc_llh.cosZenith_bins, lc_llh.Azimuth_bins, H, cmap='viridis')
    cbar = plt.colorbar()
    ax = fig.gca()
    ax.set_xlabel(r'sin($\delta$)')
    ax.set_ylabel('azimuth')
    return fig

def drawSpacBkgPDF(lc_llh):
    r"""Draw a LightcurveLLH model's self.background_vect evaluated on a 2d mesh, return figure."""
    import matplotlib.pyplot as plt
    fig=plt.figure()
    XB = np.linspace(-1.,1.,100)
    YB = np.linspace(0.,2.*np.pi,100)
    X,Y = np.meshgrid(XB,YB)
    Z = lc_llh.background_vect(Y,X,X)
    Z = np.flipud(Z)
    plt.pcolormesh(X,Y,Z,cmap = 'viridis')
    cbar = plt.colorbar()
    ax = fig.gca()
    ax.set_xlabel(r'sin($\delta$)')
    ax.set_ylabel('azimuth')
    return fig