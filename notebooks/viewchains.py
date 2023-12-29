import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import erf
from copy import deepcopy
saved_backend = plt.rcParams["backend"]
from scipy.interpolate import interp1d

from cobaya.yaml          import yaml_load_file
from cobaya.samplers.mcmc import plot_progress
from cobaya.model import get_model
#
from getdist.mcsamples    import MCSamplesFromCobaya
from getdist.mcsamples    import loadMCSamples
import getdist.plots      as     gdplt
matplotlib.use(saved_backend)

# pretty plots
import matplotlib
from matplotlib.pyplot import rc
import matplotlib.font_manager
rc('font',**{'size':'22','family':'serif','serif':['Times']})
rc('mathtext', **{'fontset':'cm'})
#rc('text', usetex=True)
rc('legend',**{'fontsize':'18'})
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['xtick.labelsize'] = 25 
matplotlib.rcParams['ytick.labelsize'] = 25
matplotlib.rcParams['legend.fontsize'] = 25
matplotlib.rcParams['xtick.major.size'] = 10
matplotlib.rcParams['ytick.major.size'] = 10
matplotlib.rcParams['xtick.minor.size'] = 5
matplotlib.rcParams['ytick.minor.size'] = 5
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['xtick.minor.width'] = 1.5
matplotlib.rcParams['ytick.minor.width'] = 1.5
matplotlib.rcParams['axes.titlesize'] = 30
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


# confidence intervals of truth -> sig
sig = np.linspace(-5,5,1000)
phi = np.array( [(1.0 + erf(x / np.sqrt(2.0))) / 2.0 for x in sig])
def get_confidence_sigma(chain,param,truth,eps=1e-3):
    pval       = 0.
    confidence = 1e-5
    while pval<=truth:
        pval        = chain.confidence(param,confidence)
        confidence += eps
    return sig[np.where(phi>confidence)[0][0]]


def get_samples(chain,param):
    pars = [x.name for x in chain.paramNames.names]    
    return chain.samples[:,pars.index(param)]

def add_S8x(chain):
    OmM = get_samples(chain,'OmM')
    sig8= get_samples(chain,'sigma8')
    S8x = sig8*(OmM/0.3)**0.4
    chain.addDerived(S8x,'S8x',label='S_8^X')

# add BAO-like prior
def add_OmM_prior(chain,OmM,sOmM):
    nchain = deepcopy(chain)
    pars   = [x.name for x in nchain.paramNames.names]
    ommsmp = chain.samples[:,pars.index('OmM')]
    nchain.weights *= np.exp(-(ommsmp-OmM)**2/2/sOmM**2)
    return nchain


def make_triangle_plot(samples,legend_labels,params_to_show,truth=None,contour_ls=None,contour_lws=None,contour_colors=None,
                       filled=True,title_x=None,title_y=None,title=None,save_path=None,markers=None,fontsize_title=None,ncol=1,
                       legend_fontsize=20,add_1d_constraints=True,add_confidence_truth=False,legend_loc='upper right',figsize=8):
    # Plot settings
    g = gdplt.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.8
    g.settings.legend_fontsize = legend_fontsize
    g.settings.axes_labelsize = 28 #19
    g.settings.axes_fontsize = 20 #16
    g.settings.axis_marker_color = 'k'
    g.settings.axis_marker_lw = 1.2
    g.settings.figure_legend_ncol = ncol
    g.settings.fig_width_inch = figsize
    g.settings.linewidth_contour = 3
    # defaults
    if contour_ls is None: contour_ls = ['-']*len(samples)
    if contour_lws is None: contour_lws = [3]*len(samples)
    if contour_colors is None: contour_colors = [f'C{i}' for i in range(len(samples))]
    # Generate the triangular plot with selected parameters
    g.triangle_plot(samples, params_to_show,
                    filled=filled,
                    contour_ls=contour_ls,
                    contour_lws=contour_lws,
                    legend_labels=legend_labels,
                    legend_loc=legend_loc,
                    contour_colors=contour_colors,
                    markers=markers,
                    line_args=[{'ls': ls, 'lw': lw, 'color': color} for ls, lw, color in zip(contour_ls, contour_lws, contour_colors)])
    if title:
        plt.suptitle(title, fontsize=fontsize_title, x=title_x, y=title_y)
    if save_path:
        plt.savefig(save_path, dpi=100)
    if truth:
        g.add_param_markers(dict(zip(params_to_show[:len(truth)],truth)),lw=3,color='k',ls='--' )
        
    Nchains = len(samples)
    if add_1d_constraints:
        for i in range(len(params_to_show)):
            for j,chain in enumerate(samples):
                s = r'{}'.format('$'+chain.getInlineLatex(params_to_show[i])+'$')
                if add_confidence_truth:
                    sigstar = get_confidence_sigma(chain,params_to_show[i],truth[i])
                    s+= ' $'+f'({sigstar:0.02f}'+r'\sigma)'+'$'
                g.add_text(s,x=0.025,y=1.+0.13*(Nchains-j),ax=[i,i],color=contour_colors[j],fontsize=15)
    plt.show()