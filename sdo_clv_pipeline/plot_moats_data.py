import matplotlib.cm as cm
import numpy as np
import pdb, warnings
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sunpy.map import Map as sun_map
from sunpy.coordinates import frames

from scipy import ndimage
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from skimage.measure import regionprops
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs.utils import proj_plane_pixel_scales

from .sdo_io import *
from .limbdark import *
from .legendre import *
from .reproject import *

def load_and_plot(npz_file):

    data = np.load("moats_data.npz")

    x = data['x']
    vels = data['vels']
    mags = data['mags'] 
    ints = data['ints']
    areas = data['areas']
    mus = data['mus']
    moats = data['moats']


    print ("trying to plot...")
    # layered plots for different moats
    thetas = []
    # plot avg velocities / dilations, mu
    for i in mus:
        i = np.arccos(i)
        thetas.append(i)
    cmap = cm.plasma
    norm = colors.Normalize(vmin=min(thetas), vmax=np.max(thetas))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    for i in range (0, len(thetas)):
        color = cmap(norm(thetas[i]))
        plt.plot(x, moats[0][i], color = color)
        # marker
        mark_dilation = np.sqrt(areas[i] / np.pi)
        vel_at_mark = np.interp(mark_dilation, x, moats[0][i])
        plt.plot(mark_dilation, vel_at_mark, marker='o', color=color, markersize=7)
    plt.colorbar(sm, label='Average Theta (rad)', ax=plt.gca())
    plt.xlabel("# of Dilations")
    plt.ylabel("Average Velocity (m/s)")
    plt.title("Average Velocity vs # of Dilations")
    plt.show()

    # plot avg velocities / dilations, area
    cmap = cm.plasma
    norm = colors.Normalize(vmin=min(areas), vmax=np.max(areas))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    for i in range (0, len(areas)):
        color = cmap(norm(areas[i]))
        plt.plot(x, moats[0][i], color = color)
        # marker
        mark_dilation = np.sqrt(areas[i] / np.pi)
        vel_at_mark = np.interp(mark_dilation, x, moats[0][i])
        plt.plot(mark_dilation, vel_at_mark, marker='o', color=color, markersize=7)
    plt.colorbar(sm, label='Area of Spot in Pixels', ax=plt.gca())
    plt.xlabel("# of Dilations")
    plt.ylabel("Average Velocity (m/s)")
    plt.title("Average Velocity vs # of Dilations")
    plt.show()

    # plot avg magnetic field strength / dilations, mu
    cmap = cm.plasma
    norm = colors.Normalize(vmin=min(thetas), vmax=np.max(thetas))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    for i in range (0, len(thetas)):
        color = cmap(norm(thetas[i]))
        plt.plot(x, moats[1][i], color = color)
        # marker
        mark_dilation = np.sqrt(areas[i] / np.pi)
        mag_at_mark = np.interp(mark_dilation, x, moats[1][i])
        plt.plot(mark_dilation, mag_at_mark, marker='o', color=color, markersize=7)
    plt.colorbar(sm, label='Average Theta (rad)', ax=plt.gca())
    plt.xlabel("# of Dilations")
    plt.ylabel("Average Magnetic Field (G)")
    plt.title("Average Magnetic Field Strength vs # of Dilations")
    plt.show()

    # plot avg magnetic field strength / dilations, area
    cmap = cm.plasma
    norm = colors.Normalize(vmin=np.min(areas), vmax=np.max(areas))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    for i in range (0, len(areas)):
        color = cmap(norm(areas[i]))
        plt.plot(x, moats[1][i], color = color)
        # marker
        mark_dilation = np.sqrt(areas[i] / np.pi)
        mag_at_mark = np.interp(mark_dilation, x, moats[1][i])
        plt.plot(mark_dilation, mag_at_mark, marker='o', color=color, markersize=7)
    plt.colorbar(sm, label='Area of Spot in Pixels', ax=plt.gca())
    plt.xlabel("# of Dilations")
    plt.ylabel("Average Magnetic Field (G)")
    plt.title("Average Magnetic Field Strength vs # of Dilations")
    plt.show()

    #plot avg intensity / dilations, mu
    cmap = cm.plasma
    norm = colors.Normalize(vmin=min(thetas), vmax=np.max(thetas))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    for i in range (0, len(thetas)):
        color = cmap(norm(thetas[i]))
        plt.plot(x, moats[2][i], color = color)
            # marker
        mark_dilation = np.sqrt(areas[i] / np.pi)
        int_at_mark = np.interp(mark_dilation, x, moats[2][i])
        plt.plot(mark_dilation, int_at_mark, marker='o', color=color, markersize=7)
    plt.colorbar(sm, label='Average Theta (rad)', ax=plt.gca())
    plt.xlabel("# of Dilations")
    plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
    plt.title("Average Intensity vs # of Dilations")
    plt.show()

    #plot avg intensity / dilations, area
    cmap = cm.plasma
    norm = colors.Normalize(vmin=np.min(areas), vmax=np.max(areas))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    for i in range (0, len(areas)):
        color = cmap(norm(areas[i]))
        plt.plot(x, moats[2][i], color = color)
            # marker
        mark_dilation = np.sqrt(areas[i] / np.pi)
        int_at_mark = np.interp(mark_dilation, x, moats[2][i])
        plt.plot(mark_dilation, int_at_mark, marker='o', color=color, markersize=7)
    plt.colorbar(sm, label='Area of Spot in Pixels', ax=plt.gca())
    plt.xlabel("# of Dilations")
    plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
    plt.title("Average Intensity vs # of Dilations")
    plt.show()

if __name__ == '__main__':
    load_and_plot('moats_data.npz')