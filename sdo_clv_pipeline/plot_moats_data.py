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
from string import ascii_letters

from .sdo_io import *
from .limbdark import *
from .legendre import *
from .reproject import *

def power_law(x, a, b):
    return a*x**b

def load_and_plot(moat_file):
    import matplotlib.cm as cm
    from string import ascii_letters

    data = np.load(moat_file)

    x = [np.arange(i) for i in data['x']]
    vels = data['vels']
    mags = data['mags']
    ints = data['ints']
    areas = data['areas']
    mus = data['mus']
    moat_types = data['types']

    moats = [vels, mags, ints, areas, mus]

    # separate moats
    simple_idxs = [i for i, val in enumerate(moat_types) if val == 0]
    complex_idxs = [i for i, val in enumerate(moat_types) if val == 1]

    # y axis labels
    ylabel = [
        "Average Velocity (m/s)",
        "Average Magnetic Field (G)",
        "Average Intensity (ergs / s / Hz / m²)"
    ]

    # Theta plots
    thetas = np.arccos(mus)
    cmap = cm.plasma
    norm_theta = colors.Normalize(vmin=thetas.min(), vmax=thetas.max())
    sm_theta = cm.ScalarMappable(norm=norm_theta, cmap=cmap)
    sm_theta.set_array([])

    for j in range(3):  # vel, mag, int
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True) # side by side subplots

        for idx_set, axis, label in zip([simple_idxs, complex_idxs], ax, ["Simple", "Complex"]):
        # plot simple on the left, complex on the right  
            for i in idx_set: # each idx in either simple or complex
                color = cmap(norm_theta(thetas[i]))
                label_letter = ascii_letters[i % 52]
                # moats_data = moats[j][i][:len(x[i])]

                # axis.plot(x[i], moats_data, color=color)

                moats_data = moats[j][i][:len(x[i])]
                mark_dilation = np.sqrt(areas[i] / np.pi)

                # only keep points before the marker
                x_plot = np.array(x[i])
                y_plot = np.array(moats_data)
                mask = x_plot <= mark_dilation
                x_plot_trunc = x_plot[mask]
                y_plot_trunc = y_plot[mask]

                axis.plot(x_plot_trunc, y_plot_trunc, color=color)


                x_data = np.array(x[i])
                y_data = np.array(moats_data)

                # valid = (x_data > 0) & (y_data > 0)
                # x_fit_data = x_data[valid]
                # y_fit_data = y_data[valid]

                valid = (x_data > 0) & (y_data > 0) & (x_data <= mark_dilation)
                x_fit_data = x_data[valid]
                y_fit_data = y_data[valid]


                if len(x_fit_data) >= 2:
                    popt, _ = curve_fit(power_law, x_fit_data, y_fit_data, p0=[1, -1])
                    x_fit = np.linspace(min(x_fit_data), max(x_fit_data), 100)
                    y_fit = power_law(x_fit, *popt)
                    axis.plot(x_fit, y_fit, linestyle='--', color=color, alpha=0.7, linewidth=1)

                mark_dilation = np.sqrt(areas[i] / np.pi)
                val_at_mark = np.interp(mark_dilation, x[i], moats_data)        
                axis.plot(mark_dilation, val_at_mark, marker='o', color=color, markersize=5)
                axis.text(mark_dilation + 0.2, val_at_mark + 0.5, f'{label_letter}', fontsize=9)


            axis.set_xscale('symlog')
            axis.set_xlabel("# of Dilations")
            axis.set_title(f"{label} Moats") # label moat types

        # y axis
        axis.set_yscale('symlog')
        ax[0].set_ylabel(ylabel[j])
        fig.colorbar(sm_theta, ax=ax, label='Average Theta (rad)')
        fig.suptitle(f"{ylabel[j]} vs # of Dilations")

        plt.show()

    # Area plots
    cmap = cm.plasma
    norm_area = colors.Normalize(vmin=min(areas), vmax=max(areas))
    sm_area = cm.ScalarMappable(norm=norm_area, cmap=cmap)
    sm_area.set_array([])

    for j in range(3):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True) # side by side subplots

        for idx_set, axis, label in zip([simple_idxs, complex_idxs], ax, ["Simple", "Complex"]): 
        # plot simple on the left, complex on the right
            for i in idx_set: # each idx in either simple or complex
                color = cmap(norm_area(areas[i]))
                label_letter = ascii_letters[i % 52]
                # moats_data = moats[j][i][:len(x[i])]
                # axis.plot(x[i], moats_data, color=color)

                moats_data = moats[j][i][:len(x[i])]
                mark_dilation = np.sqrt(areas[i] / np.pi)

                # only keep points before the marker
                x_plot = np.array(x[i])
                y_plot = np.array(moats_data)
                mask = (x_plot > 0) & (y_plot > 0) & (x_plot <= mark_dilation)
                x_plot_trunc = x_plot[mask]
                y_plot_trunc = y_plot[mask]

                axis.plot(x_plot_trunc, y_plot_trunc, color=color)

                
                # x_data = np.array(x[i])
                # y_data = np.array(moats_data)
                # valid = (x_data > 0) & (y_data > 0)
                # x_fit_data = x_data[valid]
                # y_fit_data = y_data[valid]
 
                # axis.plot(x_data, y_data, color=color)

                # valid = (x_data > 0) & (y_data > 0)
                # x_fit_data = x_data[valid]
                # y_fit_data = y_data[valid]

                valid = (x_data > 0) & (y_data > 0) & (x_data <= mark_dilation)
                x_fit_data = x_data[valid]
                y_fit_data = y_data[valid]

                if len(x_plot_trunc) >= 2:
                    popt, _ = curve_fit(power_law, x_plot_trunc, y_plot_trunc, p0=[1, -1])
                    x_fit_line = np.linspace(min(x_plot_trunc), max(x_plot_trunc), 100)
                    y_fit_line = power_law(x_fit_line, *popt)
                    axis.plot(x_fit_line, y_fit_line, linestyle='--', color=color, alpha=0.7, linewidth=1)
            

                val_at_mark = np.interp(mark_dilation, x_plot, y_plot)
                axis.plot(mark_dilation, val_at_mark, marker='o', color=color, markersize=5)
                axis.text(mark_dilation + 0.2, val_at_mark + 0.5, f'{ascii_letters[i % 52]}', fontsize=9)

            axis.set_xscale('symlog')
            axis.set_xlabel("# of Dilations")
            axis.set_title(f"{label} Moats") # label moat types

        # y axis
        axis.set_yscale('symlog')
        ax[0].set_ylabel(ylabel[j])
        fig.colorbar(sm_area, ax=ax, label='Area of Spot in Pixels')
        fig.suptitle(f"{ylabel[j]} vs # of Dilations")

        plt.show()

    exponents_simple_all = [[], [], []]   # vel, mag, int
    exponents_complex_all = [[], [], []]

    # power-law exponents
    for j in range(3):  # loop over vel, mag, int
        for idx_set, container in zip([simple_idxs, complex_idxs],
                                    [exponents_simple_all[j], exponents_complex_all[j]]):
            for i in idx_set:
                moats_data = moats[j][i][:len(x[i])]
                mark_dilation = np.sqrt(areas[i] / np.pi)

                # go to marker
                x_data = np.array(x[i])
                y_data = np.array(moats_data)
                mask = (x_data > 0) & (y_data > 0) & (x_data <= mark_dilation)
                x_fit_data = x_data[mask]
                y_fit_data = y_data[mask]

                if len(x_fit_data) >= 2:
                    popt, _ = curve_fit(power_law, x_fit_data, y_fit_data, p0=[1, -1])
                    exponent = popt[1]  # power law exponent
                    container.append(exponent)

    # combined histograms
    labels = ["Velocity", "Magnetic Field", "Intensity"]

    for j, label in enumerate(labels):
        exponents_simple = exponents_simple_all[j]
        exponents_complex = exponents_complex_all[j]

        # combine
        all_exponents = exponents_simple + exponents_complex
        x_min = min(all_exponents)
        x_max = max(all_exponents)
        bins = np.linspace(x_min, x_max, 10)

        # create subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)

        # simple moats
        ax[0].hist(exponents_simple, bins=bins, color='blue', edgecolor='black', alpha=0.8)
        ax[0].set_title(f"Simple Moats")
        ax[0].set_xlabel("Power-law exponent")
        ax[0].set_ylabel("Count")

        # complex moats
        ax[1].hist(exponents_complex, bins=bins, color='red', edgecolor='black', alpha=0.8)
        ax[1].set_title(f"Complex Moats")
        ax[1].set_xlabel("Power-law exponent")

        fig.suptitle(f"Histogram of Power-law Exponents - {label}", fontsize=12)
        plt.show()


if __name__ == '__main__':
    # load_and_plot()
    plot_loop()