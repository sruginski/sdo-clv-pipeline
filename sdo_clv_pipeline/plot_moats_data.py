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

all_simple = [[], [], []]
all_complex = [[], [], []]

def power_law(x, a, b):
    return a*x**b

def reduced_chi_squared(y_obs, y_fit, n_params=2):
    return np.sum((y_obs - y_fit) ** 2) / (len(y_obs) - n_params)

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

    exponents_simple_all = all_simple
    exponents_complex_all = all_complex

    # get exponents
    for j in range(3):
        for idx_set, container in zip([simple_idxs, complex_idxs],[exponents_simple_all[j], exponents_complex_all[j]]):

            for i in idx_set:
                moat_values = moats[j][i][:len(x[i])]
                cutoff = np.sqrt(areas[i] / np.pi)

                x_data = np.array(x[i])
                y_data = np.array(moat_values)

                mask = (x_data > 0) & (y_data > 0) & (x_data <= cutoff)
                x_fit = x_data[mask]
                y_fit = y_data[mask]

                if len(x_fit) >= 3:
                    logx = np.log10(x_fit)
                    logy = np.log10(y_fit)
                    slope, intercept = np.polyfit(logx, logy, 1)
                    y_model = 10**(intercept + slope * logx)

                    # relative chi-square
                    chi2_red = np.mean(((y_fit - y_model) / y_model)**2)

                    print(chi2_red)

                    if chi2_red < 0.5:
                        container.append(slope)
                        print("good fit")
                    else:
                        print("bad fit")
                    

def plot_final_histograms():
    labels = ["Velocity", "Magnetic Field", "Intensity"]

    for j in range(3):

        simple_vals = np.array(all_simple[j])
        complex_vals = np.array(all_complex[j])

        if len(simple_vals) == 0 and len(complex_vals) == 0:
            continue

        # Same axis limits / bin edges
        combined = np.concatenate([simple_vals, complex_vals])
        xmin, xmax = np.min(combined), np.max(combined)
        bins = np.linspace(xmin, xmax, 20)

        plt.figure(figsize=(12, 5))

        # Simple (blue)
        plt.subplot(1, 2, 1)
        plt.hist(simple_vals, bins=bins, color='blue', alpha=0.75)
        plt.title(f"{labels[j]} Simple Moats")
        plt.xlabel("Power-law exponent")
        plt.ylabel("Count")
        plt.xlim(xmin, xmax)

        # Complex (red)
        plt.subplot(1, 2, 2)
        plt.hist(complex_vals, bins=bins, color='red', alpha=0.75)
        plt.title(f"{labels[j]} Complex Moats")
        plt.xlabel("Power-law exponent")
        plt.ylabel("Count")
        plt.xlim(xmin, xmax)

        plt.suptitle(f"Power-law Exponent Distribution {labels[j]}")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # load_and_plot()
    plot_loop()
    
