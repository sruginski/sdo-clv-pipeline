import numpy as np
from .sdo_io import *
from .sdo_image import *

import pdb, warnings
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning


def plot_image(sdo_image, outdir=None, fname=None):
    # get the WCS
    wcs = sdo_image.wcs

    # initialize the figure
    fig = plt.figure(figsize=(6.4, 4.8))
    ax1 = fig.add_subplot(111, projection=wcs)
    ax1.set_aspect("equal")

    if sdo_image.is_magnetogram():
        # get cmap
        cmap = plt.get_cmap("RdYlBu").copy()
        cmap.set_bad(color="white")

        # get the norm
        norm = colors.SymLogNorm(1, vmin=-4200, vmax=4200)

        # plot the sun
        img = ax1.imshow(sdo_image.image, cmap=cmap, origin="lower", norm=norm, interpolation=None)
        sp.visualization.wcsaxes_compat.wcsaxes_heliographic_overlay(ax1, grid_spacing=15*u.deg, annotate=True,
                                                             color="k", alpha=0.5, ls="--", lw=0.5)
        limb = ax1.contour(sdo_image.mu >= 0.0, colors="k", linestyles="--", linewidths=0.5, alpha=0.5)
        clb = fig.colorbar(img)
        clb.set_label(r"${\rm Magnetic\ Field\ Strength\ (G)}$")
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        ax1.set_xlabel(r"${\rm Helioprojective\ Longitude}$")
        ax1.set_ylabel(r"${\rm Helioprojective\ Latitude}$")
        # ax1.text(1400, 4000, sdo_image.date_obs, fontsize=10, c="black")
        ax1.grid(False)


        if outdir is not None:
            if fname is None:
                fname = "mag_" + sdo_image.date_obs + ".pdf"
            fig.savefig(outdir + fname, bbox_inches="tight", dpi=500)
            plt.clf(); plt.close()

        else:
            plt.show()

        return None

    elif sdo_image.is_dopplergram():
        # get cmap
        cmap = plt.get_cmap("seismic").copy()
        cmap.set_bad(color="white")

        # plot the sun
        img = ax1.imshow(sdo_image.v_corr, origin="lower", cmap=cmap, vmin=-2000, vmax=2000, interpolation=None)
        sp.visualization.wcsaxes_compat.wcsaxes_heliographic_overlay(ax1, grid_spacing=15*u.deg, annotate=True,
                                                                     color="k", alpha=0.5, ls="--", lw=0.5)
        limb = ax1.contour(sdo_image.mu >= 0.0, colors="k", linestyles="--", linewidths=0.5, alpha=0.5)
        clb = fig.colorbar(img)
        clb.set_label(r"${\rm LOS\ Velocity\ } {\rm(m\ s}^{-1}{\rm )}$")
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        ax1.set_xlabel(r"${\rm Helioprojective\ Longitude}$")
        ax1.set_ylabel(r"${\rm Helioprojective\ Latitude}$")
        # ax1.text(1400, 4000, sdo_image.date_obs, fontsize=10, c="black")
        ax1.grid(False)

        # figure out the filename

        if outdir is not None:

            if fname is None:
                fname = "dop_" + sdo_image.date_obs + ".pdf"
            fig.savefig(outdir + fname, bbox_inches="tight", dpi=500)
            plt.clf(); plt.close()

        else:
            plt.show()

        return None

    elif sdo_image.is_continuum():
        # get cmap
        cmap = plt.get_cmap("afmhot").copy()
        cmap.set_bad(color="white")

        # plot the sun
        img = ax1.imshow(sdo_image.iflat/sdo_image.ld_coeffs[0], cmap=cmap, origin="lower", interpolation=None)#, vmin=20000)
        sp.visualization.wcsaxes_compat.wcsaxes_heliographic_overlay(ax1, grid_spacing=15*u.deg, annotate=True,
                                                                     color="k", alpha=0.5, ls="--", lw=0.5)
        limb = ax1.contour(sdo_image.mu >= 0.0, colors="k", linestyles="--", linewidths=0.5, alpha=0.5)

        # plot the colorbar
        clb = fig.colorbar(img)
        clb.set_label(r"${\rm Relative\ HMI\ Continuum\ Intensity}$")

        # axes and stuff
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        ax1.set_xlabel(r"${\rm Helioprojective\ Longitude}$")
        ax1.set_ylabel(r"${\rm Helioprojective\ Latitude}$")
        # ax1.text(1400, 4000, sdo_image.date_obs, fontsize=10, c="black")
        ax1.grid(False)

        # figure out the filename
        if outdir is not None:
            if fname is None:
                fname = "con_" + sdo_image.date_obs + ".pdf"
            fig.savefig(outdir + fname, bbox_inches="tight", dpi=500)
            plt.clf(); plt.close()
        else:
            plt.show()
        return None

    elif sdo_image.is_filtergram():
        # get cmap
        cmap = plt.get_cmap("Purples_r").copy()
        cmap.set_bad(color="white")

        # plot the sun
        img = ax1.imshow(sdo_image.iflat/sdo_image.ld_coeffs[0], cmap=cmap, origin="lower", interpolation=None)#, vmin=20000)
        sp.visualization.wcsaxes_compat.wcsaxes_heliographic_overlay(ax1, grid_spacing=15*u.deg, annotate=True,
                                                                     color="k", alpha=0.5, ls="--", lw=0.5)
        limb = ax1.contour(sdo_image.mu >= 0.0, colors="k", linestyles="--", linewidths=0.5, alpha=0.5)
        clb = fig.colorbar(img)
        clb.set_label(r"${\rm Relative\ 1700\ \AA \ Continuum\ Intensity}$")
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        ax1.set_xlabel(r"${\rm Helioprojective\ Longitude}$")
        ax1.set_ylabel(r"${\rm Helioprojective\ Latitude}$")
        # ax1.text(1400, 4000, sdo_image.date_obs, fontsize=10, c="black")
        ax1.grid(False)

        # figure out the filename
        if outdir is not None:
            if fname is None:
                fname = "aia_" + sdo_image.date_obs + ".pdf"
            fig.savefig(outdir + fname, bbox_inches="tight", dpi=500)
            plt.clf(); plt.close()
        else:
            plt.show()
        return None
    
    else:
        return None

def plot_mask(mask, outdir=None, fname=None):
    # merge the penumbra
    mask_copy = np.copy(mask.regions)
    mask_copy[mask_copy >= 3] -= 1

    # get cmap
    cmap = colors.ListedColormap(["black", "saddlebrown", "orange", "yellow", "white"])
    cmap.set_bad(color="white")
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], ncolors=cmap.N, clip=True)

    # get the WCS
    wcs = mask.wcs

    # plot the sun
    fig = plt.figure(figsize=(6.4, 4.8))
    ax1 = fig.add_subplot(111, projection=wcs)
    img = ax1.imshow(mask_copy - 0.5, cmap=cmap, norm=norm, origin="lower", interpolation=None)
    sp.visualization.wcsaxes_compat.wcsaxes_heliographic_overlay(ax1, grid_spacing=15*u.deg, annotate=True,
                                                                 color="k", alpha=0.5, ls="--", lw=0.5)
    limb = ax1.contour(mask.mu >= 0.0, colors="k", linestyles="--", linewidths=0.5, alpha=0.5)
    clb = fig.colorbar(img, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
    clb.ax.set_yticklabels([r"${\rm Umbra}$", r"${\rm Penumbra}$", r"${\rm Quiet\ Sun}$", r"${\rm Network}$", r"${\rm Plage}$"])
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.set_xlabel(r"${\rm Helioprojective\ Longitude}$")
    ax1.set_ylabel(r"${\rm Helioprojective\ Latitude}$")
    # ax1.text(1400, 4000, mask.date_obs, fontsize=10, c="black")
    ax1.grid(False)

    # figure out the filename
    if outdir is not None:
        if fname is None:
            fname = "mask_" + mask.date_obs + ".pdf"
        fig.savefig(outdir + fname, bbox_inches="tight", dpi=500)
        plt.clf(); plt.close()
    else:
        plt.show()
    return None

def label_moats_on_sun(mask, outdir=None, fname=None):
        # merge the penumbra
        mask_copy = np.copy(mask.regions)
        mask_copy[mask_copy >= 3] -= 1

        # get cmap
        cmap = colors.ListedColormap(["black", "saddlebrown", "orange", "yellow", "white"])
        cmap.set_bad(color="white")
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], ncolors=cmap.N, clip=True)

        # get the WCS
        wcs = mask.wcs

        # plot the sun
        fig = plt.figure(figsize=(6.4, 4.8))
        ax1 = fig.add_subplot(111, projection=wcs)
        img = ax1.imshow(mask_copy - 0.5, cmap=cmap, norm=norm, origin="lower", interpolation=None)
        sp.visualization.wcsaxes_compat.wcsaxes_heliographic_overlay(ax1, grid_spacing=15*u.deg, annotate=True,
                                                                    color="k", alpha=0.5, ls="--", lw=0.5)
        limb = ax1.contour(mask.mu >= 0.0, colors="k", linestyles="--", linewidths=0.5, alpha=0.5)
        clb = fig.colorbar(img, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
        clb.ax.set_yticklabels([r"${\rm Umbra}$", r"${\rm Penumbra}$", r"${\rm Quiet\ Sun}$", r"${\rm Network}$", r"${\rm Plage}$"])
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        ax1.set_xlabel(r"${\rm Helioprojective\ Longitude}$")
        ax1.set_ylabel(r"${\rm Helioprojective\ Latitude}$")
        # ax1.text(1400, 4000, mask.date_obs, fontsize=10, c="black")
        ax1.grid(False)

        data = np.load('moats_data.npz')
        print(data.files)
        area_idx_arr = data['area_idx_arr']
        letters = data['letters']
        print(letters)

        moat_colors = plt.cm.tab20.colors

        for i, (spot_idxs, label) in enumerate(zip(area_idx_arr, letters)):
            color = moat_colors[i%len(moat_colors)]
            x, y = np.unravel_index(spot_idxs, mask.regions.shape)
            ax1.scatter(x,y, color=color, transform=ax1.get_transform('pixel'))

            x_center = np.mean(x)
            y_center = np.mean(y)
            ax1.text(x_center, y_center, label, transform=ax1.get_transform('pixel'))

        # figure out the filename
        if outdir is not None:
            if fname is None:
                fname = "mask_" + mask.date_obs + ".pdf"
            fig.savefig(outdir + fname, bbox_inches="tight", dpi=500)
            plt.clf(); plt.close()
        else:
            plt.show()
        return None