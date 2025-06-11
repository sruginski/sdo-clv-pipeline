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
from reproject import reproject_interp
from skimage.measure import regionprops
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning

from .sdo_io import *
from .limbdark import *
from .legendre import *

warnings.simplefilter("ignore", category=VerifyWarning)
warnings.simplefilter("ignore", category=FITSFixedWarning)

class SDOImage(object):
    def __init__(self, file):
        # set the filename
        self.filename = file

        # get the image and the header
        self.image = read_data(self.filename)
        self.parse_header()

        # initialize mu_thresh
        self.mu_thresh = 0.0
        return None

    def parse_header(self):
        # read the header
        head = read_header(self.filename)
        self.wcs = WCS(head)

        # parse it
        self.naxis1 = head["NAXIS1"]
        self.naxis2 = head["NAXIS2"]
        self.crpix1 = head["CRPIX1"]
        self.crpix2 = head["CRPIX2"]
        self.cdelt1 = head["CDELT1"]
        self.cdelt2 = head["CDELT2"]
        self.date_obs = head["DATE-OBS"]

        self.L0 = head["CRLN_OBS"]
        self.B0 = head["CRLT_OBS"]

        self.dsun_obs = head["DSUN_OBS"]
        self.dsun_ref = head["DSUN_REF"]
        self.rsun_obs = head["RSUN_OBS"]
        self.rsun_ref = head["RSUN_REF"]

        self.obs_vr = head["OBS_VR"]
        self.obs_vw = head["OBS_VW"]
        self.obs_vn = head["OBS_VN"]

        if "CONTENT" in head.keys():
            self.content = head["CONTENT"]
        else:
            self.content = "FILTERGRAM"

        # get data quality flag
        self.instrument = head["TELESCOP"]
        if self.instrument == "SDO/AIA":
            self.quality = head["QUALLEV0"]
        elif self.instrument == "SDO/HMI":
            self.quality = head["QUALLEV1"]

        self.head = head
        return None

    def calc_geometry(self):
        # methods adapted from https://arxiv.org/abs/2105.12055
        # original implementation at https://github.com/samarth-kashyap/hmi-clean-ls
        # get sun map
        smap = sun_map(self.image, self.head)

        # do coordinate transforms / calculations
        paxis1 = np.arange(self.naxis1)
        paxis2 = np.arange(self.naxis2)
        xx, yy = np.meshgrid(paxis1, paxis2)
        self.hpc = smap.pixel_to_world(xx * u.pix, yy * u.pix)   # helioprojective cartesian
        self.rsun_solrad = self.dsun_obs/self.rsun_ref

        # transform to other coordinate systems
        hgs = self.hpc.transform_to(frames.HeliographicStonyhurst)
        hcc = self.hpc.transform_to(frames.Heliocentric)

        # get cartesian and radial coordinates
        self.xx = hcc.x
        self.yy = hcc.y
        self.rr = np.sqrt(self.hpc.Tx**2 + self.hpc.Ty**2) / (self.rsun_obs * u.arcsec)

        # heliocgraphic latitude and longitude
        self.lat = hgs.lat + 90 * u.deg
        self.lon = hgs.lon

        # get mu
        mask = self.rr <= 1.0
        self.mu = np.zeros((4096, 4096))
        self.mu[mask] = np.sqrt(1.0 - self.rr.value[mask]**2.0)
        self.mu[~mask] = np.nan
        return None

    def inherit_geometry(self, other_image):
        # self.xx = np.copy(other_image.xx)
        # self.yy = np.copy(other_image.yy)
        # self.rr = np.copy(other_image.rr)
        self.mu = np.copy(other_image.mu)
        # self.lat = np.copy(other_image.lat)
        # self.lon = np.copy(other_image.lon)
        return None

    def is_magnetogram(self):
        return self.content == "MAGNETOGRAM"

    def is_dopplergram(self):
        return self.content == "DOPPLERGRAM"

    def is_continuum(self):
        return self.content == "CONTINUUM INTENSITY"

    def is_filtergram(self):
        return self.content == "FILTERGRAM"

    def mask_low_mu(self, mu_thresh):
        self.mu_thresh = mu_thresh
        self.image[np.logical_or(self.mu < mu_thresh, np.isnan(self.mu))] = np.nan

        if self.is_continuum() | self.is_filtergram():
            self.ldark[np.logical_or(self.mu < mu_thresh, np.isnan(self.mu))] = np.nan
            self.iflat[np.logical_or(self.mu < mu_thresh, np.isnan(self.mu))] = np.nan
        return None

    def correct_magnetogram(self):
        assert self.is_magnetogram()
        self.B_obs = self.image.copy()
        self.image /= self.mu
        return None

    def correct_dopplergram(self, fit_cbs=False):
        assert self.is_dopplergram()

        # get mask excluding nans / sqrts of negatives
        # self.mask_nan = np.logical_and((self.rr <= 0.95), ~np.isnan(self.lat))
        self.mask_nan = (self.mu >= 0.1)

        # velocity components
        self.v_grav = 633 # m/s, constant
        self.calc_spacecraft_vel() # spacecraft velocity
        self.calc_bulk_vel(fit_cbs=fit_cbs) # differential rotation + meridional flows + cbs
        return None

    def calc_spacecraft_vel(self):
        # methods adapted from https://arxiv.org/abs/2105.12055
        # original implementation at https://github.com/samarth-kashyap/hmi-clean-ls
        assert self.is_dopplergram()

        # pre-compute trigonometric quantities
        sig = np.arctan(self.rr/self.rsun_solrad)
        chi = np.arctan2(self.xx, self.yy)
        sin_sig = np.sin(sig)
        cos_sig = np.cos(sig)
        sin_chi = np.sin(chi)
        cos_chi = np.cos(chi)

        # project satellite velocity into coordinate frame
        vr1 = self.obs_vr * cos_sig
        vr2 = -self.obs_vw * sin_sig * sin_chi
        vr3 = -self.obs_vn * sin_sig * cos_chi

        # reshape into 4096 x 4096 array
        self.v_obs = np.zeros((4096, 4096))
        self.v_obs[self.mask_nan] = -(vr1 + vr2 + vr3)[self.mask_nan]
        self.v_obs[~self.mask_nan] = np.nan
        return None

    def calc_bulk_vel(self, fit_cbs=False):
        # methods adapted from https://arxiv.org/abs/2105.12055
        # original implementation at https://github.com/samarth-kashyap/hmi-clean-ls
        assert self.is_dopplergram()

        # pre-compute trigonometric quantities
        cos_B0 = np.cos(self.B0)
        sin_B0 = np.sin(self.B0)

        self.lat_mask = self.lat[self.mask_nan]#.copy()
        self.lon_mask = self.lon[self.mask_nan]#.copy()
        self.rho_mask = self.rr[self.mask_nan]#.copy()

        cos_theta = np.cos(self.lat_mask)
        sin_theta = np.sin(self.lat_mask)
        cos_phi = np.cos(self.lon_mask)
        sin_phi = np.sin(self.lon_mask)

        self.lt = sin_B0 * sin_theta - cos_B0 * cos_theta * cos_phi
        self.lp = cos_B0 * sin_phi

        # calculate legendre poylnomials
        # print(">>> Generating ~Legendre~ Polynomials", flush=True)
        pl_theta, dt_pl_theta = gen_leg(5, self.lat_mask)
        if fit_cbs:
            pl_rho, dt_pl_rho = gen_leg_x(5, self.rho_mask)
        else:
            pl_rho, dt_pl_rho = gen_leg_x(0, self.rho_mask)
        # print(">>> Done generating ~Legendre~ Polynomials", flush=True)

        # figure out how many polynomials we need
        if fit_cbs:
            n_poly = 11
        else:
            n_poly = 6

        # allocate memory
        self.im_arr = np.zeros((n_poly, self.lt.shape[0]))

        # differential rotation (axisymmetric feature; s = 1, 3, 5)
        self.im_arr[0, :] = dt_pl_theta[1, :] * self.lp
        self.im_arr[1, :] = dt_pl_theta[3, :] * self.lp
        self.im_arr[2, :] = dt_pl_theta[5, :] * self.lp

        # meridional circulation (axisymmetric feature; s = 2, 4)
        # s = 0 is 0
        self.im_arr[3, :] = dt_pl_theta[2, :] * self.lt
        self.im_arr[4, :] = dt_pl_theta[4, :] * self.lt

        # axisymmetric feature (frame=pole at disk-center)
        # s = 0-5
        self.im_arr[5, :] = pl_rho[0, :]
        if fit_cbs:
            self.im_arr[6, :] = pl_rho[1, :]
            self.im_arr[7, :] = pl_rho[2, :]
            self.im_arr[8, :] = pl_rho[3, :]
            self.im_arr[9, :] = pl_rho[4, :]
            self.im_arr[10, :] = pl_rho[5, :]

        # get the data to fit and compute RHS
        self.dat = (self.image - self.v_obs - self.v_grav)[self.mask_nan].copy()
        self.RHS = self.im_arr.dot(self.dat)

        # fill the matrix
        A = np.zeros((n_poly, n_poly))
        for i in range(n_poly):
            for j in range(n_poly):
                A[i, j] = self.im_arr[i, :].dot(self.im_arr[j, :])

        # invert and compute fit params
        Ainv = inv_SVD(A, 1e5)
        self.fit_params = Ainv.dot(self.RHS)

        # get rotation component
        self.v_rot = np.zeros((4096, 4096))
        self.v_rot[self.mask_nan] = self.fit_params[:3].dot(self.im_arr[:3, :])
        self.v_rot[~self.mask_nan] = np.nan

        # get meridional circulation component
        self.v_mer = np.zeros((4096, 4096))
        self.v_mer[self.mask_nan] = self.fit_params[3:5].dot(self.im_arr[3:5, :])
        self.v_mer[~self.mask_nan] = np.nan

        # get convective blueshift w/ limb component
        self.v_cbs = np.zeros((4096, 4096))
        self.v_cbs[self.mask_nan] = self.fit_params[5:].dot(self.im_arr[5:, :])
        self.v_cbs[~self.mask_nan] = np.nan

        # get corrected velocity
        self.dat -= self.fit_params.dot(self.im_arr)
        self.v_corr = np.zeros((4096, 4096))
        self.v_corr[self.mask_nan] = self.dat
        self.v_corr[~self.mask_nan] = np.nan
        return None

    def calc_limb_darkening(self, mu_lim=0.1, num_mu=25, n_sigma=2.0):
        assert (self.is_continuum() | self.is_filtergram())

        # get average intensity in evenly spaced rings
        mu_edge = np.linspace(1.0, mu_lim, num=num_mu)
        avg_int = np.zeros(len(mu_edge)-1)
        for i in range(len(avg_int)):
            # find indices in ring that aren't nan
            inds = (self.mu > mu_edge[i+1]) & (self.mu <= mu_edge[i]) & (~np.isnan(self.image))

            # mask section that are big outliers
            ints = self.image[inds]
            ints[np.abs(ints - np.mean(ints)) >= (n_sigma * np.std(ints))] = np.nan
            avg_int[i] = np.nanmean(ints)

        # take averages in mu annuli to fit to
        mu_avgs = (mu_edge[1:] + mu_edge[0:-1]) / 2.0

        # set the initial guess parameters for optimization
        if self.is_continuum():
            p0 = [59000.0, 0.38, 0.23]
        elif self.is_filtergram():
            p0 = [1000, 0.9, -0.25]
        else:
            return None

        # do the fit and divide out the LD profile
        popt, pcov = curve_fit(quad_darkening, mu_avgs, avg_int, p0=p0)
        self.ld_coeffs = popt
        self.ldark = quad_darkening_two(self.mu, *popt[1:])
        self.iflat = self.image/self.ldark
        return None

    def rescale_to_hmi(self, hmi_image):
        assert self.is_filtergram()

        # rescale the image
        self.image = reproject_interp((self.image, self.head), hmi_image.head,
                                      return_footprint=False)

        # borrow the geometry now that the images are aligned
        self.inherit_geometry(hmi_image)
        self.wcs = hmi_image.wcs

# for creating pixel mask with thresholded regions
def calculate_weights(mag):
    # set magnetic threshold
    mag_thresh = 24.0/mag.mu

    # make flag array for magnetically active areas
    w_active = (np.abs(mag.image) > mag_thresh).astype(float)

    # convolve with boxcar filter to remove isolated pixels
    w_conv = ndimage.convolve(w_active, np.ones([3,3]), mode="constant")
    w_active = np.logical_and(w_conv >= 2., w_active == 1.)
    w_active[np.logical_or(mag.mu < mag.mu_thresh, np.isnan(mag.mu))] = False

    # make weights array for magnetically quiet areas
    w_quiet = ~w_active
    w_quiet[np.logical_or(mag.mu < mag.mu_thresh, np.isnan(mag.mu))] = False
    return w_active, w_quiet

class SunMask(object):
    def __init__(self, con, mag, dop, aia):
        # check argument order/names are correct
        assert con.is_continuum()
        assert mag.is_magnetogram()
        assert dop.is_dopplergram()
        assert aia.is_filtergram()

        # copy observation date
        self.date_obs = con.date_obs

        # inherit the geometry and the WCS
        self.wcs = WCS(read_header(con.filename))
        self.inherit_geometry(con)

        # calculate weights
        self.w_active, self.w_quiet = calculate_weights(mag)

        # calculate magnetic filling factor
        npix = np.nansum(con.mu >= con.mu_thresh)
        self.ff = np.nansum(self.w_active[con.mu >= con.mu_thresh]) / npix

        # identify regions
        self.identify_regions(con, mag, dop, aia)

        # get region fracs
        self.umb_frac = np.nansum(self.is_umbra()) / npix
        self.pen_frac = np.nansum(self.is_penumbra()) / npix
        self.blu_pen_frac = np.nansum(self.is_blue_penumbra()) / npix
        self.red_pen_frac = np.nansum(self.is_red_penumbra()) / npix
        self.quiet_frac = np.nansum(self.is_quiet_sun()) / npix
        self.network_frac = np.nansum(self.is_network()) / npix
        self.plage_frac = np.nansum(self.is_plage()) / npix

        return None

    def inherit_geometry(self, other_image):
        # self.xx = np.copy(other_image.xx)
        # self.yy = np.copy(other_image.yy)
        # self.rr = np.copy(other_image.rr)
        self.mu = np.copy(other_image.mu)
        # self.lat = np.copy(other_image.lat)
        # self.lon = np.copy(other_image.lon)
        return None

    def identify_regions(self, con, mag, dop, aia):
        # allocate memory for mask array
        self.regions = np.zeros(np.shape(con.image))

        # calculate intensity thresholds for HMI
        self.con_thresh1 = 0.89 * np.nansum(con.iflat * self.w_quiet)/np.nansum(self.w_quiet)
        self.con_thresh2 = 0.45 * np.nansum(con.iflat * self.w_quiet)/np.nansum(self.w_quiet)

        # get indices for umbrae
        ind1 = con.iflat <= self.con_thresh2      # if intensity less than thresh2, umbra (ind1)

        # get indices for penumbrae
        indp = (con.iflat <= self.con_thresh1) & (con.iflat > self.con_thresh2)
            # if flattened continuum intensity less than thresh1 and greater than thresh2, penumbra (ind1 or ind2)
        if hasattr(dop, "v_corr"):
            ind2 = indp & (dop.v_corr <= 0)    # if penumbra and bluehift or 0, ind2
            ind3 = indp & (dop.v_corr > 0)     # if penumbra and redshift, ind3
        else: # no attribute v_corr
            ind2 = indp
            ind3 = indp

        """
        # find contiguous penumbra regions
        structure = ndimage.generate_binary_structure(2,2)
        labels, nlabels = ndimage.label(indp, structure=structure)

        # find the mean mean of each bin
        index = np.arange(1, nlabels+1)
        mean_mus = ndimage.labeled_comprehension(self.mu, labels, index, np.mean, float, np.nan)

        # allocate memory, get indices for near- and far- penumbrae
        ind2_new = np.zeros(np.shape(self.regions), dtype=bool)
        ind3_new = np.zeros(np.shape(self.regions), dtype=bool)
        for i in index:
            ind2_new += ((labels == i) & (self.mu < mean_mus[i-1]))
            ind3_new += ((labels == i) & (self.mu >= mean_mus[i-1]))
        """

        # get indices for quiet sun
        ind4 = (con.iflat > self.con_thresh1) & self.w_quiet
            # if continuum intensity is greater than thresh1 and weak B field

        # calculate intensity thresholds for AIA
        weights = self.w_active * (~ind1) * (~ind2) * (~ind3)
        self.aia_thresh = np.nansum(aia.iflat * weights)/np.nansum(weights)

        # get indices for bright regions (plage/faculae + network)
        ind5a = (con.iflat > self.con_thresh1) & self.w_active    # intensity greater than thresh 1 and strong B field
        ind5b = (aia.iflat > self.aia_thresh) & (~ind1) & (~ind2) & (~ind3)  # > aia thresh and not umbra or penumbra
        ind5 = ind5a | ind5b # if ind5a or ind5b, bright

        # set mask indices
        self.regions[ind1] = 1 # umbrae
        self.regions[ind2] = 2 # blueshifted penumbrae
        self.regions[ind3] = 3 # redshifted penumbrae
        self.regions[ind4] = 4 # quiet sun
        self.regions[ind5] = 5 # bright areas (will separate into plage + network)

        # label unique contiguous bright regions (label islands of bright stuff)
        binary_img = self.regions == 5  # get bright areas 
        structure = ndimage.generate_binary_structure(2,2) # array of bools, defines feature connections
        labels, nlabels = ndimage.label(binary_img, structure=structure) 
            # takes bright areas and feature connections, gives each island a label

        # get labeled region areas and perimeters for bright areas
        rprops = regionprops(labels) # get the list of labels
        areas = np.array([rprop.area for rprop in rprops]).astype(float)    
            # for each label in list, get area as float (pixels) and put that in array
        areas *= (1e6/np.sum(self.mu > 0.0)) # convert to microhemispheres
        perims = np.array([rprop.perimeter for rprop in rprops]).astype(float) # for each island, get perimeter as float

        # area thresh is 20ppm of pixels on hemisphere
        pix_hem = np.nansum(con.mu > 0.0)
        area_thresh = 20e-6 * pix_hem

        # assign region type to plage for ratios less than ratio thresh
        ind6 = np.concatenate(([False], areas > area_thresh))[labels]
        self.regions[ind6] = 6 # plage


        # label each penumbra island and include umbra so we only expand outwards
        binary_img = (self.regions == 2) | (self.regions == 3)|(self.regions == 1) # get penumbra and umbra 
        structure = ndimage.generate_binary_structure(2,2) # binary structure (rank, connectivity)
        labels, nlabels = ndimage.label(binary_img, structure=structure) # label each island of umbra and penumbra

        plt.imshow(labels)
        plt.colorbar()
        plt.show()

        # print(np.shape(ind2 | ind3))

        # get labeled region areas and perimeters for umbra and penumbra
        rprops = regionprops(labels) # get list of islands
        areas = np.array([rprop.area for rprop in rprops]).astype(float) # get area for each island as float and put in arr
        areas *= (1e6/np.sum(self.mu > 0.0)) # convert to microhemispheres
        perims = np.array([rprop.perimeter for rprop in rprops]).astype(float) # get perimeter of each island

        areas_array = np.concatenate(([0.0], areas))[labels] 
        # plt.imshow(areas_array)
        # plt.colorbar()
        # # plt.show()

        perims_array = np.concatenate(([0.0], perims))[labels]
        # plt.imshow(perims_array)
        # plt.colorbar()
        # # plt.show()

        # max_area = np.max(areas_array) # get the max value in the array
        # max_area_idx = areas_array == max_area


        # save original array first
        save_arr = areas_array

        vels = []
        mags = []
        ints = []
        while (len(areas_array) > 0): # while the array is not empty --> change to while there is a non-zero value in the array
            # take island with max area // set specific area to get
            max_area = np.max(areas_array) # get the max value in the array
            max_area_idx = areas_array == max_area # go through areas_array and get list of indices of the pixels with specific area
            plt.imshow(max_area_idx) 
            plt.colorbar()
            plt.show()
            dilation_arr, avg_vel_arr = SunMask.plot_vel(dop, max_area_idx, structure)
            vels.append(avg_vel_arr)
            print(avg_vel_arr)
            dilation_arr, avg_mag_arr = SunMask.plot_mag(mag, max_area_idx, structure)
            mags.append(avg_mag_arr)
            print(avg_mag_arr)
            dilation_arr, avg_int_arr = SunMask.plot_int(con, max_area_idx, structure)
            ints.append(avg_int_arr)
            print(avg_int_arr)
            # remove pixels with max area value --> replace area of pixels with max area value with 0
            for area in areas_array:
                if area.any() == max_area:
                    areas_array[area] = 0

            print(areas_array)

            

        # layered graphs for different moats
        x = dilation_arr
        # plot avg velocities / dilations
        print ("trying to graph...")
        for i in avg_vel_arr:
            plt.plot(x, avg_vel_arr)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Velocity (m/s)")
        plt.title("Average Velocity vs # of Dilations")
        plt.show()
        
        # plot avg magnetic field strength / dilations
        for j in avg_mag_arr:
            plt.plot(x, avg_mag_arr)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Magnetic Field (G)")
        plt.title("Average Magnetic Field Strength vs # of Dilations")
        plt.show()

        #plot avg intensity / dilations
        for k in avg_int_arr:
            plt.plot(x, avg_int_arr)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
        plt.title("Average Intensity vs # of Dilations")
        plt.show()
        

        # investigating area and how many dilations we expect to go out
        # print(max_area)
        # r = np.sqrt((max_area)/pi)
        # print(r)

        '''
        max_area_idx = areas_array == max_area # go through areas_array and get list of indices of the pixels with the max area
        plt.imshow(max_area_idx) 
        plt.colorbar()
        plt.show()

        # set-up x axis for dilations plots
        max_dilations = 50  # how many dilations?
        dilation_arr = []   
        for i in range (1, max_dilations+1):
            dilation_arr.append(i)
        '''

        # velocity/dilations plot
        '''
        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        vel_arr = np.array(dop.v_corr[dilation])
        avg_vel = np.average(vel_arr)
        avg_vel_arr = []
        avg_vel_arr.append(avg_vel)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, prev_dilation)   # new outline, only that ring
            vel_arr = np.array(dop.v_corr[new_dilation]) 
            avg_vel = np.average(vel_arr)
            avg_vel_arr.append(avg_vel)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        print(dilation_arr) # check that numbers make sense
        print(avg_vel_arr)
        
        # plot
        x = dilation_arr
        y = avg_vel_arr
        plt.plot(x,y)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Velocity (m/s)")
        plt.title("Average Velocity vs # of Dilations")
        plt.show()
        '''
        
        # velocity/dilations plot inclusive
        '''
        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        vel_arr = np.array(dop.v_corr[dilation])
        avg_vel = np.average(vel_arr)
        avg_vel_arr = []
        avg_vel_arr.append(avg_vel)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, max_area_idx) # new outline including previous dilations
            vel_arr = np.array(dop.v_corr[new_dilation]) 
            avg_vel = np.average(vel_arr)
            avg_vel_arr.append(avg_vel)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        print(dilation_arr) # check that numbers make sense
        print(avg_vel_arr)
        
        # plot
        x = dilation_arr
        y = avg_vel_arr
        plt.plot(x,y)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Velocity (m/s)")
        plt.title("Average Velocity vs # of Dilations")
        plt.show()
        '''

        # intensity/dilations plot
        '''
        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        int_arr = np.array(con.image[dilation])
        avg_int = np.average(int_arr)
        avg_int_arr = []
        avg_int_arr.append(avg_int)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, prev_dilation)   # new outline, only that ring
            # new_dilation = np.logical_xor(new_dilated_idx, max_area_idx) # new outline including previous dilations
            int_arr = np.array(con.image[new_dilation]) 
            avg_int = np.average(int_arr)
            avg_int_arr.append(avg_int)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        print(dilation_arr) # check that numbers make sense
        print(avg_int_arr)
        
        # plot
        x = dilation_arr
        y = avg_int_arr
        plt.plot(x,y)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
        plt.title("Average Intensity vs # of Dilations")
        plt.show()
        '''

        # intensity/dilations plot inclusive
        '''
        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        int_arr = np.array(con.image[dilation])
        avg_int = np.average(int_arr)
        avg_int_arr = []
        avg_int_arr.append(avg_int)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, max_area_idx) # new outline including previous dilations
            int_arr = np.array(con.image[new_dilation]) 
            avg_int = np.average(int_arr)
            avg_int_arr.append(avg_int)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        print(dilation_arr) # check that numbers make sense
        print(avg_int_arr)
        
        # plot
        x = dilation_arr
        y = avg_int_arr
        plt.plot(x,y)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
        plt.title("Average Intensity vs # of Dilations")
        plt.show()
        '''

        # magnetic field/dilations plot
        '''
        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        mag_arr = np.array(mag.image[dilation])
        avg_mag = np.average(mag_arr)
        avg_mag_arr = []
        avg_mag_arr.append(avg_mag)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, prev_dilation)   # new outline, only that ring
            # new_dilation = np.logical_xor(new_dilated_idx, max_area_idx) # new outline including previous dilations
            mag_arr = np.array(mag.image[new_dilation]) 
            avg_mag = np.average(mag_arr)
            avg_mag_arr.append(avg_mag)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        print(dilation_arr) # check that numbers make sense
        print(avg_mag_arr)
        
        # plot
        x = dilation_arr
        y = avg_mag_arr
        plt.plot(x,y)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Magnetic Field (G)")
        plt.title("Average Magnetic Field Strength vs # of Dilations")
        plt.show()
        '''

        # magnetic field/dilations plot inclusive
        '''
        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        mag_arr = np.array(mag.image[dilation])
        avg_mag = np.average(mag_arr)
        avg_mag_arr = []
        avg_mag_arr.append(avg_mag)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, max_area_idx) # new outline including previous dilations
            mag_arr = np.array(mag.image[new_dilation]) 
            avg_mag = np.average(mag_arr)
            avg_mag_arr.append(avg_mag)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        print(dilation_arr) # check that numbers make sense
        print(avg_mag_arr)
        
        # plot
        x = dilation_arr
        y = avg_mag_arr
        plt.plot(x,y)
        plt.xlabel("# of Dilations")
        plt.ylabel("Average Magnetic Field (G)")
        plt.title("Average Magnetic Field Strength vs # of Dilations")
        plt.show()
        '''

        # early work
        '''
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure) 
        # use feature connections and indices of largest island to go out 1 pixel
        plt.imshow(dilated_idx)
        plt.colorbar()
        plt.show()

        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left
        plt.imshow(dilation) 
        plt.colorbar()
        plt.show()

        # have the outline of first dilation out from the perimeter of the penumbra of the island of max area plotted
        # calculate velocity/intensity of each pixel in the outline (dilation 1)
        # calculate average velocity/intensity 
        # plot average velocity vs # of dilations

        plt.imshow(dop.v_corr * dilation)
        plt.colorbar()
        plt.show()

        new_arr = dop.v_corr * dilation
        new_arr[new_arr == 0.0] = np.nan
        plt.imshow(new_arr)
        plt.colorbar()
        plt.show()

        vel_arr = np.array(dop.v_corr[dilation])
        print(vel_arr) #YAYYY array of velocities in dilation 1
        avg_vel = np.average(vel_arr)
        print(avg_vel) 
        '''

        # set isolated bright pixels to quiet sun
        ind_iso = np.concatenate(([False], areas == 1))[labels]
        self.regions[ind_iso] = 4 # quiet sun

        # make any remaining unclassified pixels quiet sun
        ind_rem = ((con.mu >= con.mu_thresh) & (self.regions == 0))
        self.regions[ind_rem] = 4 # quiet sun

        # set values beyond mu_thresh to nan
        self.regions[np.logical_or(con.mu <= con.mu_thresh, np.isnan(con.mu))] = np.nan

        return None

    def mask_low_mu(self, mu_thresh):
        self.mu_thresh = mu_thresh
        self.regions[np.logical_or(self.mu < mu_thresh, np.isnan(self.mu))] = np.nan
        return None

    def is_umbra(self):
        return self.regions == 1

    def is_penumbra(self):
        return np.logical_or(self.regions == 2, self.regions == 3)

    def is_blue_penumbra(self):
        return self.regions == 2

    def is_red_penumbra(self):
        return self.regions == 3

    def is_quiet_sun(self):
        return self.regions == 4

    def is_network(self):
        return self.regions == 5

    def is_plage(self):
        return self.regions == 6
    
    def plot_vel(dop, max_area_idx, structure):

        # set-up x axis for dilations plots
        max_dilations = 50  # how many dilations?
        dilation_arr = []   
        for i in range (1, max_dilations+1):
            dilation_arr.append(i)

        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        vel_arr = np.array(dop.v_corr[dilation])
        avg_vel = np.average(vel_arr)
        avg_vel_arr = []
        avg_vel_arr.append(avg_vel)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, prev_dilation)   # new outline, only that ring
            vel_arr = np.array(dop.v_corr[new_dilation]) 
            avg_vel = np.average(vel_arr)
            avg_vel_arr.append(avg_vel)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        # print(dilation_arr) # check that numbers make sense
        # print(avg_vel_arr)
        
        # # plot
        # x = dilation_arr
        # y = avg_vel_arr
        # plt.plot(x,y)
        # plt.xlabel("# of Dilations")
        # plt.ylabel("Average Velocity (m/s)")
        # plt.title("Average Velocity vs # of Dilations")
        # plt.show()

        return(dilation_arr, avg_vel_arr)
    
    def plot_mag(mag, max_area_idx, structure):

        # set-up x axis for dilations plots
        max_dilations = 50  # how many dilations?
        dilation_arr = []   
        for i in range (1, max_dilations+1):
            dilation_arr.append(i)

        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        mag_arr = np.array(mag.image[dilation])
        avg_mag = np.average(mag_arr)
        avg_mag_arr = []
        avg_mag_arr.append(avg_mag)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, prev_dilation)   # new outline, only that ring
            mag_arr = np.array(mag.image[new_dilation]) 
            avg_mag = np.average(mag_arr)
            avg_mag_arr.append(avg_mag)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        # print(dilation_arr) # check that numbers make sense
        # print(avg_mag_arr)
        
        # # plot
        # x = dilation_arr
        # y = avg_mag_arr
        # plt.plot(x,y)
        # plt.xlabel("# of Dilations")
        # plt.ylabel("Average Magnetic Field (G)")
        # plt.title("Average Magnetic Field Strength vs # of Dilations")
        # plt.show()

        return(dilation_arr, avg_mag_arr)
    
    def plot_int(con, max_area_idx, structure):
        
        # set-up x axis for dilations plots
        max_dilations = 50  # how many dilations?
        dilation_arr = []   
        for i in range (1, max_dilations+1):
            dilation_arr.append(i)
        
        # first dilation
        dilated_idx = ndimage.binary_dilation(max_area_idx, structure = structure)
        dilation = np.logical_xor(dilated_idx, max_area_idx) # dilated area - area = only outline left 
        int_arr = np.array(con.image[dilation])
        avg_int = np.average(int_arr)
        avg_int_arr = []
        avg_int_arr.append(avg_int)
        dilation_count = 1

        # y axis
        prev_dilation = dilated_idx
        while dilation_count < max_dilations:
            new_dilated_idx = ndimage.binary_dilation(prev_dilation, structure = structure)   # dilate
            new_dilation = np.logical_xor(new_dilated_idx, prev_dilation)   # new outline, only that ring
            # new_dilation = np.logical_xor(new_dilated_idx, max_area_idx) # new outline including previous dilations
            int_arr = np.array(con.image[new_dilation]) 
            avg_int = np.average(int_arr)
            avg_int_arr.append(avg_int)
            dilation_count += 1 # update dilation count
            prev_dilation = new_dilated_idx

        # print(dilation_arr) # check that numbers make sense
        # print(avg_int_arr)
        
        # # plot
        # x = dilation_arr
        # y = avg_int_arr
        # plt.plot(x,y)
        # plt.xlabel("# of Dilations")
        # plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
        # plt.title("Average Intensity vs # of Dilations")
        # plt.show()

        return(dilation_arr, avg_int_arr)
        
