"""
Codes to normalize reflectance to nadir BRDF adjusted reflectance
Last updated by Yuean Qiu: 2024-01-29, adapted from Myung Sik Cho: 2024-01-26.

Landsat and Sentinel-2 share same coefficients (F_VALUES). 
Users should provide the band names, with the same order to the image bands.

For one Landsat ARD tile (1*5000*5000), it takes about 10~20 s (similar performance for more bands).

The code is based on https://github.com/sentinel-hub/custom-scripts/tree/main/sentinel-2/brdf


Reference
(1) Roy, D. P., Zhang, H. K., Ju, J., Gomez-Dans, J. L., Lewis, P. E., Schaaf, C. B., Sun Q., Li J., Huang H., & Kovalskyy, V. (2016). 
A general method to normalize Landsat reflectance data to nadir BRDF adjusted reflectance. 
Remote Sensing of Environment, 176, 255-271.

(2) Zhang, H. K., Roy, D. P., & Kovalskyy, V. (2016). 
Optimal solar geometry definition for global long-term Landsat time-series bidirectional reflectance normalization. 
IEEE Transactions on Geoscience and Remote Sensing, 54(3), 1410-1418.

(3) Roy, D. P., Li, J., Zhang, H. K., Yan, L., Huang, H., & Li, Z. (2017). 
Examination of Sentinel-2A multi-spectral instrument (MSI) reflectance anisotropy and the suitability of a general method to normalize MSI reflectance to nadir BRDF adjusted reflectance. 
Remote Sensing of Environment, 199, 25-38.

(4) Roy, D.P., Li, Z., Zhang, H.K., 2017. 
Adjustment of Sentinel-2 multi-spectral instrument (MSI) red-edge band reflectance to nadir BRDF adjusted reflectance (NBAR) and quantification of red-edge band BRDF effects. 
Remote Sensing, 9(12), 1325.
"""
###############################################################################

import numpy as np

class PRO_NBAR():
    """A calculator to get NBAR for Landsat and Sentinel-2.
    
    To use it, 1) make an instance, 2) prepare input, 3) call .evaluatePixel()
    See input requirement in .evaluatePixel().
    
    """
    
    # Kernel Parameters (Roy et al. 2017, Table 1)
    # the coefficient for brdf kernel, for both landsat and sentinel-2
    F_VALUES = { #[f_iso, f_geo, f_vol]
        'blue':         [0.0774, 0.0079, 0.0372],
        'green':        [0.1306, 0.0178, 0.0580],
        'red':          [0.1690, 0.0227, 0.0574],
        'red edge 1':   [0.2085, 0.0256, 0.0845],
        'red edge 2':   [0.2316, 0.0273, 0.1003],
        'red edge 3':   [0.2599, 0.0294, 0.1197],
        'nir':          [0.3093, 0.0330, 0.1535],
        'swir 1':       [0.3430, 0.0453, 0.1154],
        'swir 2':       [0.2658, 0.0387, 0.0639]}
    
    def __init__(self):
        self.name = "NBAR Calculator"
    
    def __sec_cal(self, x):
        # Calculate the secant of a value
        return(1/np.cos(x))
    
    def __relative_azimuth(self, saa, vaa):
        # Calculate relative azimuth angle
        # Angles in RAD !
        phi = np.abs(saa-vaa)
        diff = phi
        is_obtuse = phi > np.pi
        diff[is_obtuse] = 2*np.pi - phi[is_obtuse]
        return(diff)
    
    def __build_constants(self, sza, vza, saa, vaa):
        # calculates constants from viewing geometry that are often needed in the
        # calculations and are expensive to calculate (i.e. tan)
        phi = self.__relative_azimuth(saa, vaa)
        
        cos = {
            'sza': np.cos(sza),
            'vza': np.cos(vza),
            'phi': np.cos(phi)}
        sin = {
            'sza': np.sin(sza),
            'vza': np.sin(vza),
            'phi': np.sin(phi)}
        sec = {
            'sza': self.__sec_cal(sza),
            'vza': self.__sec_cal(vza)}    
        tan = {
            'sza':sin.get('sza') / cos.get('sza'),
            'vza':sin.get('vza') / cos.get('vza')}
        cos['cos_xi'] = cos.get('sza')*cos.get('vza')+sin.get('sza')*sin.get('vza')*cos.get('phi')
        
        c = {'cos':cos,'sin':sin,'sec':sec,'tan':tan}
        return(c)
    
    def __calc_kgeo(self, c):
        # Calculate the LiSparse kernel from Lucht et al. 2000
        # Angles in RAD !
        dsq = pow(c.get('tan').get('sza'),2) + pow(c.get('tan').get('vza'),2) - 2*c.get('tan').get('sza')*c.get('tan').get('vza')*c.get('cos').get('phi')
        tantansin = c.get('tan').get('sza')*c.get('tan').get('vza')*c.get('sin').get('phi')
        costtop = np.sqrt(dsq + pow(tantansin,2))
        
        cost = 2*costtop/(c.get('sec').get('sza')+c.get('sec').get('vza'))
        t = np.arccos(np.clip(cost, None, 1))
        
        big_o = (1/np.pi)*(t-np.sin(t)*np.cos(t))*(c.get('sec').get('sza')+c.get('sec').get('vza'))
        big_o_2 = (c.get('sec').get('sza')+c.get('sec').get('vza'))
        big_o_3 = (0.5)*(1+c.get('cos').get('cos_xi'))*c.get('sec').get('sza')*c.get('sec').get('vza')
        kgeo = big_o - big_o_2 + big_o_3
        
        return kgeo

    def __calc_kvol(self, c):
        # Calculate the RossThick kernel (k_vol) from Lucht et al. 2000 equation 38
        # Angles in RAD !
        xi = np.arccos(c.get('cos').get('cos_xi'))
        kvol = ((np.pi/2-xi)*c.get('cos').get('cos_xi')+np.sin(xi))/(c.get('cos').get('sza')+c.get('cos').get('vza'))-(np.pi/4)
        return(kvol)
        

    def __calc_rho_modis(self, kgeo, kvol, f):
        # Eq. 6 in Roy et al 2017, Eq. 37 in Lucht et al 2000
        return (f[0]+ f[1]*kgeo + f[2]*kvol)

    def __calc_c_lambda(self, kernels, f):
        # Part 2 of Eq. 5 in Roy et al 2017
        return (self.__calc_rho_modis(kernels.get('kgeo_vza_zero'), kernels.get('kvol_vza_zero'), f) / self.__calc_rho_modis(kernels.get('kgeo'), kernels.get('kvol'), f))

    def __calc_nbar(self, r_s2, f, kernels):
        # Part 1 of Eq. 5 in Roy et al 2017
        # r_s2: reflectance in band 
        # f: f values for band
        c_lambda = self.__calc_c_lambda(kernels, f)
        return (c_lambda*r_s2)

    def evaluatePixel (self, in_img, saa, sza, vaa, vza, band_names, flag_deg=True):
        """Make NBAR from off-ndair image and the angle images
        
        NBAR: Nadir BRDF adjusted reflectance.

        Args:
            in_img (np.array): shape (band, height, width).
            
            saa~vza (np.array): the angle images. Shape (1, height, width).
                (Angle convension: zenith [-90, 90], azimuth [0, 360])
                (NOTE!!! azimuth might be ok to range in [-180, 180], because only the relative azimuth is required.)
            
            band_names (list): the band names. Shape (band), same order as in_img. 
                (Supported: {'blue', 'green', 'red', 'red edge 1', 'red edge 2', 'red edge 3', 'nir', 'swir 1', 'swir 2'})
            
            flag_deg (bool): True if the angles are in degree. False if in radian.
            
        Return:
            np.array, NBAR, shape (band, height, width).
            
        """
        
        ### in_img shape should be h,5 (bands),w
        if flag_deg:
            saa = np.deg2rad(saa)
            sza = np.deg2rad(sza)
            vaa = np.deg2rad(vaa)
            vza = np.deg2rad(vza)
            
        constant  = self.__build_constants(sza, vza, saa, vaa)
        c_vza_zero = self.__build_constants(sza, 0, saa, vaa)
        
        kgeo = self.__calc_kgeo(constant)
        kvol = self.__calc_kvol(constant)
        kgeo_vza_zero= self.__calc_kgeo(c_vza_zero)
        kvol_vza_zero= self.__calc_kvol(c_vza_zero)
        
        kernels = {
            'kgeo':kgeo,
            'kvol':kvol,
            'kgeo_vza_zero':kgeo_vza_zero,
            'kvol_vza_zero':kvol_vza_zero}
        
        out = np.full_like(in_img, 0, dtype=np.float32)
        
        # process for each band, with same kernels but different coefficients
        for i, r_sr in enumerate(in_img):
            sel_f = self.F_VALUES.get(band_names[i])
            out[i] = self.__calc_nbar(r_sr, sel_f, kernels)
        
        return out
