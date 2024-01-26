## Codes to normalize reflectance to nadir BRDF adjusted reflectance (last updated by Myung Sik Cho: 2024-01-26)
## input data should be: a signle image with 5 bands in order: (1) reflectance, (2) view azimuth angles, (3) view zenith angle, (4) solar azimuth angle. and (5) solar zenith angle
## The code is based on Sentinel-2 MSI 20 m imagery, so you may make some changes when you apply it to Landsat.
##############################################################################
## This code is based on Hankui Zhang's c codes
## HZ's code is based on below research papers
'''
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
'''
###############################################################################

import rasterio, glob, math
import numpy as np

def deg2rad(x):
    return(x*math.pi/180)

def sec_cal(x):
    return(1/math.cos(x))

def relative_azimuth(saa, vaa):
    phi = abs(saa-vaa)
    diff = 0
    if phi>math.pi:
        diff = 2*math.pi - phi
    else:
        diff = phi
    return(diff)

def build_constants(sza, vza, saa, vaa):
    phi = relative_azimuth(saa, vaa)
    
    cos = {
        'sza': math.cos(sza),
        'vza': math.cos(vza),
        'phi': math.cos(phi)}
    sin = {
        'sza': math.sin(sza),
        'vza': math.sin(vza),
        'phi': math.sin(phi)}
    sec = {
        'sza': sec_cal(sza),
        'vza': sec_cal(vza)}    
    tan = {
        'sza':sin.get('sza') / cos.get('sza'),
        'vza':sin.get('vza') / cos.get('vza')}
    cos['cos_xi'] = cos.get('sza')*cos.get('vza')+sin.get('sza')*sin.get('vza')*cos.get('phi')
    
    c = {'cos':cos,'sin':sin,'sec':sec,'tan':tan}
    return(c)

def calc_kgeo(c):
    dsq = pow(c.get('tan').get('sza'),2) + pow(c.get('tan').get('vza'),2) - 2*c.get('tan').get('sza')*c.get('tan').get('vza')*c.get('cos').get('phi')
    tantansin = c.get('tan').get('sza')*c.get('tan').get('vza')*c.get('sin').get('phi')
    costtop = math.sqrt(dsq + pow(tantansin,2))
    
    cost = 2*costtop/(c.get('sec').get('sza')+c.get('sec').get('vza'))
    t = math.acos(np.min([1,cost]))
    
    big_o = (1/math.pi)*(t-math.sin(t)*math.cos(t))*(c.get('sec').get('sza')+c.get('sec').get('vza'))
    big_o_2 = (c.get('sec').get('sza')+c.get('sec').get('vza'))
    big_o_3 = (0.5)*(1+c.get('cos').get('cos_xi'))*c.get('sec').get('sza')*c.get('sec').get('vza')
    kgeo = big_o - big_o_2 + big_o_3
      
    return kgeo

def calc_kvol(c):
    xi = math.acos(c.get('cos').get('cos_xi'))
    kvol = ((math.pi/2-xi)*c.get('cos').get('cos_xi')+math.sin(xi))/(c.get('cos').get('sza')+c.get('cos').get('vza'))-(math.pi/4)
    return(kvol)
    
f_values = { #[f_iso, f_geo, f_vol]
    'B03':[0.1306, 0.0178, 0.0580],
    'B04':[0.1690, 0.0227, 0.0574],
    'B05':[0.2085, 0.0256, 0.0845],
    'B06':[0.2316, 0.0273, 0.1003],
    'B07':[0.2599, 0.0294, 0.1197 ],
    'B8A':[0.3093, 0.033, 0.1535],
    'B11':[0.3430, 0.0453, 0.1154],
    'B12':[0.2658, 0.0387, 0.063]}
    
def calc_rho_modis(kgeo, kvol, f):
    return (f[0]+ f[1]*kgeo + f[2]*kvol)

def calc_c_lambda(kernels, f):
    return (calc_rho_modis(kernels.get('kgeo_vza_zero'), kernels.get('kvol_vza_zero'), f) / calc_rho_modis(kernels.get('kgeo'), kernels.get('kvol'), f))

def calc_nbar(r_s2, f, kernels):
    c_lambda = calc_c_lambda(kernels, f)
    return (c_lambda*r_s2)
    
def evaluatePixel (in_img):
    ### in_img shape should be h,5 (bands),w
    saa = deg2rad(in_img[3])
    sza = deg2rad(in_img[4])
    vaa = deg2rad(in_img[1])
    vza = deg2rad(in_img[2])
    
    r_s2 = in_img[0]
    sel_f = f_values.get(band_name)
    
    constant  = build_constants(sza, vza, saa, vaa)
    c_vza_zero = build_constants(sza, 0, saa, vaa)
    
    kgeo = calc_kgeo(constant)
    kvol = calc_kvol(constant)
    kgeo_vza_zero= calc_kgeo(c_vza_zero)
    kvol_vza_zero= calc_kvol(c_vza_zero)
    
    kernels = {
        'kgeo':kgeo,
        'kvol':kvol,
        'kgeo_vza_zero':kgeo_vza_zero,
        'kvol_vza_zero':kvol_vza_zero}
    
    out = calc_nbar(r_s2,sel_f,kernels)
    
    return(out)

def run_nbar (in_img_dir,out_dir):
    ## load
    img_open = rasterio.open(in_img_dir)
    img_profile = img_open.profile
    img_read = img_open.read()    
    
    ## change the array strucutre for running
    img_read_ma = np.moveaxis(img_read,0,-1)
    h,w,b = img_read_ma.shape
    img_read_ma_2d = np.reshape(img_read_ma,(h*w,b))
    
    ## run nbar
    n_bar = np.array(list(map(evaluatePixel,img_read_ma_2d)))
    print('nbar 1d -> 2d')
    n_bar_2d = np.reshape(n_bar,(h,w))
    
    ## export
    out_profile = img_profile.copy()
    out_profile.update({'count': 1,
                        'dtype': 'float64',})
    
    file_root = in_img_dir.split('/')[-1].split('_sr_')[0]
    out_file_n = out_dir+file_root+'_nbar.tif'
    
    with rasterio.open(out_file_n, 'w', **out_profile) as dst:
        dst.write(n_bar_2d,1)
        

if __name__=="__main__":
    base_dir = '/mnt/gs21/scratch/chomyun2/240125_nbar/'
    in_img_dir_l = glob.glob(base_dir+'imgs/*B8A*.tif')
    out_dir = base_dir+'nbar/'
    
    print(len(in_img_dir_l))
    for di in in_img_dir_l:
        print(di)
        band_name = di.split('/')[-1].split('_sr_')[0].split('_')[-2] ## you should specify band name which will be used in line 112 in function evaluatePixel
        print(band_name)
        run_nbar(di,out_dir)
        
    print('done')
    
    
    