import numpy as np
from scipy import stats
from astropy.io import fits
from scipy import ndimage
from skimage.filters import roberts
from matplotlib import pyplot as plt

import pdb 
from CraamTools.fit import Circle

fitsnames= ['20220725_flt-000001-206_13_25_28_813_32.fits' , 
            '20220725_flt-000001-206_13_25_28_813_10.fits' , 
            '20220725_flt-000001-206_13_25_28_813_128.fits', 
            '20220725_flt-000001-206_13_25_28_813_82.fits' ] 
image_size = 1280
camera_size = [640,480]
px_size = 2.6

########################################################################
#
# A python collection of scripts to create a mosaic image of the sun.
# I used the individual images obtained by scanning with the telescope
# used to create a flat field. Images were registered on 2022-07-25 at 16:25 UTC.
#
# I did not flatfield the images. Images were chosen to cover the entire
# solar disc.
#
# I normalized individually each image. However the first one has an
# illumination issue that I decided not to elliminate. It is not clear what
# is its origin, since it is not observed in the other images. Unfortunately,
# all the images from this "corner" have the same illimunation problem.
#
#
# Use sequence:
# >>> import limb
# >>> image,limbs = limb.FullSun(angle=10)
# The 'angle' parameter is the angle [deg] between successive slices
# of the solar disc. These slices are then averaged to get the 'mean limb'.
#
# The final mean limb seems to be very reasonable, despite the illumination problem
# that creates a 'bumb' in some slices.
#
#-----------------------------------------------------------------------
#
# Author: @guiguesp,
# Date:   2026-09-04, while I listen the good music selection
#         from Radioeins, "Nur für Erwachsener, natürlich!"
#
#################################################################################

def FullSun(angle=10):

    image = np.zeros([image_size,image_size])
    
    for name in fitsnames:
        f = fits.open(name)
        cx,cy = FindCenter(Normalize(f))
        dx = int(image_size//2 - cx)
        dy = int(image_size//2 - cy)
        image[dy:dy+camera_size[1],dx:dx+camera_size[0]] = np.copy(f[0].data)

    Create_Fits_Image(image)
    limbs = getLimbs(image,angle)
    Create_Fits_Limbs(limbs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,extent=[-640*2.6,640*2.6,-640*2.6,640*2.6],origin='lower',cmap='gray',vmin=0.3,vmax=1.1)
    ax.set_xlabel('X [arcsec]')
    ax.set_ylabel('Y [arcsec]')
    ax.set_title('AR30T Mosaic')
    plt.savefig('AR30T_Mosaic.pdf', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format='pdf',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                metadata=None)    
    
    return image,limbs


def FindCenter(f):
    
    image= f[0].data.astype(np.float32)
    # Normalize FITS images
    image -= np.min(image)
    max_val = np.max(image)
    if max_val > 0:
        image /= max_val
    
    # --- Find the center of the solar disk ---
    # Identify bright, non-saturated pixels as part of the disk
    max_val = np.max(image)
    disk_mask = image > (max_val - max_val / 5.0)
        
    if not np.any(disk_mask):
        return -99,-99

    # Use Roberts filter to find the limb (edge)
    limb_image = roberts(disk_mask)
    limb_points = np.argwhere(limb_image > 0)
        
    if limb_points.shape[0] < 50: # Need enough points to fit a circle
        return -99,-99

    # Fit a circle to the limb points
    y_coords, x_coords = limb_points[:, 0], limb_points[:, 1]
    points = np.column_stack([x_coords, y_coords])
         
    try:
        par,_ = Circle.fit(x_coords,y_coords)
        xc = par[0]
        yc = par[1]
        r = par[2]

    except Exception:
        print(f"  Failed to fit the limb")
        return # Skip if fitting fails

    return int(round(xc)), int(round(yc))
    
def Normalize(f):

    # I tried many different methods.
    # The histograms do not 

    # Second Method: Median
    x = f[0].data < 300
    m = np.median(f[0].data[x])
    x = f[0].data > 400
    M = np.median(f[0].data[x])

    # Third Method: Mode
    x = f[0].data < 300
    m = stats.mode(f[0].data[x]).mode
    x = f[0].data > 400
    M = stats.mode(f[0].data[x]).mode
    
    f[0].data = (f[0].data-m)/(M-m)

    return f


def getLimbs(im,a):

    N = int(180/a)
    l = np.zeros([image_size,N])
    ml = np.zeros(image_size)
    
    for i in np.arange(N):
        angle = a * i 
        r = ndimage.rotate(im,angle=angle,reshape=False,order=1)
        l[:,i] = r[int(image_size//2),:]
        ml = (ml+l[:,i])

    x = np.linspace(-image_size/2,image_size/2,image_size) * px_size

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,ml/N,'-k')
    ax.set_title('AR30T Mean Normalized Limb',fontsize=12)
    ax.set_xlabel('X [arcsec]')
    ax.set_ylabel('Normalized')
    plt.savefig('AR30T_Mean_Normalized_Limb.pdf', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format='pdf',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                metadata=None)    
    
    return {'x':x,'limbs':l,'mean_limb':ml/N,'angle':a}


def Create_Fits_Image(image):

    name = 'AR30T'
    instrument = 'FLIR A645sc'
    origin = 'OAFA/UNSJ - CRAAM/UPM'
    observatory = 'Obs. Astronomico Felix Aguilar'
    place = 'El Leoncito - San Juan, Argentina'
    longitude = -69.318889
    latitude = -31.802222
    elevation = 2430
    telescope = 'AR30T'
    comments = ['Level 0.0: uncalibrated image','COPYRIGHT. Grant of use.',
                'These data are property of Universidad Presbiteriana Mackenzie and Observatorio Astronomico Felix Aguilar.',
                'The Centro de Radio Astronomia e Astrofisica Mackenzie',
                'and Observatorio Astronomico Felix Aguilar are responsible for their distribution.',
                'Grant of use permission is given for Academic purposes only.',
                'Contact:guigue@craam.mackenzie.br; fernando.lopez@um.edu.ar' ]

    hdr = fits.Header()

    # General information
    hdr['SIMPLE']   = True
    hdr['BITPIX']   = -32
    hdr['NAXIS']    = 2
    hdr['NAXIS1']   = (image_size, ' ')
    hdr['NAXIS2']   = (image_size, ' ')
    hdr['EXTEND']   = (True, ' ')
    hdr['TELESCOP'] = (name, ' ')
    hdr['INSTRUME'] = (instrument, ' ')
    hdr['ORIGIN']   = (origin, ' ')
    hdr['DATE']     = ('2022-07-05', ' ')
    hdr['DATE-OBS'] = ('2022-07-05T13:25:28', ' ')
    hdr['DATA_TYP'] = ('Mosaic',' ')
    hdr['LVL_NUM']  = ('0.0', ' ')
    hdr['EXPTYPE']  = ('EXPOSURE', ' ')
    hdr['WAVELNTH'] = (10, 'center wavelength of bandpass filter')
    hdr['WAVEUNIT'] = ('micrometers', ' ')
    hdr['OBSERVAT'] = (observatory, ' ')
    hdr['PLACE']    = (place, ' ')
    hdr['LONGITUD'] = (longitude, ' ')
    hdr['LATITUDE'] = (latitude, ' ')
    hdr['ELEVATIO'] = (elevation, 'Altitude in meters')
    hdr['CUNIT1']   = ('arcsec', ' ')
    hdr['CRVAL1']   = (0.0, ' ')
    hdr['CRPIX1']   = (2.6, ' ')
    hdr['CUNIT2']   = ('arcsec', ' ')
    hdr['CRVAL2']   = (0.0, ' ')
    hdr['CRPIX2']   = (2.6, ' ')

    # Comments
    hdr.add_comment('Original Files')
    hdr.add_comment(fitsnames[0])
    hdr.add_comment(fitsnames[1])
    hdr.add_comment(fitsnames[2])
    hdr.add_comment(fitsnames[3])
    hdr.add_comment(comments[0])
    hdr.add_comment(comments[1])
    hdr.add_comment(comments[2])
    hdr.add_comment(comments[3])
    hdr.add_comment(comments[4])
    hdr.add_comment(comments[5])

    hdu = fits.PrimaryHDU(data=image, header=hdr)
    hdul = fits.HDUList([hdu])

    # Define filename and save .fits files
    image_fits_name = 'AR30T_Mosaic-2022-07-05.fits'
    hdul.writeto(image_fits_name, overwrite=True)
    print(f'\n {image_fits_name} file created with the Sun image') 
    
    return

def Create_Fits_Limbs(limbs):
    
    name = 'AR30T'
    instrument = 'FLIR A645sc'
    origin = 'OAFA/UNSJ - CRAAM/UPM'
    observatory = 'Obs. Astronomico Felix Aguilar'
    place = 'El Leoncito - San Juan, Argentina'
    longitude = -69.318889
    latitude = -31.802222
    elevation = 2430
    telescope = 'AR30T'
    comments = ['Level 0.0: uncalibrated image','COPYRIGHT. Grant of use.',
                'These data are property of Universidad Presbiteriana Mackenzie and Observatorio Astronomico Felix Aguilar.',
                'The Centro de Radio Astronomia e Astrofisica Mackenzie',
                'and Observatorio Astronomico Felix Aguilar are responsible for their distribution.',
                'Grant of use permission is given for Academic purposes only.',
                'Contact:guigue@craam.mackenzie.br; fernando.lopez@um.edu.ar' ]

    hdr = fits.PrimaryHDU()

    # General information
    hdr.header.append(('TELESCOP',name, ' '))
    hdr.header.append(('INSTRUME',instrument, ' '))
    hdr.header.append(('ORIGIN',origin, ' '))
    hdr.header.append(('DATE','2022-07-05', ' '))
    hdr.header.append(('DATE-OBS','2022-07-05T13:25:28', ' '))
    hdr.header.append(('WAVELNTH',10, 'center wavelength of bandpass filter'))
    hdr.header.append(('WAVEUNIT','micrometers', ' '))
    hdr.header.append(('OBSERVAT',observatory, ' '))
    hdr.header.append(('PLACE',place, ' '))
    hdr.header.append(('LONGITUD',longitude, ' '))
    hdr.header.append(('LATITUDE',latitude, ' '))
    hdr.header.append(('ELEVATIO',elevation, 'Altitude in meters'))

    # Comments
    hdr.header.append(('Comment','Fits File Created from Limbs extracted from a Mosaic',' '))
    hdr.header.append(('Comment',comments[1],' '))
    hdr.header.append(('Comment',comments[2],' '))
    hdr.header.append(('Comment',comments[3],' '))
    hdr.header.append(('Comment',comments[4],' '))
    hdr.header.append(('Comment',comments[5],' '))
    
    Nl = limbs['limbs'].shape[1]
    
    fits_cols = [
        fits.Column(name   = 'x',format = '1E', unit   = 'arcsec',bscale = 1,bzero  = 0,array  = limbs['x']),
        fits.Column(name   = 'limbs',format = str(Nl)+'E',unit   = 'norm',bscale = 1,bzero  = 0,array  = limbs['limbs']),
        fits.Column(name   = 'mean_limb',format = '1E',unit   = 'norm',bscale = 1,bzero  = 0,array  = limbs['mean_limb'])
        ]

    coldefs = fits.ColDefs(fits_cols)
    tbhdu   = fits.BinTableHDU.from_columns(coldefs)
    hduList = fits.HDUList([hdr,tbhdu])

    limbs_fits_name = 'AR30T_Limbs-2022-07-05_'+str(limbs['angle'])+'_deg.fits'
    hduList.writeto(limbs_fits_name, overwrite=True)
    print(f'\n {limbs_fits_name} file created with the Sun limbs')
    
    return
    
