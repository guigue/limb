import datetime as dt
import numpy as np
from scipy import stats
from astropy.io import fits
from scipy import ndimage
from skimage.filters import roberts
from matplotlib import pyplot as plt
from sunpy import coordinates
from astropy import units as u
from astropy.visualization.mpl_normalize import simple_norm
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body

import pdb 
from CraamTools.fit import Circle
import CASLEO

# Global data
    
D20220725   = ['20220725_flt-000001-206_13_25_28_813_32.fits'  , 
               '20220725_flt-000001-206_13_25_28_813_10.fits'  , 
               '20220725_flt-000001-206_13_25_28_813_128.fits' , 
               '20220725_flt-000001-206_13_25_28_813_82.fits'  ] 

D20261006   = ['AR30T_20261006T200317_264_level00.fits' ,
               'AR30T_20261006T200518_883_level00.fits' ,
               'AR30T_20261006T200636_493_level00.fits' ,
               'AR30T_20261006T200805_535_level00.fits']

files_lists = {'20220725':D20220725,'20261006':D20261006}

comments    = ['Level 0.0: uncalibrated image','COPYRIGHT. Grant of use.'                  ,
               'These data are property of Universidade Presbiteriana Mackenzie (UPM) and' ,
               'Observatorio Astronomico Felix Aguilar (OAFA). The Centro de Radio'        ,
               'Astronomia e Astrofisica Mackenzie (CRAAM) and OAFA are responsible'       ,
               'for their distribution. Grant of use permission is given for Academic'     ,
               'purposes only.'                                                            ,
               'Contact: guigue@craam.mackenzie.br; fernando.lopez@um.edu.ar'              ]

# Information for the FITS files
name        = 'AR30T'
instrument  = 'FLIR A645sc'
origin      = 'OAFA/UNSJ - CRAAM/UPM'
observatory = 'Obs. Astronomico Felix Aguilar'
place       = 'El Leoncito - San Juan, Argentina'
longitude   = -69.318889
latitude    = -31.802222
elevation   = 2430
telescope   = 'AR30T'

class AR30TLimb(object):
    
    """
 A Python collection of scripts to create a mosaic image of the sun.
 I used the individual images obtained by scanning the sun 
 to create a flat field. Images were registered on 2022-07-25 at 16:25 UTC.

 I did not flat-field the images. Images were chosen to cover the entire
 solar disc.

 I normalized each image individually. However, the first one has an
 illumination issue that I decided not to eliminate. It is not clear what
 its origin is, since it is not observed in the other images. Unfortunately,
 all the images from this "corner" have the same illumination problem.

 Images are not co-aligned with the Sun's North Pole.

 Use sequence:
 >>> import limb
 >>> image,limbs = limb.FullSun(angle=10,**kwargs)
 The 'angle' parameter is the angle [deg] between successive slices
 of the solar disc. These slices are then averaged to get the 'mean limb'.

 Use **kwargs to chose between different normalization functions and add their own keyword arguments:
    'linear', 'sqrt', etc
 See https://docs.astropy.org/en/stable/api/astropy.visualization.simple_norm.html

 The final mean limb seems to be very reasonable, despite the illumination problem
 that creates a 'bump' in some slices.

 Results are shown in figures, and saved in FITS files. 

-----------------------------------------------------------------------

 Author: @guiguesp,
 Date:   2026-09-04, while I listen to the good music selection
         from Radioeins, "Nur für Erwachsene, natürlich!"

 Change Record:
         2026-10-15, normalization functions from astropy.visualization,
                     two different list fo files, and the Output Fits file
                     name as a keyword argument.
         2026-10-16, normalization function name as a comment in the FITS
         2026-10-17, 'limb' is now an object
    
    """

    def __init__(self,date='2026-10-06',slice_interval=10*u.deg,**kwargs):

        self.image_size  = 1280
        self.camera_size = [640,480]
        self.px_size     = 2.6
        self.rot_angle   = -15.774 * u.deg # [deg] Camera rotation angle

        self.disc       = np.zeros([self.image_size,self.image_size])
        self.limbs      = {}
        self.fitsnames  = files_lists[date[:4]+date[5:7]+date[8:]]
        self.slice_interval = slice_interval
        self.kwargs     = kwargs
        self.dateobs    = date 

        return

    def getTime(self,f):

        if 'T_START' in f[0].header:
            tstart = f[0].header['T_START']
        else:
            tstart = f[0].header['DATE-OBS']
            
        return dt.datetime(int(tstart[:4]),
                           int(tstart[5:7]),
                           int(tstart[8:10]),
                           int(tstart[11:13]),
                           int(tstart[14:16]),
                           int(tstart[17:19]))
    
    def Get_Sun_Coordinates(self):
        
        #
        #  Get_Sun_Coordinates
        #
        #     This method obtains the Sun Coordinates both in (Ra,Dec) and (Alt, Az)
        #     While astropy may use very accurate calculations and corrections from
        #     IERS, for some reason it cannot download the data. So, I decided to
        #     work off-line with a lesser precision, although enoug for us.
        #
        #
        #  @guiguesp - 2020-04-05
        #            - 2020-05-30 : Now using sunpy 1.1
        #_____________________________________________________________________________

        t = Time(self.tmean)

        with solar_system_ephemeris.set('builtin'):
            sun_radec=get_body('sun',t,self.cg)

        self.L0          = coordinates.sun.L0(self.tmean)
        self.B0          = coordinates.sun.B0(self.tmean)
        self.P           = coordinates.sun.P(self.tmean)
        self.R_Sun       = coordinates.sun.angular_radius(self.tmean)
        self.Carrington  = coordinates.sun.carrington_rotation_number(self.tmean)
        self.radec       = sun_radec
        self.north       = coordinates.sun.orientation(self.cg,t)
        return 

    #----------------------------------------------------------------------------------------------
    
    def ComposeMosaic(self):
        
        """
        This is the main code. It opens the files in a predetermined order (do 
        not dare to change the order!), finds the center of the disc, and
        copies the shifted image to the final full sun image. The result is saved
        as FITS file. Then extract slices at each angle=angle and average them. 
        The result is saved in a FITS file.
        Images are provided for the full (Mosaic) Sun and the averaged normalized 
        limb.
    
        """
        tobs = []
        for name in self.fitsnames:
            f = fits.open(name)
            tobs.append(self.getTime(f))
            cx,cy = self.FindCenter(self.Normalize(f))
            dx = int(self.image_size//2 - cx)
            dy = int(self.image_size//2 - cy)
            self.disc[dy:dy+self.camera_size[1],dx:dx+self.camera_size[0]] = np.copy(f[0].data)

        self.tmean = min(tobs) + (max(tobs)-min(tobs)) / 2
    
        # Rotate the Sun, North is up
        # First, correct the camera rotation angle respect to the telescope axis
        self.disc  = ndimage.rotate(self.disc,angle=self.rot_angle,reshape=False,order=1)
    
        # Now use sunpy to get the North
        self.cg    = CASLEO.Observatory_Coordinates()
        self.Get_Sun_Coordinates()
        self.disc  = ndimage.rotate(self.disc,angle=self.north,reshape=False,order=1)
        self.getLimbs()
        return

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.disc,extent=[-640*2.6,640*2.6,-640*2.6,640*2.6],origin='lower',cmap='gray',vmin=0.3,vmax=1.1)
        ax.set_xlabel('X [arcsec]')
        ax.set_ylabel('Y [arcsec]')
        ax.set_title('AR30T Mosaic')
    
        plt.savefig('AR30T_Mosaic.pdf', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format='pdf',
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    metadata=None)    

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

        return

    def FindCenter(self,f):
    
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
    
    def Normalize(self,f):

        norm = simple_norm(f[0].data,**self.kwargs)
        f[0].data = norm(f[0].data)
        return f


    def getLimbs(self):

    ##################################################################
    #
    # The "equatorial" limb is extracted from every rotated image slice.
    # 
        N = int(180/self.slice_interval.value)
        l = np.zeros([self.image_size,N])
        ml = np.zeros(self.image_size)
    
        for i in np.arange(N):
            angle = self.slice_interval * i 
            r = ndimage.rotate(self.disc,angle=angle,reshape=False,order=1)
            l[:,i] = r[int(self.image_size//2),:]
            ml = (ml+l[:,i])

        x = np.linspace(-self.image_size/2,self.image_size/2,self.image_size) * self.px_size

        self.limbs = {'x':x,'clv':l,'mean_clv':ml/N}

        return

    def Create_Fits_Image(self):

        hdr = fits.Header()
        # General information
        hdr['SIMPLE']   = True
        hdr['BITPIX']   = -32
        hdr['NAXIS']    = 2
        hdr['NAXIS1']   = (self.image_size, ' ')
        hdr['NAXIS2']   = (self.image_size, ' ')
        hdr['EXTEND']   = (True, ' ')
        hdr['TELESCOP'] = (name, ' ')
        hdr['INSTRUME'] = (instrument, ' ')
        hdr['ORIGIN']   = (origin, ' ')
        hdr['DATE']     = (self.dateobs, ' ')
        hdr['DATE-OBS'] = (self.tmean.isoformat(), ' ')
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
        hdr['CROTA']    = (self.rot_angle.value, '[deg] Rot angle, CCW from Y axis')
    
        # Comments
        hdr.add_comment('Original Files:')
        for fname in self.fitsnames:
            hdr.add_comment(fname)
        hdr.add_comment('Normalization function and parameters')
        for k in self.kwargs:
            hdr.add_comment(f'{k} : {self.kwargs[k]}')
        hdr.add_comment(' ')        
        hdr.add_comment('Sun North Rotation Angle: {0:6.3f} [deg]'.format(self.north))
        hdr.add_comment(' ')
        for c in comments:
            hdr.add_comment(c)

        hdu = fits.PrimaryHDU(data=self.disc, header=hdr)
        hdul = fits.HDUList([hdu])

        # Define filename and save .fits files
        image_fits_name = 'AR30T_Mosaic-'+self.dateobs+'.fits'
        hdul.writeto(image_fits_name, overwrite=True)
        print(f'\n {image_fits_name} file created with the Sun image') 
        
        return

    def Create_Fits_Limbs(self):

        hdr = fits.PrimaryHDU()

        # General information
        hdr.header.append(('TELESCOP',name, ' '))
        hdr.header.append(('INSTRUME',instrument, ' '))
        hdr.header.append(('ORIGIN',origin, ' '))
        hdr.header.append(('DATE',self.dateobs, ' '))
        hdr.header.append(('DATE-OBS',self.tmean.isoformat(), ' '))
        hdr.header.append(('WAVELNTH',10, 'center wavelength of bandpass filter'))
        hdr.header.append(('WAVEUNIT','micrometers', ' '))
        hdr.header.append(('OBSERVAT',observatory, ' '))
        hdr.header.append(('PLACE',place, ' '))
        hdr.header.append(('LONGITUD',longitude, ' '))
        hdr.header.append(('LATITUDE',latitude, ' '))
        hdr.header.append(('ELEVATIO',elevation, 'Altitude in meters'))

        # Comments
        hdr.header.append(('History','Fits File Created from Limbs extracted from a Mosaic',' '))
        hdr.header.append(('History','Normalization function and parameters:'))
        for k in self.kwargs:
            hdr.header.append(('History','{0:s} : {1:s}'.format(k,kwargs[k])))
        hdr.header.append(' ')
        for c in comments:
            hdr.header.append(('Comment',c,' '))
    
        Nl = self.limbs['clv'].shape[1]
    
        fits_cols = [
            fits.Column(name   = 'x',format = '1E', unit   = 'arcsec',bscale = 1,bzero  = 0,array  = limbs['x']),
            fits.Column(name   = 'limbs',format = str(Nl)+'E',unit   = 'norm',bscale = 1,bzero  = 0,array  = limbs['limbs']),
            fits.Column(name   = 'mean_limb',format = '1E',unit   = 'norm',bscale = 1,bzero  = 0,array  = limbs['mean_limb'])
        ]

        coldefs = fits.ColDefs(fits_cols)
        tbhdu   = fits.BinTableHDU.from_columns(coldefs)
        hduList = fits.HDUList([hdr,tbhdu])
        
        limbs_fits_name = 'AR30T_Limbs-'+rootname+'_'+str(limbs['angle'].value)+'_deg.fits'
        hduList.writeto(limbs_fits_name, overwrite=True)
        print(f'\n {limbs_fits_name} file created with the Sun limbs')
    
        return
    
