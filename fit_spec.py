#!/usr/bin/env python3
########################################################################
##
## File:
##    fit_spec.py
##
## Description:
##
##    Fit the average spectra with a sum of Gaussian.
##
## Reference:
##    Clumpix/ROHSA
##
## Usage:
##    python3 fit_spec.py
##
## Author:
##    Quentin Salomé, FINCA, University of Turku, Finland
##
## Updated:
##    26-APR-2023 -- v10, Initial creation
##    27-APR-2023 -- v11, Add some columns in the output file
##
## Todo: general functions for more flexibility
##       call parameters
##       read file with the list of sources (as parameter?)
##       put user parameters in an input file

import os
import importlib
import numpy as np
import lmfit
import astropy.units as u
from specutils import Spectrum1D

import mod_tools
importlib.reload(mod_tools)


# Input and output folders
input  = ''
output = ''

# Source list
source_list=[
 #  Name        RA              Dec           VLSR   S     Int   # Comments
 #              HMS             DMS           km/s   Jy    s
### DETECTED MASERS
 [ "G9.621",   "18:06:14.67", "-20:31:32.4", -10.93, 5196,   120 ], # Det. strong, v.low Dec.
 [ "G12.681",  "18:13:54.75", "-18:01:46.6",  45.58,  350,   465 ], # Mixed detection with G12.909
 [ "G12.909",  "18:14:39.53", "-17:52:00.0",  43.70,  245,   515 ], # Mixed detection with G12.681
 [ "G23.010",  "18:34:40.27", "-09:00:38.3",  59.77,  483,   245 ], # Det.
 [ "G24.790",  "18:36:12.56", "-07:12:11.30",113.40,  115, 30000 ], # Det. v.weak
 [ "G25.710",  "18:38:03.2",  "-06:24:14.9",  80.46,  607,   300 ], # Det.
 [ "G29.955",  "18:46:03.7",  "-02:39:22.2",  96.0,   238,  7000 ], # Det. v.weak
 [ "G30.703",  "18:47:36.82", "-02:00:53.80", 88.20,  131, 18000 ], # Det. v.weak
 [ "G31.281",  "18:48:12.43", "-01:26:30.10",110.40,  131, 28000 ], # Det. v.weak
 [ "G32.045",  "18:49:36.56", "-0:45:45.90",  92.9,   163, 24000 ], # Det. v.weak
 [ "G33.641",  "18:53:32.560", "0:31:39.30",  59.6,   155, 86400 ], # Det. v.weak
 [ "G35.197",  "18:58:13.05",  "1:40:35.7",   28.5,   174,  9000 ], # Det. v.weak
 [ "G37.427",  "18:54:13.8",   "04:41:32.0",  24.08,  375,   265 ], # Det.
 [ "G49.488",  "19:23:43.9",   "14:30:31.0",  41.47,  979,   135 ], # Det. ** multisource
 [ "G69.540",  "20:10:09.1",   "31:31:36.0",  14.6,    79,  9000 ], # Det. v.weak
 [ "G81.877",  "20:38:36.8",   "42:37:59.5", -11.66,  543,   310 ], # Det. nr.Cyg
 [ "G82.308",  "20:40:16.7",   "42:56:28.60", 10.3,    58, 20000 ], # Det. extremely weak
 [ "G85.410",  "20:54:13.70",  "44:54:06.8", -28.53,  105,  9000 ], # Det. extremely weak
 [ "G109.871", "22:56:18.1",   "62:01:49.3", -13.13, 1420,   115 ], # Det.
 [ "G111.542", "23:13:45.3",   "61:28:10.7", -67.76,  346,   370 ], # Det. weak
 [ "G133.949", "02:27:04.2",   "61:52:25.4", -48.17, 3880,    50 ], # Det.
# [ "G174.201", "05:30:48.015", "33:47:54.61",  1.48,   91,  days ], # Det. extremely weak
 [ "G188.946", "06:08:53.7",   "21:38:29.7",  23.75,  553,   155 ], # Det. nr.Tau
 [ "G192.600", "06:12:53.990", "17:59:23.70",  4.6,    68, 40000 ], # Det. v.weak
 [ "G213.705", "06:07:47.85",  "-6:22:55.2",  10.8,   278, 20000 ], # Det. extremely weak
 [ "G232.620", "07:32:09.79", "-16:58:12.4",  22.9,   178, 15000 ], # Det. weak


### DETECTION TESTS?
 [ "G108.184", "22:28:51.4",   "64:13:41.3", -11.0,    42,  300 ], # 108.184+5.519
 [ "G136.837", "02:49:29.8",   "60:47:29.5", -45.00,  208, 1800 ], # Suvi/Lena ???
 [ "G183.348", "05:51:11.1",   "25:46:16.5",  -4.89,   13,  900 ], # Suvi/Lena / Faint
]


# user parameters
lim_gauss = 10           # maximum number of Gaussians
lim_sigma = [1, 30.]   # limits for the sigma (in channels)
dvel      = 7            # velocity (in km/s) range containing the emission vlsr+/-vel
spec_reso = 1e-3*u.MHz
c = 299792458.0*u.m/u.s
f_rest  = 6668.5192*u.MHz
wl_rest = c/f_rest.to(u.Hz)
wl_rest = wl_rest.to(u.mm)


for src in source_list:
   # Extract the relevant values from the Mini-Catalogue
   src_name = src[0]
   src_vel  = src[3]*u.km/u.s

   # Read file
   infile  = input+src_name+'_av_spec'
   outfile = output+src_name+'_gaussians'

   if not os.path.exists(infile):
      #print('There is no data for %s'%(src_name))
      continue
   if not os.path.exists(output): os.makedirs(output)
   print(src_name)

   # Read the spectrum and calculate the rms
   nu,Tsource = np.loadtxt(infile,usecols=[0,1],delimiter=',',unpack=True)
   samp = np.where(np.isfinite(Tsource))   # Remove NaN at the end of the spectrum
   spec = Spectrum1D(spectral_axis=nu[samp]/1e6*u.MHz,flux=Tsource[samp]*u.K,
                     rest_value=6668.5192*u.MHz,velocity_convention='optical')

   rms  = mod_tools.compute_rms(spec,src_vel.value,dvel)
   print('rms=',rms)

   # Initial guest for the first Gaussian
   A     = np.nanmax(Tsource)
   mu    = np.where(Tsource==A)[0][0]
   sigma = 10.

   samp = np.where((spec.velocity>=src_vel-dvel*u.km/u.s)&
                   (spec.velocity<=src_vel+dvel*u.km/u.s))
   pars = lmfit.Parameters()
   pars.add('g1_A',    value=A,    min=3*rms,     max=np.nanmax(Tsource)) 
   pars.add('g1_mu',   value=mu,   min=samp[0][0],  max=samp[0][-1]) 
   pars.add('g1_sigma',value=sigma,min=lim_sigma[0],max=lim_sigma[1])

   # Fit of the average spectrum
   guess = mod_tools.first_guest(pars,spec,src_vel.value,dvel,lim_sigma,lim_gauss,'leastsq')
   print('The best fit (reduced chi2=%4.2f) predicts %i gaussians.'
            %(guess.redchi,len(guess.params)/3))

   # Save the Gaussian parameters
   parvals = guess.params.valuesdict()
   Ngauss  = int(len(parvals)/3)

   out = open(outfile,'w')
   out.write('# Gaussian      A        mu         sig       nu0       FWHM       v0        FWHM\n')
   out.write('#              [K]                           [MHz]      [kHz]    [km/s]      [m/s]\n')

   for k in np.arange(Ngauss)+1:
       string  = '     %2i   '%(k)
       string += '   %6.3f'   %(parvals['g%i_A'%(k)])
       string += '   %9.4f'   %(parvals['g%i_mu'%(k)])
       string += '   %6.3f'   %(parvals['g%i_sigma'%(k)])

       nu0  = spec.spectral_axis[int(round(parvals['g%i_mu'%(k)]))]
       FWHM = 2.354*parvals['g%i_sigma'%(k)]*spec_reso*1e3 # kHz
       string += '   %9.4f'   %(nu0.value)
       string += '   %6.3f'   %(FWHM.value)

       v0   = spec.velocity[int(round(parvals['g%i_mu'%(k)]))]
       FWHM = 2.354*parvals['g%i_sigma'%(k)]*spec_reso*wl_rest # km/s
       FWHM = FWHM.to(u.m/u.s)
       string += '   % 7.4f'  %(v0.value)
       string += '   %7.2f'   %(FWHM.value)
       out.write(string+'\n')

   out.close()


