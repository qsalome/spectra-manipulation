########################################################################
##
## File:
##    plot_spec.py
##
## Description:
##
##    Plot the average spectra and the Gaussians.
##
## Reference:
##    None
##
## Usage:
##    python3 plot_spec.py
##
## Authors:
##    Quentin Salomé, FINCA, University of Turku, Finland
##
## Updated:
##    26-APR-2023 -- v10, Initial creation
##    28-APR-2022 -- v11, The spectrum are now plotted as step plots
##
## Todo: general functions for more flexibility
##       call parameters
##       read file with the list of sources (as parameter?)
##       put user parameters in an input file

import os
import importlib
import numpy as np
import astropy.units as u
from specutils import Spectrum1D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


#---------------------------------------------------------------------------------------
def gaussian(x,A,mu,sigma):
    return A*np.exp(-((x-mu)**2)/(2.*sigma**2))

#---------------------------------------------------------------------------------------
def freq2vel(nu,frest):
   nu  = np.array(nu, float)
   vel = (frest-nu)/nu*299792.458   #km/s
   return vel

#---------------------------------------------------------------------------------------
def vel2freq(vel,frest):
   vel = np.array(vel, float)
   nu  = frest/(1+vel/299792.458)
   return nu

#---------------------------------------------------------------------------------------


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


pdf2 = PdfPages(output+'Average_spectra.pdf')


for src in source_list:
   # Extract the relevant values from the Mini-Catalogue
   src_name = src[0]
   src_vel  = src[3]*u.km/u.s

   # Read file
   specfile  = input+src_name+'_av_spec'
   gaussfile = output+src_name+'_gaussians'

   if not os.path.exists(gaussfile):
      continue
   print(src_name)


   # Read the spectrum
   nu,Tsource = np.loadtxt(specfile,usecols=[0,1],delimiter=',',unpack=True)
   samp = np.where(np.isfinite(Tsource))   # Remove NaN at the end of the spectrum
   spec = Spectrum1D(spectral_axis=nu[samp]/1e6*u.MHz,flux=Tsource[samp]*u.K,
                     rest_value=6668.5192*u.MHz,velocity_convention='optical')


   # Read the Gaussian parameters
   A,mu,sigma = np.loadtxt(gaussfile,usecols=[1,2,3],ndmin=2,unpack=True)

   # Plot
   pdf = PdfPages(output+'%s_mean_spectrum.pdf'%(src_name))
   fig,ax1 = plt.subplots()
#   ax2 = ax1.twiny()

   # automatically update ylim of ax2 when ylim of ax1 changes.
   ax1.step(spec.velocity,spec.flux,where='mid',color='r',linewidth=1.,label='Mean spectrum')
   x1,x2 = ax1.get_xlim()
#   ax2.set_xlim(vel2freq(x1,6668.5192),vel2freq(x2,6668.5192))
#   ax2.figure.canvas.draw()
   ax1.set_xlim([src_vel.value-5,src_vel.value+5])

   ax1.set_title(src_name+' - Methanol')
   ax1.set_xlabel('v [ km/ s]')
   ax1.set_ylabel('T [ K ]')
#   ax2.set_ylabel('$\\nu$ [ MHz ]')

   chan = np.arange(len(spec.flux))
   sum  = np.zeros(len(spec.flux))

   for k in np.arange(len(A)):
       sum += gaussian(chan,A[k],mu[k],sigma[k])
       if k == 1:
           ax1.plot(spec.velocity,gaussian(chan,A[k],mu[k],sigma[k]),
                  'b--',linewidth=1.,label='Gaussian')
       else:
           ax1.plot(spec.velocity,gaussian(chan,A[k],mu[k],sigma[k]),
                  'b--',linewidth=1.)

   ax1.plot(spec.velocity,sum,'k--',linewidth=1.,label='Model')
   ax1.legend(loc = 1, numpoints = 1)
   leg = ax1.get_legend()
   ltext  = leg.get_texts()
   plt.setp(ltext, fontsize = 'small')

   # Save mean spectrum
   plt.savefig(pdf, format='pdf')
   plt.savefig(pdf2, format='pdf')
   plt.close()
   pdf.close()

pdf2.close()


