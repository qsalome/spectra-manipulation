########################################################################
##
## File:
##    mod_tools.py
##
## Description:
##
##    Package containing functions to fit gaussians.
##
## Reference:
##    Clumpix/ROHSA
##
## Usage:
##    import as package in a script
##
## Authors:
##    Antoine Marchal, Institut d'Astrophysique Spatiale, France
##    Quentin Salomé, FINCA, University of Turku, Finland
##
## Updated:
##    13-APR-2017 -- v10, Original version by Antoine Marchal
##    10-JUN-2022 -- v20, Adapted by Quentin Salomé
##    26-APR-2023 -- v21, Change the function compute_rms()

import numpy as np
import lmfit


def compute_rms(spectrum,v0,dvel):
    """!
    Compute the standard deviation between v0-dvel and v0+dvel
    
    @param spectrum: Spectrum1D : Spectrum to process
    @param v0: float : Central velocity of the emission
    @param dvel: float : Define the range of velocity containing the emission
    
    @return standard deviation
    """
    if(np.nansum(spectrum.flux)==0): return float('nan')
    else:
        samp = np.where((spectrum.velocity.value<v0-dvel)|(spectrum.velocity.value>v0+dvel))
        return np.nanstd(spectrum.flux[samp].value)


def gaussian(x,A,mu,sigma):
    """!
    Gaussian function
    
    @param x: 1D array : velocity in channel unit  
    @param A: Float : amplitude
    @param mu: Float : center
    @param sigma: Float : dispersion

    @return gaussian(x)
    """
    return A*np.exp(-((x-mu)**2)/(2.*sigma**2))


def first_guest(pars,spectrum,v0,dvel,lim_sigma,lim,method):
    """!
    Fit the initial specturm to pass it to the hierarchical descent
    
    @param pars: lmfit method : 1 set of gaussian parameters
    @param spectrum: 1D array : spectrum / brightness Temperature
    @param v0: float : Central velocity of the emission
    @param dvel: float : Define the range of velocity containing the emission
    @param lim_sigma: List : limits of the range where sigma is fitted
    @param lim: Int : max number of Gaussian fitted
    @param method: String : minimization method

    @return lmfit obj
    """
    y = spectrum.flux.value
    rms = compute_rms(spectrum,v0,dvel)
    x = np.arange(len(y))
    err = np.ones(len(y))*rms

    n_gauss = len(pars)/3

    global_fit = minimize(y,pars,rms,method)

    if global_fit.redchi > 1.:
        redchi2 = 99.
        saveredchi2 = global_fit.redchi
        while ((redchi2>0.90) & (len(pars)/3<lim)):
            new_params = add_gaussian(y,pars,lim_sigma,rms)
            fit = minimize(y,new_params,rms,method)
            redchi2 = fit.redchi
            if ((redchi2<saveredchi2)&(redchi2>0.98)):
                saveredchi2 = redchi2
                save = fit
            pars = fit.params

    try:
        return save
    except:
        return global_fit


def add_gaussian(y,params,lim_sigma,rms):
    """!
    Add a new gaussian
    
    @param y: 1D array : spectrum
    @param params: lmfit method : set of previous parameters
    @param lim_sigma: List : limits of the range where sigma is fitted
    
    @return list of params with a new gaussian
    """
    n = int(len(params)/3)
    
    residu = residual(params,np.arange(len(y)),y)
    pars = lmfit.Parameters()
    for i in np.arange(n)+1:
        new_gauss = lmfit.Model(gaussian,prefix='g%i_'%(i))
    
        pars.update(new_gauss.make_params())
        
        A = params['g%i_A'%(i)].value
        mu = params['g%i_mu'%(i)].value
        sigma = params['g%i_sigma'%(i)].value
        
        pars.add('g%i_A'%(i),    value=A,    min=3*rms,            max=np.max(y))
        pars.add('g%i_mu'%(i),   value=mu,   min=params['g1_mu'].min,max=params['g1_mu'].max)
        pars.add('g%i_sigma'%(i),value=sigma,min=lim_sigma[0],       max=lim_sigma[1])
                        
    new_gauss = lmfit.Model(gaussian, prefix='g%i_'%(n+1))
    pars.update(new_gauss.make_params())

    p = np.argsort(np.random.randn(n))[0]+1

    loc = np.where(residu == np.min(residu))[0][0]
    A = y[loc]
    mu = loc
    sigma = params['g%i_sigma'%(p)].value

    pars['g%i_A'%(n+1)].set(A,        min=0.,max=np.max(y))
    pars['g%i_mu'%(n+1)].set(mu,      min=0.,max=len(y))
    pars['g%i_sigma'%(n+1)].set(sigma,min=lim_sigma[0],max=lim_sigma[1])
    return pars


def residual(pars,x,data=None,eps=None):
    """!
    Compute model, residual with and without errors
    
    @param pars: lmfit method : list of Gaussian
    @param x: 1D array : velocity in channel unit
    @param data: 1D array : spectrum / Brightness Temperature
    @param eps: 1D array : errors
    
    @return model if data is None \n
    model - data if eps is None \n
    (model - data) / eps else
    """
    parvals = pars.valuesdict()

    model = 0

    n = len(parvals) / 3

    for i in np.arange(n) + 1:
        model += gaussian(x,parvals['g%i_A'%(i)],parvals['g%i_mu'%(i)],parvals['g%i_sigma'%(i)])
        
    if data is None:
        return model
    if eps is None:
        return (model-data)
    return (model-data)/eps


def minimize(y,params,rms,method):
    """!
    Minimize the spectrum with a list of Gaussian
    
    @param y: 1D array : spectrum / Brightness Temperature
    @param params: lmfit method : list of Gaussian
    @param rms: Float : rms of spectrum to fit
    @param method: String : minimization method
    
    @return lmfit obj
    """
    x = np.arange(len(y))
    err = np.ones(len(y))*rms
    n = len(params)/3
    
    fitter = lmfit.Minimizer(residual,params,fcn_args=(x,y,err))
    try:
        fit = fitter.minimize(method=method)
    except Exception as mes: 
        print("Something wrong with fit: ", mes)
        pass
    return fit


