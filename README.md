# spectra-manipulation

If part of the repository is used for a scientific publication,
a mention in the acknowledgement and the addition of the github link
would be appreciated.

## In short

fit_spec.py -- this code reads a spectrum in ascii format and fits a sum
               of Gaussian. The model is iterative by adding a Gaussian and
               comparing the reduced chi-square. The process stops when adding
               a Gaussian increased the reduced chi-square.

plot_spec.py -- this code simply plots the spectrum with the fitted model

mod_tools.py -- a collection of functions that are used to fit the model.

