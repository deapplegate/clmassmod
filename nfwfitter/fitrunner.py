#!/usr/bin/env python
#######################
# Runs a model fit to simulated shear data.
# Galaxies are binned into an average shear profile before fitting to NFW model.
# Options to explore radial fit range, mass-concentration relation, and binning scheme in fits.
########################

import cPickle, sys, os
import numpy as np
import pymc
import scipy.integrate

import nfwutils
import varcontainer
import pymc_mymcmc_adapter as pma
import simutils



#######################




#####




class MCMCFitter(object):

    def configure(self, config):

        self.model = config['massmodel']
        self.deltas = [200, 500, 2500]
        self.nsamples = 30000
        if 'nsamples' in config:
            self.nsamples = config['nsamples']

        


    def __call__(self, profile):

        chains = {}

        for delta in self.deltas:

            mcmc_model = None
            for i in range(20):
                try:
                    mcmc_model = self.model.makeMCMCModel(profile, delta = delta)
                    break
                except pymc.ZeroProbability:
                    pass
            if mcmc_model is None:
                raise pymc.ZeroProbability

            manager = varcontainer.VarContainer()
            options = varcontainer.VarContainer()
            manager.options = options

            options.singlecore = True
            options.adapt_every = 100
            options.adapt_after = 100
            options.nsamples = self.nsamples
            manager.model = mcmc_model

            runner = pma.MyMCMemRunner()
            runner.run(manager)
            runner.finalize(manager)

            reducedchain = dict(cdelta = np.hstack(manager.chain['cdelta'][5000::2]).astype(np.float32),
                                mdelta = np.hstack(manager.chain['mdelta'][5000::2]).astype(np.float32),
                                likelihood = np.hstack(manager.chain['likelihood'][5000::2]).astype(np.float32))

            chains[delta] = reducedchain


        return chains


##########



#######
        
class BadPDFException(Exception): pass

class PDFScanner(object):
    '''Assumes a mass-con relation, to create a 1D problem.'''

    def configure(self, config):

        self.model = config['massmodel']
        self.deltas = [200, 500, 2500]


        self.masses = np.arange(-1.005e15, 6e15, 1e13)
        if 'scanpdf_minmass' in config:
            self.masses = np.arange(config['scanpdf_minmass'], config['scanpdf_maxmass'], config['scanpdf_massstep'])



    def __call__(self, profile):


        self.model.setData(profile)


        pdfs = {}

        masses = self.masses   #assumed to be mdelta == 200. 

        for delta in self.deltas:

            deltamasses = np.zeros(len(masses))

            logprob = np.zeros(len(masses))


            for i in range(len(masses)):

                m200 = masses[i]


                deltamasses[i], logprob[i] = self.model.likelihood(m200, targetdelta=delta)





            pdf = np.exp(logprob - np.max(logprob))
            pdf = pdf/scipy.integrate.trapz(pdf, deltamasses)

            if np.any(np.logical_not(np.isfinite(pdf))):
                raise BadPDFException

            pdfs[delta] = (deltamasses, pdf)


        return pdfs

    #######

    
    
    



########################

def savefit(bootstrap_vals, outputname):

    with open(outputname, 'wb') as output:

        print '!!!!!!!!!!!!!!!!!!!!!!!', outputname

        cPickle.dump(bootstrap_vals, output, -1)


########################

def runFit(catalogname, configname, outputname):

    config, simreader = preloadFit(configname)

    runFit_Preloaded(simreader, catalogname, config, outputname)

##########################

def preloadFit(configname):

    config = simutils.readConfiguration(configname)
    simreader = config['simreader']

    nfwutils.global_cosmology.set_cosmology(simreader.getCosmology())

    return config, simreader

###########################

def runFit_Preloaded(simreader, catalogname, config, outputname):

    sim = simreader.load(catalogname)

    profilebuilder = config['profilebuilder']
    fitter = config['fitter']

    profile = profilebuilder(sim)

    fitvals = fitter(profile)

    savefit(fitvals, outputname)

############################


class FailedCreationException(Exception): pass


if __name__ == '__main__':


    catname = sys.argv[1]
    configname = sys.argv[2]
    outname = sys.argv[3]
    

    runFit(catname, configname, outname)

    if not os.path.exists(outname):
        raise FailedCreationException


