#####
# Base class for shear profile models
#####

import numpy as np
import pymc

import simutils
import nfwutils

#####



class MassModel(object):

    def __init__(self):

        self.massScale = 1e14
        self.c200_low = 0.1
        self.c200_high = 30.
        

    def configure(self, config):

        if 'massprior' in config and config['massprior'] == 'log':
            self.massprior = 'log'
            self.m200_low = 1e10
            self.m200_high = 1e17
        else:
            self.massprior = 'linear'
            self.m200_low = -1e16
            self.m200_high = 1e16



    def paramLimits(self):

        return {'m200' : (self.m200_low/self.massScale,self.m200_high/self.massScale),
                'c200' : (self.c200_low, self.c200_high)}

    def guess(self):

        guess = [10**(np.random.uniform(14, 15.5)),
                 np.random.uniform(1., 20.)]

        guess[0] = guess[0] / self.massScale

        return guess


    def setData(self, profile):

        self.profile = profile
        
        #handle cases where shear signal has been rescaled to a different cluster redshift
        if profile.zlens is None:
            self.zlens = profile.zcluster
        else:
            self.zlens = profile.zlens
            
        self.beta_s = profile.beta_s
        self.beta_s2 = profile.beta_s2
        self.zcluster = profile.zcluster
        self.rho_c = nfwutils.global_cosmology.rho_crit(self.zcluster)
        #note the mixed usage of zlens and zcluster. Lensing properties were rescaled. Mass related properties (here H2, from rho_crit) were not.
        self.rho_c_over_sigma_c = 1.5 * nfwutils.global_cosmology.angulardist(self.zlens) * nfwutils.global_cosmology.beta([1e6], self.zlens)[0] * nfwutils.global_cosmology.hubble2(self.zcluster) / nfwutils.global_cosmology.v_c**2



    def makeMCMCModel(self, profile, delta = 200):

        self.setData(profile)

        parts = {}

        if self.massprior == 'linear':
            parts['scaledmdelta'] = pymc.Uniform('scaledmdelta', self.m200_low/self.massScale, self.m200_high/self.massScale)
            
            @pymc.deterministic(trace=True)
            def mdelta(scaledmdelta = parts['scaledmdelta']):
                return self.massScale*scaledmdelta
            parts['mdelta'] = mdelta

        else:
            parts['logMdelta'] = pymc.Uniform('logMdelta', np.log(self.m200_low), np.log(self.m200_high))
            
            @pymc.deterministic(trace=True)
            def mdelta(logMdelta = parts['logMdelta']):

                return np.exp(logMdelta)
            parts['mdelta'] = mdelta

        parts['cdelta'] = pymc.Uniform('cdelta', self.c200_low, self.c200_high)


        @pymc.observed
        def data(value = 0.,
                 mdelta = parts['mdelta'],
                 cdelta = parts['cdelta']):

            try:
            
                logp= self.likelihood(mdelta, cdelta, mdelta=delta, cdelta=delta)

                return logp

            except (ValueError, ZeroDivisionError):
                
                raise pymc.ZeroProbability



        parts['data'] = data

        return pymc.Model(parts)


    #################################



class MC_Model(object):

    def __init__(self):
        self.massmodel = None
        self.massconRelation = None

    def __getattr__(self, attr):

        try:
            return getattr(self.massmodel, attr)
        except AttributeError:
            if self.massmodel is None:
                raise simutils.NotConfiguredException()
            raise
        

    def configure(self, config):

        self.massmodel = config['base_massmodel']
        self.massconRelation = config['massconRelation']

    def guess(self):

        guess = [10**(np.random.uniform(14, 15.5))]

        guess[0] = guess[0] / self.massmodel.massScale

        return guess


    def paramLimits(self):

        limits = self.massmodel.paramLimits()

        return {'m200' : limits['m200']}

    
    #############################

    def convertMass(self, m200, targetdelta):
        
        c200 = self.massconRelation(np.abs(m200)*nfwutils.global_cosmology.h, 
                                    self.massmodel.zcluster)


        mdelta = self.massmodel.convertMass(m200, c200, 200., targetdelta)

        if m200 < 0:
            mdelta = -mdelta

        return mdelta


    #############################

    def likelihood(self, m200, targetdelta):

        c200 = self.massconRelation(np.abs(m200)*nfwutils.global_cosmology.h, 
                                    self.massmodel.zcluster)

        mdelta = self.convertMass(m200, targetdelta)
        

        return mdelta, self.massmodel.likelihood(m200, c200)


    #############################



    def __call__(self, x, m200):


        c200 = self.massconRelation(np.abs(m200)*nfwutils.global_cosmology.h, 
                                    self.massmodel.zcluster)

        return self.massmodel(x, m200, c200)

###############################

