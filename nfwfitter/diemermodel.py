import numpy as np
import colossusMassCon as cmc
import colossus.halo.profile_dk14 as dk14prof
import nfwutils
import massmodel

class Diemer_Model(massmodel.MassModel):


    def convertMass(self, mass, concen, sourcedelta, targetdelta):
        #mass & concen are defined at the same delta

        density_profile = dk14prof.getDK14ProfileWithOuterTerms(M = mass, c = concen, z = self.zcluster, mdef = '{:d}c'.format(sourcedelta))

        return density_profile.MDelta(self.zcluster, '{:d}c'.format(targetdelta))

    ########
    

    def likelihood(self, mass, concen, mdelta=200., cdelta=200.):

        assert(mdelta == cdelta)


        modelg = self.__call__(self.profile.r_mpc, mass, concen, mdelta = mdelta, cdelta = cdelta)

        delta_over_sigmna = (modelg - self.profile.ghat)/self.profile.sigma_ghat

        logProb = np.sum(-0.5*delta_over_sigma**2 - np.log(np.sqrt(2*np.pi)) - np.log(self.profile.sigma_ghat))

        

    #############################
            


    def __call__(self, x, mass, concen, mdelta = 200., cdelta=200.):

        assert(mdelta == cdelta)

        if mass == 0.:
            return np.zeros_like(x)

        isNegative=mass < 0
        if isNegative:
            mass = np.abs(mass)


        cmc.matchCosmo()
        
        
        density_profile = dk14prof.getDK14ProfileWithOuterTerms(M = mass, c = concen, z = self.zcluster, mdef = '{:d}c'.format(delta))
        surfacedensity_func, deltaSigma_func = calcLensingTerms(density_profile, np.max(r_kpch))
        surfacedensity = surfacedensity_func(r_kpch)
        deltaSigma = deltaSigma_func(r_kpch)

        curcosmo = nfwutils.global_cosmology
        Dl = curcosmo.angulardist(zcluster)
        beta_inf = curcosmo.beta([1e6], zcluster)
        sigma_crit = (curcosmo.v_c**2/(4*np.pi*curcosmo.G))/(Dl*beta_inf)  #units are M_dot / Mpc^2
        convert_units = 1./(curcosmo.h*1e6)
        converted_sigma_crit = sigma_crit * convert_units
       
        gamma_t_inf = deltaSigma/converted_sigma_crit
        kappa_inf = surfacedensity/converted_sigma_crit



        if isNegative:
            gamma_t_inf = -gamma_t_inf
        
        g = self.beta_s*gamma_t_inf / (1 - ((self.beta_s2/self.beta_s)*kappa_inf) )

        return g


