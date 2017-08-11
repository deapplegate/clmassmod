import numpy as np
import nfwmodeltools as tools
import nfwutils
import massmodel

class NFW_Model(massmodel.MassModel):


    def convertMass(self, mass, concen, sourcedelta, targetdelta):

        assert(sourcedelta == 200.)
        
        rdelta = (3*abs(mass)/(4*sourcedelta*np.pi*self.rho_c))**(1./3.)
        rscale = rdelta / concen

        newmass = nfwutils.Mdelta(rscale, concen, self.zcluster, targetdelta)

        return newmass

    ###############


    def likelihood(self, m200, c200):


        return tools.shearprofile_like(m200, c200,
                                       self.profile.r_mpc,
                                       self.profile.ghat,
                                       self.profile.sigma_ghat,
                                       self.beta_s,
                                       self.beta_s2,
                                       self.rho_c,
                                       self.rho_c_over_sigma_c,
                                       200.)

        

    #############################
            


    def __call__(self, x, mass, concen, mdelta=200., cdelta=200.):

        assert(cdelta == 200. or mdelta == cdelta)

        if m200 == 0.:
            return np.zeros_like(x)

        isNegative=m200 < 0
        if isNegative:
            m200 = np.abs(m200)

        if mdelta == cdelta:
            rdelta = (3*abs(mass)/(4*mdelta*np.pi*self.rho_c))**(1./3.)
            r_scale = rdelta / concen
        else:  #cdelta == 200.
            r_scale = nfwutils.rscaleConstM(mass, concen, self.zcluster, delta)
    
        
        nfw_shear_inf = tools.NFWShear(x, concen, r_scale, self.rho_c_over_sigma_c, cdelta)
        nfw_kappa_inf = tools.NFWKappa(x, concen, r_scale, self.rho_c_over_sigma_c, cdelta)

        if isNegative:
            nfw_shear_inf = -nfw_shear_inf
        
        g = self.beta_s*nfw_shear_inf / (1 - ((self.beta_s2/self.beta_s)*nfw_kappa_inf) )

        return g


###########################################################

