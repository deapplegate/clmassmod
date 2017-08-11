##
# Implement a spike version of the pymc3 based model I want to write
# . in this case, fitting a 4 parameter shear profile model to stacked cluster data
###

import cPickle
import numpy as np
import pymc3

import scipy.interpolate

import theano
import theano.tensor as T
import theano.compile.ops as ops



import colossus.halo.profile_nfw as nfwprofile
import colossus.halo.profile_dk14 as dk14profile
import colossus.halo.profile_outer as outerprofile
import colossus.cosmology.cosmology as ccosmo
import colossus.defaults as cDefaults


import nfwfitter.nfwutils as nfwutils
import nfwfitter.colossusMassCon as cmc

#####

mxxlcosmo = nfwutils.Cosmology(omega_m = 0.25, omega_l = 0.75, h=0.73)
nfwutils.global_cosmology.set_cosmology(mxxlcosmo)
cmc.matchCosmo()


#####
scaling = 1e15  #m_dot/h

@ops.as_op(itypes=[T.dscalar,T.dscalar,T.dscalar,
                   T.dscalar,T.dvector,T.dvector,T.dvector],
           otypes=[T.dvector])
def t_gpredictor(scaledm200, c200, bias, zcluster, r_kpch, beta_s, beta_s2):
    return gpredictor(float(scaledm200), float(c200), float(bias), float(zcluster), r_kpch, beta_s, beta_s2)


twohalodata = np.loadtxt('twohalo.npy')
log_twohalo_interp = scipy.interpolate.InterpolatedUnivariateSpline(twohalodata[:,0], twohalodata[:,1])

def calcLensingTerms(density_profile, rmax, max_r_integrate = cDefaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE, bias=0.):

    assert(rmax < cDefaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTERPOLATE)

    log_min_r = np.log(cDefaults.HALO_PROFILE_DELTA_SIGMA_MIN_R_INTERPOLATE)
    log_max_r = np.log(np.max(rmax))
    table_log_r = np.arange(log_min_r, log_max_r + 0.01, 0.01)
    table_r = np.exp(table_log_r)
    sigma = density_profile.surfaceDensity(table_r) +   bias*np.exp(log_twohalo_interp(table_r))
    table_log_Sigma = np.log(sigma)

    log_surface_density_interp = scipy.interpolate.InterpolatedUnivariateSpline(table_log_r, table_log_Sigma)

    kappa_enc_integrand = np.exp(table_log_Sigma)*(table_r**2)

    kappa_enc_traprule_element = (table_log_r[1:] - table_log_r[:-1])*(kappa_enc_integrand[1:] + kappa_enc_integrand[:-1])

    kappa_enc_traprule = 0.5*np.cumsum(kappa_enc_traprule_element)

    log_kappa_enc_interp = scipy.interpolate.InterpolatedUnivariateSpline(table_log_r[1:], np.log(kappa_enc_traprule))

    surface_density = lambda r: np.exp(log_surface_density_interp(np.log(r)))

    deltaSigma = lambda r: np.exp(log_kappa_enc_interp(np.log(r)))*2./r**2 - surface_density(r)

    return surface_density, deltaSigma

###

def calcLensing(density_profile, r_kpch, zcluster, max_r_integrate = cDefaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE, bias=0.):

    surfacedensity_func, deltaSigma_func = calcLensingTerms(density_profile, np.max(r_kpch), max_r_integrate = max_r_integrate, bias=bias)
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

    return gamma_t_inf, kappa_inf


###


def gpredictor(scaledm200, c200, bias, zcluster, r_kpch, beta_s, beta_s2):
    m200 = scaledm200*scaling

    zcluster = zcluster
        
#    profile = nfwprofile.NFWProfile(M=m200, mdef='200c', z=zcluster, c = c200)
    profile = dk14profile.DK14Profile(M=m200, mdef='200c', z=zcluster, c = c200)


    gamma_t_inf, kappa_inf = calcLensing(profile, r_kpch, zcluster, max_r_integrate=1000*ccosmo.getCurrent().R_xi[-1], bias=bias)

    g = beta_s*gamma_t_inf / (1 - ((beta_s2/beta_s)*kappa_inf) )

    return g

    ###


#####


def buildSample(profileset):




    with pymc3.Model() as model:

        #prep data
        ave_ghat = np.mean(profileset['ghats'], axis=0)
        sig_ghat = np.std(profileset['ghats'], axis=0, ddof=1)/np.sqrt(ave_ghat.shape[0])

        
        r_kpch = theano.shared(profileset['r_mpc']*1000*nfwutils.global_cosmology.h)
        zcluster = T.sharedvar.scalar_constructor(profileset['zcluster'])
        beta_s = theano.shared(profileset['beta_s'])
        beta_s2 = theano.shared(profileset['beta_s2'])

    


        #priors
        scaledm200 = pymc3.HalfNormal('scaledm200', sd=10., testval=1.)
        c200 = pymc3.HalfNormal('c200', sd=10., testval=4.)
        bias = pymc3.HalfNormal('bias', sd=10., testval=1.)

        
        g_pred = t_gpredictor(scaledm200, c200, bias,
                            zcluster, r_kpch, beta_s, beta_s2)
        
        g_obs = pymc3.Normal('g_obs', mu=g_pred, sd=sig_ghat, observed=ave_ghat)

    return model

def sample(model, nsamples=2000):

    with model:
        start = pymc3.find_MAP(model=model)
        print start
        step = pymc3.Slice(vars=model.free_RVs)
        trace = pymc3.sample(nsamples, step=step, start=start)


    return trace

########

def run():

    with open('../output/stackedprofile.pkl', 'rb') as input:
        profileset = cPickle.load(input)

    model, trace = buildSample(profileset, 50)

    with open('../output/stackedprofile.trace.pkl', 'wb') as output:
        cPickle.dump(trace, output)

########

if __name__ == '__main__':
    run()
