#####################
# Exploration of noise issues in perfect NFW fits
#####################


import numpy as np
import fitrunner, nfwutils
import nfwmodeltools as tools
import fitmodel
import scipy.misc
import basicMassCon
import varcontainer
import pymc
import pymc_mymcmc_adapter as pma
import nfwmodel




######################



def createPerfectProfile(m200, c, zcluster, r_mpc, beta_s):


    rho_c_over_sigma_c = 1.5 * nfwutils.global_cosmology.angulardist(zcluster) * nfwutils.global_cosmology.beta([1e6], zcluster)[0] * nfwutils.global_cosmology.hubble2(zcluster) / nfwutils.global_cosmology.v_c**2


    r_scale = nfwutils.rscaleConstM(m200, c, zcluster, 200)

    nfw_shear_inf = tools.NFWShear(r_mpc, c, r_scale, rho_c_over_sigma_c)
    nfw_kappa_inf = tools.NFWKappa(r_mpc, c, r_scale, rho_c_over_sigma_c)
        
    g = beta_s*nfw_shear_inf / (1 - (beta_s*nfw_kappa_inf))

    return g



#########################

def createClusterSet(config, mtrues, zcluster, r_mpc_edges, beta_s, galdensity, shapenoise, concens = None):

    nclusters = len(mtrues)

    model = nfwfit.buildModel(config)
    profile_rmpcs = []
    shearprofiles = []
    shearerrprofiles = []


    arcmin_max = (r_mpc_edges[-1] / nfwutils.global_cosmology.angulardist(zcluster))*(180./np.pi)*60
    arcminarea = arcmin_max**2
    ngals = int(arcminarea * galdensity)


    if concens is None:

        try:
            mcrelation = model.massconRelation
        except AttributeError:
            mcrelation = nfwfit.buildObject('basicMassCon', 'Duffy', config)



    for i in range(nclusters):
        
        m200 = mtrues[i]

        if concens is None:
            c200 = mcrelation(m200*nfwutils.global_cosmology.h, zcluster, 200)
        else:
            c200 = concens[i]


        xs = np.random.uniform(-r_mpc_edges[-1], r_mpc_edges[-1], size=ngals)
        ys = np.random.uniform(-r_mpc_edges[-1], r_mpc_edges[-1], size=ngals)
        gal_r_mpcs = np.sqrt(xs**2 + ys**2)
        gtrue = createPerfectProfile(m200, c200, zcluster, gal_r_mpcs, beta_s)
        gobs = gtrue + shapenoise*np.random.standard_normal(size=ngals)
    
        radii = []
        shear = []
        shearerr = []
        for i in range(len(r_mpc_edges)-1):
            mintake = r_mpc_edges[i]
            maxtake = r_mpc_edges[i+1]
            selected = np.logical_and(gal_r_mpcs >= mintake,
                                      gal_r_mpcs < maxtake)

            ngal = len(gal_r_mpcs[selected])            
        
            if ngal == 0:
                continue
            
            radii.append(np.mean(gal_r_mpcs[selected]))
            shear.append(np.mean(gobs[selected]))
            shearerr.append(shapenoise/np.sqrt(ngal))

        profile_rmpcs.append(np.array(radii))
        shearprofiles.append(np.array(shear))
        shearerrprofiles.append(np.array(shearerr))


    return profile_rmpcs, shearprofiles, shearerrprofiles


##########################

def fitClusterSet(config, zcluster, r_mpcs, beta_s, shearprofiles, shearerrs):

    fitter = nfwfit.buildFitter(config)

    fitMasses = np.zeros(len(shearprofiles))
    fitErrs = np.zeros(len(shearprofiles))
    mask = np.ones(len(shearprofiles))

    for i in range(len(shearprofiles)):

        fitres = fitter.minChisqMethod(r_mpcs[i], shearprofiles[i], shearerrs[i], beta_s, beta_s**2, zcluster)
        if fitres is None:
            fitres = fitter.minChisqMethod(r_mpcs[i], shearprofiles[i], shearerrs[i], beta_s, beta_s**2, 
                                           zcluster, useSimplex=True)

        if fitres is None:
            mask[i] = 0
        else:

            

            fitMasses[i] = fitres[0]['m200']*fitter.model.massScale
#            fitMasses[i] = nfwutils.Mdelta(fitres[0]['rscale'], 4.0, zcluster, 200.)
            asymerrs = fitres[1]['m200']
            try:
                fitErrs[i] = fitter.model.massScale*(asymerrs[1] - asymerrs[0])/2.
            except TypeError:
                fitErrs[i] = asymerrs
#
#    mask = (mask == 1)

    return fitMasses, fitErrs, mask

#    return fitMasses, mask


#########################

def bootstrapMean(distro, nboots = 1000):

    means = np.zeros(nboots)
    for i in range(nboots):
        curboot = np.random.randint(0, len(distro), len(distro))
        means[i] = np.mean(distro[curboot])

    return means



    

#########################

def createFakeChains(config, nclusters, zcluster, r_mpc_edges, beta_s, galdensity, shapenoise, nsamples=1000, mass=10**15.2):

    mtrues = mass*np.ones(nclusters)

    r_mpcs, shearprofiles, shearerrs = createClusterSet(config, mtrues, zcluster, r_mpc_edges, beta_s, galdensity, shapenoise)

    fitter = nfwfit.buildFitter(config)


    chains = []

    for i in range(nclusters):

        mcmc_model = None
        for j in range(10):
            try:
                mcmc_model = fitter.model.makeMCMCModel(r_mpcs[i], shearprofiles[i], shearerrs[i], beta_s, beta_s**2, zcluster)
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
        options.nsamples = nsamples
        manager.model = mcmc_model
        
        runner = pma.MyMCMemRunner()
        runner.run(manager)
        runner.finalize(manager)

        chains.append(manager.chain['m200'][200:])

    return mtrues, chains


        



#########################

def runNoiseScaling(config, nclusters, zcluster, r_mpc_edges, beta_s, galdensities, shapenoises):

    mfitsets = []
    merrsets = []
    masksets = []

    mfitsets2 = []
    merrsets2 = []
    masksets2 = []

#    mtrues = 10**np.random.uniform(14., 15.5, size=nclusters)
    mtrues = 10**14.4*np.ones(nclusters)

    for i in range(len(galdensities)):

        r_mpcs, shearprofiles, shearerrs = createClusterSet(config, mtrues, zcluster, r_mpc_edges, beta_s, galdensities[i], shapenoises[i])

        fitMasses, fitErrs, mask  = fitClusterSet(config, zcluster, r_mpcs, beta_s, shearprofiles, shearerrs)
        mfitsets.append(fitMasses)
        merrsets.append(fitErrs)
        masksets.append(mask)


#        fitMasses2, fitErrs2, mask2 = fitClusterSet(config, zcluster, r_mpc, beta_s, shearprofiles, shearerr/10.)
#        mfitsets2.append(fitMasses2)
#        merrsets2.append(fitErrs2)
#        masksets2.append(mask2)
#    return mtrues, mfitsets, merrsets, masksets, mfitsets2, merrsets2, masksets2

#    return mtrues, mfitsets, merrsets, masksets
    return mtrues, mfitsets, merrsets, masksets

    
##########################


def calcNoiseBias(zcluster=0.5, concen=4.):

    m200s = np.arange(1, 50)
    r_mpc = np.arange(0.1, 3.0, 0.1)

    duffy = basicMassCon.Duffy(None)

    background = 1.0
    beta_s = nfwutils.global_cosmology.beta_s([background], zcluster)
    beta_s2 = beta_s**2
    beta_cor = beta_s2/beta_s

    rho_c_over_sigma_c = 1.5 * nfwutils.global_cosmology.angulardist(zcluster) * nfwutils.global_cosmology.beta([1e6], zcluster)[0] * nfwutils.global_cosmology.hubble2(zcluster) / nfwutils.global_cosmology.v_c**2

    def reducedShear(scaledm200):
        m200 = 1e14*scaledm200
        c = duffy(m200, zcluster)
        rs = nfwutils.rscaleConstM(m200, c, zcluster, 200)

        gamma = tools.NFWShear(r_mpc, c, rs, rho_c_over_sigma_c)
        kappa = tools.NFWKappa(r_mpc, c, rs, rho_c_over_sigma_c)

        g = beta_s*gamma/(1-beta_cor*kappa)

        return g

    shearprime = np.zeros((len(m200s), len(r_mpc)))
    shearprimeprime = np.zeros((len(m200s), len(r_mpc)))
    for i, m200 in enumerate(m200s):

        shearprime[i,:] = scipy.misc.derivative(reducedShear, m200, dx = .01, order=5)
        shearprimeprime[i,:] = scipy.misc.derivative(reducedShear, m200, dx = .01, n=2, order=7)
            
    return shearprime, shearprimeprime, -0.5*shearprimeprime/(shearprime**3)


#########################

class Likelihood(object):

    def __init__(self, config):

        self.config = config
        self.model = nfwmodel.NFW_Model()

    ###

    
    def bindData(self, r_mpc, g, gerr, beta_s, zcluster):

        self.r_mpc = r_mpc
        self.g = g
        self.gerr = gerr
        self.beta_s = beta_s
        self.beta_s2 = self.beta_s**2
        self.zcluster = zcluster

        self.rho_c = np.float64(nfwutils.global_cosmology.rho_crit(zcluster))
        self.rho_c_over_sigma_c = 1.5 * nfwutils.global_cosmology.angulardist(zcluster) * nfwutils.global_cosmology.beta([1e6], zcluster)[0] * nfwutils.global_cosmology.hubble2(zcluster) / nfwutils.global_cosmology.v_c**2


    ###

    def __call__(self, m200, c200 = None):

        if c200 is None:
            c200 = self.model.massconRelation(np.abs(m200*nfwutils.global_cosmology.h), self.zcluster, 200)
        return tools.shearprofile_like(m200,
                                       c200,
                                       self.r_mpc,
                                       self.g,
                                       self.gerr,
                                       self.beta_s*np.ones_like(self.r_mpc),
                                       self.beta_s2*np.ones_like(self.r_mpc),
                                       self.rho_c,
                                       self.rho_c_over_sigma_c,
                                       200.)


        

#############################


def calcMassUncert(config, r_mpc, beta_s, zcluster, shearprofiles, shearerr,     
                   testmasses = np.arange(-5.02e15, 5e15, 5e13)):

    likelihood = Likelihood(config)

    nclusters = len(shearprofiles)

    testconcens = np.linspace(1.1, 19.9, 100)
    CGrid, MGrid = np.meshgrid(testconcens, testmasses)
    logprobgrid = np.zeros_like(CGrid)
    probgrid = np.zeros_like(logprobgrid)
    margprobgrid = np.zeros_like(testmasses)

    massprobs = np.zeros((nclusters, len(testmasses)))

    masserrs = np.zeros(nclusters)

    for curcluster in range(nclusters):

        likelihood.bindData(r_mpc, shearprofiles[curcluster], shearerr, beta_s, zcluster)

        for i in range(logprobgrid.shape[0]):
            for j in range(logprobgrid.shape[1]):
                logprobgrid[i,j] = likelihood(MGrid[i,j], CGrid[i,j])

        probgrid[:] = np.exp(logprobgrid - np.max(logprobgrid))
        probgrid /= np.sum(probgrid)
        margprobgrid[:] = np.sum(probgrid, axis=1)
        
        massprobs[curcluster,:] = margprobgrid

        avemass = np.sum(margprobgrid*testmasses)
        masserr = np.sqrt(np.sum(margprobgrid*(testmasses - avemass)**2))
        masserrs[curcluster] = masserr


    return masserrs, massprobs

        
        

def testnoisebias(noise = 0.04, niters = 10000):

    r_mpc =  np.linspace(0.25, 3.0, 20)
    beta_s = 0.5
    zcluster = 0.5
    g = createPerfectProfile(1e15, 4.0, zcluster, r_mpc, beta_s)
    gerr = noise/np.sqrt(r_mpc)
    
#    m200scan = np.logspace(np.log10(1e13), np.log10(5e15), 200)
    m200scan = np.arange(-1e15 + 5e12, 5e15, 1e13)
    
    config = nfwfit.readConfiguration('testnoisebias.config')
    likelihood = Likelihood(config)

    likelihood.bindData(r_mpc, g, gerr, beta_s, zcluster)
    perfectpdf = np.array([likelihood(m200 = curm200) for curm200 in m200scan])

    maxs = np.zeros(niters)
    for i in range(niters):
        gwnoise = g + gerr*np.random.standard_normal(size=len(g))
        likelihood.bindData(r_mpc, gwnoise, gerr, beta_s, zcluster)
        logprob = np.array([likelihood(m200 = curm200) for curm200 in m200scan])
        maxs[i] = m200scan[logprob == np.max(logprob)]


    return r_mpc, g, gerr, m200scan, perfectpdf, maxs

    


    
    
