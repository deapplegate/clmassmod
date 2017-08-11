#######################
# Fit a model which is lognorm intrinsic scatter and linear norm noise
########################



import glob, cPickle, os, shutil, tempfile, pkg_resources
import numpy as np
import pymc
import consolidate_fits
import deconvolvedlognormtools as dlntools
import varcontainer
import pymc_mymcmc_adapter as pma
import scipy.stats
import nfwutils, fitrunner


########################

class BadPDFException(Exception): pass


#######################

def MCMCReader(inputfile, halo, delta, truth, thin=1, cprior = None):


    with open(inputfile, 'rb') as input:
        masschains = cPickle.load(input)


    msamples = np.array(masschains[delta]['mdelta'][200::thin])*nfwutils.global_cosmology.h

    if cprior is not None:
        cfilter = masschains[delta]['cdelta'][200::thin] < cprior
        msamples = msamples[cfilter]


    halo['mass_samples'] = msamples
    
    return halo

###



def PDFReader(pdffile, halo, delta, truth, model=None):

    
    with open(pdffile, 'rb') as input:
        data = cPickle.load(input)

    if isinstance(data, dict):
        #dealing with outputfiles redefined in Jan 2017
        pdfs = data
        masses, pdf = pdfs[delta]        

    else:
        #files defined before Jan 2017
        masses, pdfs = data

        if type(pdfs) != dict:
            if delta != 200:
                print 'Skipping ', filebase
                raise BadPDFException(filebase)
            pdfs = {200:pdfs}  #historical reasons. If it isn't a pdf, it was computed as 200.

        if delta in pdfs:
            pdf = pdfs[delta]
        else:
            pdf = nfwfit.convertLikelihoodScan(model, delta, masses, pdfs[200], truth['redshift'])




    if np.any(np.logical_not(np.isfinite(pdf))):
        raise BadPDFException(filebase)

    halo['masses'] = masses*nfwutils.global_cosmology.h
    
    halo['pdf'] = pdf/nfwutils.global_cosmology.h

    return halo

###




def loadPosteriors(pdfdir, simtype, simreader, delta, selector,
                   reader = MCMCReader, **kwds):

    mass = 'm%d' % delta

    nfwutils.global_cosmology.set_cosmology(simreader.getCosmology())

    idpattern = consolidate_fits.idpatterns[simtype]

    answers = cPickle.load(pkg_resources.resource_stream('nfwfitter', 'data/{0}_answers.pkl'.format(simtype)))

        
    halos = []

    

    for pdffile in glob.glob('%s/*.out' % pdfdir):

        filebase = os.path.basename(pdffile)

        match = idpattern.match(filebase)
        

        try:
            haloid = int(match.group(1))
        except AttributeError as e:
            print filebase
            raise e
        except ValueError:
            haloid = match.group(1)

        try:
            truth = answers[haloid]
        except KeyError:
            print 'Failure at {0}'.format(output)
            raise

        if not selector(truth):
            continue

        halo =  dict(id = haloid,
                     true_mass = truth['m%d' % delta])
                      

        try:

            halos.append(reader(pdffile, halo, delta, truth, **kwds))

        except BadPDFException, e:
            print e
            continue

    print 'Num Halos: ', len(halos)
                         
    return halos




#######################

#def posteriorPredictivePDFs(logmu, logsigma, mtrues, config, nmlsamples=5, masses=np.arange(-5e14, 5e15, 1e13), beta = 0.28):
#
#    if config.binspacing == 'linear':
#        binedges = np.linspace(config.minradii, config.maxradii, config.nbins+1)
#    else:
#        binedges = np.logspace(np.log10(config.minradii), np.log10(config.maxradii), config.nbins+1)
#    
#
#    fitter = nfwfit.buildFitter(config)
#    fitter.model.setData(beta, beta**2, config.targetz)
#
#    if masses is None:
#        doMaxlike=True
#        maxlikes = np.zeros((nclusters, nmlsamples))
#    else:
#        doMaxlike = False
#        nmasses = len(masses)
#        pdfs = np.zeros((nclusters, nmlsamples, nmasses))
#
#    nclusters = len(mtrues)
#
#    for i in range(nclusters):
#        mls = np.exp(logmu + np.log(mtrues[i]) + np.exp(logsigma)*np.random.standard_normal(nmlsamples))
#        profile_rmpcs, shearprofiles, shearerrprofiles = nfwnoise.createClusterSet(config,
#                                                                                   mls,
#                                                                                   config.targetz,
#                                                                                   binedges,
#                                                                                   beta,
#                                                                                   config.nperarcmin,
#                                                                                   config.shapenoise,
#                                                                                   concens = 4.*np.ones(nmlsamples))
#
#        for j in range(nmlsamples):
#
#            if doMaxlike is True:
#                maxlikes[i,j] = modelfitter.minChisqMethod(profile_rmpcs[j], 
#                                                           shearprofiles[j], 
#                                                           shearerrprofiles[j], 
#                                                           beta, beta**2, config.targetz)[0]*fitter.model.massScale
#
#            else:
#
#
#                modelfitter = fitmodel.FitModel(profile_rmpcs[j], 
#                                                shearprofiles[j], 
#                                                shearerrprofiles[j], 
#                                                fitter.model,
#                                                guess = fitter.model.guess())
#
#
#                for k, mass in enumerate(masses):
#                
#                    pdfs[i,j,k] = modelfitter.statfunc(modelfitter.ydata,
#                                                       modelfitter.yerr,
#                                                       modelfitter.model(modelfitter.xdata,
#                                                                         mass / fitter.model.massScale))
#
#
#
#    if doMaxlike:
#        return maxlikes
#    else:
#        pdfs = np.exp(pdfs - np.max(pdfs))
#        return pdfs
#    
#                
######################
#
#####################

def evalGaussModel(xs, chain, ngauss, index):

    assert((xs >= 0).all())

    pis = []
    mus = []
    taus = []
    for i in range(ngauss):
        mus.append(chain['xmus_{}'.format(i)][index])
        taus.append(np.exp(chain['logxsigma_{}'.format(i)][index]))
        if i < ngauss - 1:
            pis.append(chain['piprior_{}'.format(i)][index])

    pis.append(1. - np.sum(np.array(pis)))

    pdf = np.zeros_like(xs)
    for pi, mu, tau in zip(pis, mus, taus):
        zerobound = -mu/tau
        pdf += pi*np.exp(-0.5*(xs - mu)**2/tau**2)/((np.sqrt(2*np.pi)*tau)*(1. - scipy.stats.norm.cdf(zerobound)))

    return pdf

###

def buildGaussMixture1DModel(halos, ngauss, modeltype='ratio'):

    parts = {}

    ### PDF handling

    massnorm = 1e15

    masses = halos[0]['masses']
    nmasses = len(masses)





    nclusters = len(halos)
    delta_masses = np.zeros((nclusters, nmasses-1))
    delta_mls = np.zeros((nclusters, nmasses))
    pdfs = np.zeros((nclusters, nmasses))

    #also need to collect some statistics, to init mixture model
    pdfmeans = np.zeros(nclusters)
    pdfwidths = np.zeros(nclusters)

    for i in range(nclusters):

        if modeltype == 'additive':
            delta_masses[i,:] = (masses[1:] - masses[:-1])/massnorm
            delta_mls[i,:] = (masses - halos[i]['true_mass'])/massnorm
            pdfs[i,:] = halos[i]['pdf']*massnorm   #preserve unitarity under integration
        elif modeltype == 'ratio':
            delta_masses[i,:] = (masses[1:] - masses[:-1])/halos[i]['true_mass']
            delta_mls[i,:] = masses / halos[i]['true_mass']
            pdfs[i,:] = halos[i]['pdf']*halos[i]['true_mass']
        
        pdfmeans[i] = scipy.integrate.trapz(delta_mls[i,:]*pdfs[i,:], delta_mls[i,:])
        pdfwidths[i] = np.sqrt(scipy.integrate.trapz(pdfs[i,:]*(delta_mls[i,:] - pdfmeans[i])**2, delta_mls[i,:]))
        
    datacenter = np.mean(pdfmeans)
    dataspread = np.std(pdfmeans)
    datatypvar = np.mean(pdfwidths)
    dataminsamp = np.min(delta_masses)

    print datacenter, dataspread, datatypvar, dataminsamp

    #### Mixture model priors

    
    piprior = pymc.Dirichlet('piprior', np.ones(ngauss))
    parts['piprior'] = piprior

    
    mu0 = pymc.Uninformative('mu0', datacenter + np.random.uniform(-5*dataspread, 5*dataspread))
    parts['mu0'] = mu0

# kelly07 xvars prior. 
#    w2 = pymc.Uniform('w2', 0.1/dataspread**2., 100*max(1./dataspread**2, 1./datatypvar**2))
#    print w2.parents
#    parts['w2'] = w2
#
#
#    xvars = pymc.InverseGamma('xvars', 0.5, 0.5*w2, size=ngauss+1)  #dropping the 1/2 factor on w2, because I don't think it matters

    logxsigma = pymc.Uniform('logxsigma', np.log(2*dataminsamp), np.log(5*dataspread), size = ngauss+1)
    parts['logxsigma'] = logxsigma

    @pymc.deterministic(trace=False)
    def xvars(logxsigma = logxsigma):
        return np.exp(logxsigma)**2
    parts['xvars'] = xvars

    @pymc.deterministic(trace=False)
    def tauU2(xvars = xvars):
        return 1./xvars[-1]
    parts['tauU2'] = tauU2

    xmus = pymc.Normal('xmus', mu0, tauU2, size = ngauss)
    parts['xmus'] = xmus


    

    @pymc.observed
    def data(value = 0., 
             delta_mls = delta_mls, 
             delta_masses = delta_masses,
             pdfs = pdfs,
             piprior = piprior,
             xmus = xmus,
             xvars = xvars):

        #complete pi
        pis = pymc.extend_dirichlet(piprior)

#        print pis


#        #enforce identiability by ranking means
#        for i in range(xmus.shape[0]-1):
#            if (xmus[i] >= xmus[i+1:]).any():
#                raise pymc.ZeroProbability
#

        return dlntools.pdfGaussMix1D(delta_mls = delta_mls,
                                      delta_masses = delta_masses,
                                      pdfs = pdfs,
                                      pis = pis,
                                      mus = xmus,
                                      tau2 = xvars[:-1])
    parts['data'] = data

    return parts





#####################


def buildPDFModel(halos, sigmapriors = None):

    parts = {}

    parts['logmu'] = pymc.Uniform('logmu', -1., 1.)


    if sigmapriors is None:
        parts['logsigma'] = pymc.Uniform('logsigma', np.log(0.05), np.log(10))
    else:
        parts['logsigma'] = pymc.Normal('logsigma', sigmapriors[0], 1./sigmapriors[1]**2)

    @pymc.deterministic(trace=True)
    def sigma(logsigma = parts['logsigma']):
        return np.exp(logsigma)
    
    parts['sigma'] = sigma

    masses = halos[0]['masses']
    posmasses = masses[masses > 0]
    nmasses = len(posmasses)
    deltamasses = posmasses[1:] - posmasses[:-1]


    nclusters = len(halos)

    delta_logmls = np.zeros((nclusters, nmasses))
    pdfs = np.zeros((nclusters, nmasses))

    for i in range(nclusters):

        delta_logmls[i,:] = np.log(posmasses) - np.log(halos[i]['true_mass'])
        rawpdf = halos[i]['pdf'][masses>0]
        pdfs[i,:] = rawpdf / scipy.integrate.trapz(rawpdf, posmasses)

    

    @pymc.observed
    def data(value = 0., ml_ints = posmasses, deltamasses = deltamasses,
             delta_logmls = delta_logmls, pdfs = pdfs,
             logmu = parts['logmu'], sigma = parts['sigma']):

        return dlntools.pdfloglinearlike(ml_ints = ml_ints,
                                          deltamasses = deltamasses,
                                          delta_logmls = delta_logmls,
                                          pdfs = pdfs,
                                          logmu = logmu,
                                          sigma = sigma)
    parts['data'] = data

    return parts


        
                                                                                   
        
        
    

########################

def buildMCMCModel(halos, maxsamples = 2000):

    massnorm = 1e15

    parts = {}

    parts['logmu'] = pymc.Uniform('logmu', -1., 1.)
    parts['logsigma'] = pymc.Uniform('logsigma', np.log(1e-2), np.log(10))

    @pymc.deterministic(trace=True)
    def sigma(logsigma = parts['logsigma']):
        return np.exp(logsigma)
    
    parts['sigma'] = sigma

    nclusters = len(halos)
    ml_ints = np.zeros((nclusters, maxsamples))
    delta_logmls = np.zeros((nclusters, maxsamples))
    ngoodsamples = np.zeros(nclusters, dtype=np.int)
    for i in range(nclusters):
        mass_samples = halos[i]['mass_samples']
        positive_samples = mass_samples[mass_samples > 0]
        navailablesamples = len(positive_samples)
        takesamples = min(navailablesamples, maxsamples)
        if navailablesamples < 25:
            print 'Need more samples: ', halos[i]['id'], navailablesamples

        ml_ints[i,:takesamples] = np.random.permutation(positive_samples)[:takesamples]
        delta_logmls[i,:takesamples] = np.log(ml_ints[i,:takesamples]) - np.log(halos[i]['true_mass'])
        ngoodsamples[i] = takesamples

    

    @pymc.observed
    def data(value = 0., ml_ints = ml_ints/massnorm, delta_logmls = delta_logmls, ngoodsamples = ngoodsamples,
             logmu = parts['logmu'], sigma = parts['sigma']):

        return dlntools.mcmcloglinearlike(ml_ints = ml_ints,
                                          delta_logmls = delta_logmls,
                                          ngoodsamples = ngoodsamples,
                                          logmu = logmu,
                                          sigma = sigma)
    parts['data'] = data

    return parts


    

########################


def buildModel(mlens, merr, mtrue, nsamples = 150, noisefloor = False):

    massnorm = 1e15

    if noisefloor is True:
        merr = np.maximum(np.sqrt(merr**2 + (0.2*mtrue)**2), 0.1*np.abs(mlens))

    parts = {}

    parts['logmu'] = pymc.Uniform('logmu', -1., 1.)
    parts['logsigma'] = pymc.Uniform('logsigma', np.log(1e-4), np.log(10))

    @pymc.deterministic(trace=True)
    def sigma(logsigma = parts['logsigma']):
        return np.exp(logsigma)
    
    parts['sigma'] = sigma

    nclusters = len(mlens)
    ml_ints = np.zeros((nclusters, nsamples))
    delta_logmls = np.zeros((nclusters, nsamples))
    for i in range(nclusters):
        a, b = (0. - mlens[i]) /merr[i], (1e17 - mlens[i]) / merr[i]
        ml_ints[i,:] = mlens[i] + merr[i]*scipy.stats.truncnorm.rvs(a,b,size=nsamples)
        delta_logmls[i,:] = np.log(ml_ints[i,:]) - np.log(mtrue[i])

    

    @pymc.observed
    def data(value = 0., ml_ints = ml_ints/massnorm, delta_logmls = delta_logmls, 
             logmu = parts['logmu'], sigma = parts['sigma']):

        return dlntools.loglinearlike(ml_ints = ml_ints,
                                      delta_logmls = delta_logmls,
                                      logmu = logmu,
                                      sigma = sigma)
    parts['data'] = data

    return parts


#######################################

def buildOutlierModel(mlens, merr, mtrue, nsamples = 150, outlierfactor=100):

    massnorm = 1e15

    parts = {}

    parts['logmu'] = pymc.Uniform('logmu', -1., 1.)
    parts['logsigma'] = pymc.Uniform('logsigma', np.log(1e-4), np.log(10))

    @pymc.deterministic(trace=True)
    def sigma(logsigma = parts['logsigma']):
        return np.exp(logsigma)
    
    parts['sigma'] = sigma

    parts['fracoutliers'] = pymc.Uniform('fracoutliers', 0., 1.)

    nclusters = len(mlens)
    ml_ints = np.zeros((nclusters, nsamples))
    delta_logmls = np.zeros((nclusters, nsamples))
    outlier_ml_ints = np.zeros((nclusters, nsamples))
    outlier_delta_logmls = np.zeros((nclusters, nsamples))
    for i in range(nclusters):
        a, b = (0. - mlens[i]) /merr[i], (1e17 - mlens[i]) / merr[i]
        ml_ints[i,:] = mlens[i] + merr[i]*scipy.stats.truncnorm.rvs(a,b,size=nsamples)
        delta_logmls[i,:] = np.log(ml_ints[i,:]) - np.log(mtrue[i])

        a, b = (0. - mlens[i]) /outlierfactor*merr[i], (1e17 - mlens[i]) / outlierfactor*merr[i]
        outlier_ml_ints[i,:] = mlens[i] + outlierfactor*merr[i]*scipy.stats.truncnorm.rvs(a,b,size=nsamples)
        outlier_delta_logmls[i,:] = np.log(outlier_ml_ints[i,:]) - np.log(mtrue[i])

    

    @pymc.observed
    def data(value = 0., ml_ints = ml_ints/massnorm, delta_logmls = delta_logmls,
             outlier_ml_ints = outlier_ml_ints, outlier_delta_logmls = outlier_delta_logmls,
             logmu = parts['logmu'], sigma = parts['sigma'],
             fracoutliers = parts['fracoutliers']):

        return dlntools.outlierloglinearlike(ml_ints,
                                             delta_logmls,
                                             outlier_ml_ints,
                                             outlier_delta_logmls,
                                             logmu,
                                             sigma,
                                             fracoutliers)
    parts['data'] = data

    return parts


    

    

#######################################


def runFit(parts):

    N = pymc.NormApprox(parts)
    N.fit()
    
    return (N.mu[N.logmu], np.sqrt(N.C[N.logmu])), (N.mu[N.logsigma], np.sqrt(N.C[N.logsigma]))


#######################################

def sample(parts, outputfile, samples, adaptevery = 100, adaptafter = 100, singlecore = False):

    tempoutputdir = tempfile.mkdtemp()
    outputdir, outputbase = os.path.split(outputfile)

    options = varcontainer.VarContainer()
    options.outputFile = '%s/%s' % (tempoutputdir, outputbase)
    options.singlecore = singlecore
    options.nsamples = samples
    options.adapt_every = adaptevery
    options.adapt_after = adaptafter
    options.restore=True

    manager = varcontainer.VarContainer()
    manager.options = options
    manager.model = pymc.Model(parts)
    
    assert(np.isfinite(manager.model.logp))

    runner = pma.MyMCRunner()

    try:
        runner.run(manager)
        runner.finalize(manager)
    finally:
        outputfiles = glob.glob('%s/*' % tempoutputdir)
        for curoutputfile in outputfiles:
            curoutputbase = os.path.basename(curoutputfile)
            destination_output = '%s/%s' % (outputdir, curoutputbase)

            if os.path.exists(destination_output):
                os.remove(destination_output)
            shutil.copyfile(curoutputfile, destination_output)
        shutil.rmtree(tempoutputdir)




##############################

def memsample(model, samples, adaptevery = 100, adaptafter = 100, outputFile = None):

    options = varcontainer.VarContainer()
    options.singlecore = True
    options.nsamples = samples
    options.adapt_every = adaptevery
    options.adapt_after = adaptafter
    if outputFile:
        options.outputFile = outputFile

    manager = varcontainer.VarContainer()
    manager.options = options
    manager.model = model

    runner = pma.MyMCMemRunner()

    runner.run(manager)

    if outputFile:
        runner.dump(manager)

    runner.finalize(manager)

    return manager.chain


#################

