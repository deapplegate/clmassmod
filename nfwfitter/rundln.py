#!/usr/bin/env python
###########################
# Run dln fit for one mass bin of one noise sim
###########################

import sys, re
import random
import deconvolvedlognorm as dln
import nfwutils
import numpy as np
import pymc
import cPickle
import simutils


def defineBins(masses, minclusters = 56, maxclusters = 600, binstep=0.2):

    if len(masses) < minclusters:
        return [np.min(masses), np.max(masses)]

    nmasses = len(masses)
    smass = np.sort(masses)[::-1]
    l10smass = np.log10(smass)
    indexes = np.arange(nmasses)
    edges = [np.log10(smass[0] + 1e12)]
    
    startindex = 0

    while(startindex < (nmasses-minclusters)):


        minbinsize = l10smass[startindex] - l10smass[startindex+minclusters]

        maxstopindex = np.min([startindex+maxclusters, nmasses-1])
        maxbinsize = l10smass[startindex] - l10smass[maxstopindex]
        ismax = False

        if minbinsize > binstep:
            print 'Min'
            stopval = l10smass[startindex+minclusters]
        elif maxbinsize < binstep:
            print 'Max'
            ismax = True
            stopval = l10smass[maxstopindex]
            if maxstopindex == (nmasses -1):
                break
        else:
            print 'Interval'
            stopval = l10smass[startindex]-binstep

        stopindex = indexes[l10smass < stopval][0]
        if (nmasses - stopindex) < minclusters or (ismax and (nmasses - stopindex < maxclusters)) :
            print 'End Adjust'
            equalstop = int((nmasses + startindex)/2.)
            stopval = l10smass[equalstop]
            stopindex = indexes[l10smass < stopval][0]

        edges.append(stopval)
        startindex = stopindex



    edges.append(l10smass[-1])

    return 10**np.array(edges[::-1])


class NoDataException(Exception): pass

###

class Selector(object):

    def __init__(self, mass, masslow, masshigh, otherselect = None):
        self.mass = mass
        self.masslow = masslow
        self.masshigh = masshigh
        self.otherselect = otherselect
    def __call__(self, truth):
        inbin = truth[self.mass] >= self.masslow and truth[self.mass] < self.masshigh
        if self.otherselect is not None:
            select = inbin and self.otherselect(truth)
        else:
            select = inbin
        return select

###

def convertIntoSelectors(edges, mass):
    selectors = []
    for i in range(len(edges)-1):
        selectors.append(Selector(mass, edges[i],edges[i+1]))
    return selectors

###

def takeAllMasses(simtype, delta):

    if simtype == 'bk11snap124':
        if delta == 500:

            massedges = np.array([  1.5e+14, 6.5e+14])
            selectors = convertIntoSelectors(massedges, 'm500')
        elif delta == 200:

            massedges = np.array([  3.0e+14,  9.9e+14])
            selectors = convertIntoSelectors(massedges, 'm200')
        else:
            raise NoDataException

    elif simtype == 'bk11snap141':
        if delta == 500:

            massedges = np.array([2.0e+14, 8.8e+14])
            selectors = convertIntoSelectors(massedges, 'm500')
        elif delta == 200:

            massedges = np.array([4.5e+14, 1.6e+15])
            selectors = convertIntoSelectors(massedges, 'm200')
        else:
            raise NoDataException

    elif simtype == 'mxxlsnap41':
        if delta == 200:

            massedges = np.array([3.5e+14, 1.4e+15])
            selectors = convertIntoSelectors(massedges, 'm200')
        elif delta == 500:

            massedges = np.array([3.0e+14, 1.2e+15])
            selectors = convertIntoSelectors(massedges, 'm500')
        elif delta == 2500:

            massedges = np.array([1.6e+14, 3.9e+14])
            selectors = convertIntoSelectors(massedges, 'm2500')

    elif simtype == 'mxxlsnap54':
        if delta == 200:

            massedges = np.array([7.4e14, 4e15])
            selectors = convertIntoSelectors(massedges, 'm200')

        elif delta == 500:

            massedges = np.array([6.2e14, 2.7e15])
            selectors = convertIntoSelectors(massedges, 'm500')

        elif delta == 2500:
            #1 bin
            massedges = np.array([3.15e14, 1.2e15])
            selectors = convertIntoSelectors(massedges, 'm2500')


    return selectors


    

###

def defineMassEdges(simtype, delta):

    match = re.match('analytic_(.+)', simtype)
    if match is not None:
        simtype = match.group(1)
        

    if simtype == 'bk11snap124':
        if delta == 500:
            #3 bins
            massedges = np.array([  1.5e+14,   2.2e+14,   3.5e+14,6.5e+14])
            selectors = convertIntoSelectors(massedges, 'm500')
        elif delta == 200:
            #3 bins
            massedges = np.array([  3.0e+14,   3.5e+14,   5.0e+14, 9.9e+14])
            selectors = convertIntoSelectors(massedges, 'm200')
        else:
            raise NoDataException

    elif simtype == 'bk11snap141':
        if delta == 500:
            #4 bins
            massedges = np.array([2.0e+14, 2.3e+14, 2.9e+14, 4.4e+14, 8.8e+14])
            selectors = convertIntoSelectors(massedges, 'm500')
        elif delta == 200:
            #2 bins
            massedges = np.array([4.5e+14,   6.7e+14,  1.6e+15])
            selectors = convertIntoSelectors(massedges, 'm200')
        else:
            raise NoDataException

    elif simtype == 'mxxlsnap41':
        if delta == 200:
            #5 bins
            massedges = np.array([3.5e+14, 3.9e+14, 4.5e+14, 6.4e+14, 9.1e+14, 1.4e+15])
            selectors = convertIntoSelectors(massedges, 'm200')
        elif delta == 500:
            #4 bins
            massedges = np.array([3.0e+14, 3.2e+14, 4.6e+14, 6.5e+14, 1.2e+15])
            selectors = convertIntoSelectors(massedges, 'm500')
        elif delta == 2500:
            #3 bins
            massedges = np.array([1.6e+14, 1.8e+14, 2.4e+14, 3.9e+14])
            selectors = convertIntoSelectors(massedges, 'm2500')

    elif simtype == 'mxxlsnap54':
        if delta == 200:
            #8 bins: suggested to run 0,1,2,5,14,16, 17, 18
            massedges = np.array([1.6e14, 2.6e14, 5e14, 7.2e14, 7.4e14, 7.5e14, 7.7e14, 7.9e14, 8.1e14, 8.4e14, 8.6e14, 8.9e14, 9.3e14, 9.7e14, 1e15, 1.1e15, 1.2e15, 1.4e15, 2.3e15, 4e15])
            allselectors = convertIntoSelectors(massedges, 'm200')
            selectors = [allselectors[0],allselectors[1],allselectors[2],allselectors[5],allselectors[14],allselectors[16],allselectors[17],allselectors[18]]
        elif delta == 500:
            #7 bins
            highmassedges = np.array([6.2e14, 6.4e14, 6.6e14, 7e14, 7.5e14, 8.2e14, 9.8e14, 1.6e15, 2.7e15])
            allselectors = [Selector('m500', 1.5e14, 3.5e14, lambda truth:truth['m200']  < 0),
                            Selector('m500', 3.5e14, 5.5e14, lambda truth:truth['m200']  < 0)]
            allselectors.extend(convertIntoSelectors(highmassedges, 'm500'))
            selectors = [allselectors[0], allselectors[1], allselectors[2], allselectors[5], allselectors[7], allselectors[8], allselectors[9]]
        elif delta == 2500:
            #1 bin
            massedges = np.array([3.15e14, 1.2e15])
            selectors = convertIntoSelectors(massedges, 'm2500')


    return selectors

#########

    

def run(simtype, chaindir, outfile, delta, pdftype, massbin=0, sigmapriorfile = None):

    config = simutils.readConfiguration('%s/config.py' % chaindir)
    simreader = config['simreader']
    nfwutils.global_cosmology.set_cosmology(simreader.getCosmology())
    model = config['model']


    if massbin == -1:
        selector = takeAllMasses(simtype, delta)[0]
    else:
        selectors = defineMassEdges(simtype, delta)
        selector = selectors[massbin]

    isPDF = False
    if pdftype == 'pdf':
        isPDF = True
        halos = dln.loadPosteriors(chaindir, simtype, simreader, delta, selector,
                                   reader = dln.PDFReader, model = model)
    elif pdftype == 'mcmc':
        halos = dln.loadPosteriors(chaindir, simtype, simreader, delta, selector,
                                   reader = dln.MCMCReader,
                                   cprior = 100.)
        
        

    if len(halos) < 10:
        sys.exit(0)

    if len(halos) > 1000:
        print 'Down sampling'
        halos = random.sample(halos, 1000)

    sigmapriors = None
    if sigmapriorfile is not None:
        print 'Using Sigma Priors!'
        sigmapriors = cPickle.load(open(sigmaprior, 'rb'))[massbin]



    success = False
    for i in range(20):

        try:
            
            if isPDF == True:
                parts = dln.buildPDFModel(halos, sigmapriors = sigmapriors)
            else:
                parts = dln.buildMCMCModel(halos)

            model = pymc.Model(parts)
            assert(np.isfinite(model.logp))


            success=True
            break

        except (AssertionError, pymc.ZeroProbability) as e:
            
            continue

    assert(success is True)

        

    with open('%s.massrange' % outfile, 'w') as output:
        output.write('%f\n%f\n' % (selector.masslow, selector.masshigh))
                
    dln.memsample(model, 10000, outputFile = outfile)




if __name__ == '__main__':

    massbin = 0
    sigmaprior = None

    simtype=sys.argv[1]
    chaindir=sys.argv[2]
    outfile=sys.argv[3]
    delta=int(sys.argv[4])
    pdftype=sys.argv[5]
    if len(sys.argv) > 6:
        massbin=int(sys.argv[6])
    if len(sys.argv) > 7:
        sigmaprior = sys.argv[7]

    print 'Called with:', dict(simtype=simtype, chaindir=chaindir, 
                               outfile=outfile, delta=delta, 
                               pdftype = pdftype,
                               massbin=massbin,
                               sigmapriorfile = sigmaprior)

    
    run(simtype, chaindir, outfile, delta, pdftype, massbin, sigmaprior)
        
