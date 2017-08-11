#!/usr/bin/env python
############################

import glob, cPickle, sys, os, re
import numpy as np
import nfwutils, fitrunner

###########################


idpatterns = dict(mxxlsnap41 = re.compile('halo_41_(\d+_\d)\.out'),
                  mxxlsnap54 = re.compile('halo_54_(\d+_\d)\.out'),
                  bcc = re.compile('cluster_(\d+)\.hdf5.out'),
                  bk11snap141 = re.compile('haloid(\d+)_zLens.+'),
                  bk11snap124 = re.compile('haloid(\d+)_zLens.+'),
                  analytic_mxxlsnap54 = re.compile('analytic_(\d+)\.out'),
                  analytic_mxxlsnap41 = re.compile('analytic_(\d+)\.out'),
                  analytic_bk11snap141 = re.compile('analytic_(\d+)\.out'),
                  analytic_bk11snap124 = re.compile('analytic_(\d+)\.out'))



def consolidateFits(workdir, simtype, outdir):

    failfile = open('{0}/fails'.format(outdir), 'w')


    idpattern = idpatterns[simtype]

    answers = cPickle.load(open('{0}/{1}_answers.pkl'.format(workdir, simtype), 'rb'))

    outputfiles = glob.glob('%s/*.out' % outdir)
    nhalos = len(outputfiles)

    ids = []
    measured_m200s = np.zeros(nhalos)
    measured_m200errs = np.zeros(nhalos)
    measured_m500s = np.zeros(nhalos)
    measured_m500errs = np.zeros(nhalos)
    measured_cs = np.zeros(nhalos)
    measured_rs = np.zeros(nhalos)

    true_m200s = np.zeros(nhalos)
    true_m500s = np.zeros(nhalos)
    true_cs = np.zeros(nhalos)
    redshifts = np.zeros(nhalos)

    results = dict(ids = ids,
                        measured_m200s = measured_m200s, 
                        measured_m200errs = measured_m200errs,
                        measured_m500s = measured_m500s,
                        measured_m500errss = measured_m500errs,
                        measured_cs = measured_cs,
                        measured_rs = measured_rs,
                        true_m200s = true_m200s,
                        true_m500s = true_m500s,
                        true_cs = true_cs,
                        redshifts = redshifts)

    class WeirdException(Exception): pass 


    #load up the environment for cosmology, and mc relation if used
    config = nfwfit.readConfiguration('{0}/config.sh'.format(outdir))
    simreader = nfwfit.buildSimReader(config)
    nfwutils.global_cosmology.set_cosmology(simreader.getCosmology())
    fitter = nfwfit.buildFitter(config)

    configname = os.path.basename(outdir)


    for i,output in enumerate(outputfiles):

        filebase = os.path.basename(output)

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
        ids.append(haloid)
        true_m200s[i] = truth['m200']
        true_m500s[i] = truth['m500']
        true_cs[i] = truth['concen']
        redshifts[i] = truth['redshift']



        input = open(output)
        measured = cPickle.load(input)
        input.close()


        if measured is None:
            print 'Fail {0} {1}'.format(configname, haloid)
            failfile.write('Fail {0} {1}\n'.format(configname, haloid))
            continue


        measured_m200s[i] = measured[0]['m200']*fitter.model.massScale*nfwutils.global_cosmology.h
        measured_m200errs[i] = np.mean(np.abs(measured[1]['m200']))*fitter.model.massScale*nfwutils.global_cosmology.h
        if 'c200' in measured:
            measured_cs[i] = measured['c200']
        else:
            ## need to dig up the mc relation
            measured_cs[i] = fitter.model.massconRelation(np.abs(measured_m200s[i]), 
                                                          redshifts[i], 
                                                          fitter.model.overdensity)

        #####
        #calculate m500

        measured_rs[i] = nfwutils.rscaleConstM(np.abs(measured_m200s[i]), measured_cs[i],redshifts[i],
                                      fitter.model.overdensity)
        measured_m500s[i] = nfwutils.Mdelta(measured_rs[i],
                                            measured_cs[i],
                                            redshifts[i],
                                            500)

        if measured_m200s[i] < 0:
            measured_m500s[i] = -measured_m500s[i]

        if not np.isfinite(measured_m500s[i]):
            print 'NOT FINITE'
            print haloid
            print measured








    cPickle.dump(results, open('%s/consolidated.pkl' % outdir, 'w'))

    failfile.close()    




if __name__ == '__main__':

    workdir=sys.argv[1]
    simtype=sys.argv[2]
    outdir=sys.argv[3]

    consolidateFits(workdir, simtype, outdir)
