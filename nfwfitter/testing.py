#!/usr/bin/env python
#############
# Home of some unit tests for nfwfitter
############

import numpy as np
import unittest
import cPickle
import pkg_resources
import scipy.interpolate

import fitrunner
import simutils

############


class TestNFWModelPDFScan(unittest.TestCase):

    def test_nfwfit(self):

        profile = cPickle.load(pkg_resources.resource_stream('nfwfitter',
                                                             'data/testprofile.pkl'))

        config = simutils.readConfiguration(pkg_resources.resource_filename('nfwfitter',
                                                                            'data/testconfig.py'))

        fitter = config['fitter']
        newpdfs = fitter(profile)

        original_masses, original_pdfs = cPickle.load(pkg_resources.resource_stream('nfwfitter',
                                                                                    'data/original.out'))

        self.assertItemsEqual(original_pdfs.keys(), newpdfs.keys())

        for delta in original_pdfs.keys():

            interp_pdf = scipy.interpolate.interp1d(original_masses,
                                                    original_pdfs[delta],
                                                    kind='cubic')

            new_masses, new_pdf = newpdfs[delta]

            expected_pdf = interp_pdf(new_masses)
            self.assertTrue((np.abs(new_pdf/expected_pdf - 1.)[np.isfinite(expected_pdf)] < 0.001).all())

            

        

############

if __name__ == '__main__':
    unittest.main()

