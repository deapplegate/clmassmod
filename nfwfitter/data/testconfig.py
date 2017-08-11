import nfwfitter.profilebuilder as profilebuilder
profilebuilder=profilebuilder.ProfileBuilder()


massprior='linear'
import nfwfitter.rescalecluster as rescalecluster
rescalecluster=rescalecluster.NoRedshiftRescaling()
import nfwfitter.betacalcer as betacalcer
betacalcer=betacalcer.FixedRedshift()
zsource=1.5
import nfwfitter.basicBinning as basicBinning
binner=basicBinning.GaussianFixedBins()
profilecol='r_mpc'
binspacing='linear'
nbins=12
import nfwfitter.galaxypicker as galaxypicker
densitypicker=galaxypicker.DensityPicker()
nperarcmin=20

import nfwfitter.shearnoiser as shearnoiser
shearnoiser=shearnoiser.GaussianShapeNoise()
shapenoise=0.25

galaxypicker=densitypicker
import nfwfitter.binnoiser as binnoiser
binnoiser = binnoiser.NoBinNoise()

import nfwfitter.readMXXL as readMXXL
simreader=readMXXL.MXXLSimReader()

import nfwfitter.nfwmodel as nfwmodel
base_massmodel=nfwmodel.NFW_Model()
import nfwfitter.massmodel as massmodel
massmodel=massmodel.MC_Model()

import nfwfitter.colossusMassCon as colossusMassCon
massconRelation=colossusMassCon.ColossusMC()
colossusmcname='diemer15'

import nfwfitter.fitrunner as fitrunner
fitter=fitrunner.PDFScanner()


profileMin=.5
profileMax=2.5 

zsource=2.0
import nfwfitter.centergenerator as centergenerator
centergenerator = centergenerator.NoOffset()
