import os, sys
#//--------------------------------------------------------------------------------------
def getXsec(samplename, year):
  samplename = str(samplename)
  if 'DYJetsToLL_M-50_HT-100to200'   in samplename: Xsec  =	161
  if 'DYJetsToLL_M-50_HT-200to400'   in samplename: Xsec  =	48.58
  if 'DYJetsToLL_M-50_HT-400to600'   in samplename: Xsec  =	6.982
  if 'DYJetsToLL_M-50_HT-600to800'   in samplename: Xsec  =	1.747
  if 'DYJetsToLL_M-50_HT-800to1200'   in samplename: Xsec  =	0.8052
  if 'DYJetsToLL_M-50_HT-1200to2500'   in samplename: Xsec  =	0.1927
  if 'DYJetsToLL_M-50_HT-2500toInf'   in samplename: Xsec  =	0.003478

  if 'DYJetsToLL_Pt-50To100' in samplename: Xsec = 354.30
  if 'DYJetsToLL_Pt-100To250' in samplename: Xsec = 83.12
  if 'DYJetsToLL_Pt-250To400' in samplename: Xsec = 3.047
  if 'DYJetsToLL_Pt-400To650' in samplename: Xsec = 0.3921
  if 'DYJetsToLL_Pt-650ToInf' in samplename: Xsec = 0.03636
  # #############From XSDB#############
  # if 'DYJetsToLL_Pt-0To50' in samplename: Xsec = 106300.0
  # if 'DYJetsToLL_Pt-50To100' in samplename: Xsec = 407.9
  # if 'DYJetsToLL_Pt-100To250' in samplename: Xsec = 96.8
  # if 'DYJetsToLL_Pt-250To400' in samplename: Xsec = 3.774
  # if 'DYJetsToLL_Pt-400To650' in samplename: Xsec = 0.5164
  # if 'DYJetsToLL_Pt-650ToInf' in samplename: Xsec = 0.04796
  # #############From XSDB#############

  if 'DY1JetsToLL_M-50_LHEZpT_50-150' in samplename: Xsec  = 315.1
  if 'DY1JetsToLL_M-50_LHEZpT_150-250' in samplename: Xsec  = 9.5
  if 'DY1JetsToLL_M-50_LHEZpT_250-400' in samplename: Xsec  = 1.097
  if 'DY1JetsToLL_M-50_LHEZpT_400-inf' in samplename: Xsec  = 0.1207
  # #############From XSDB#############
  # if 'DY1JetsToLL_M-50_LHEZpT_50-150' in samplename: Xsec  = 316.6
  # if 'DY1JetsToLL_M-50_LHEZpT_150-250' in samplename: Xsec  = 9.543
  # if 'DY1JetsToLL_M-50_LHEZpT_250-400' in samplename: Xsec  = 1.098
  # if 'DY1JetsToLL_M-50_LHEZpT_400-inf' in samplename: Xsec  = 0.1193
  # #############From XSDB#############

  if 'DY2JetsToLL_M-50_LHEZpT_50-150' in samplename: Xsec  = 169
  if 'DY2JetsToLL_M-50_LHEZpT_150-250' in samplename: Xsec  = 15.73
  if 'DY2JetsToLL_M-50_LHEZpT_250-400' in samplename: Xsec  = 2.74
  if 'DY2JetsToLL_M-50_LHEZpT_400-inf' in samplename: Xsec  = 0.4492
  # #############From XSDB#############
  # if 'DY2JetsToLL_M-50_LHEZpT_50-150' in samplename: Xsec  = 169.6
  # if 'DY2JetsToLL_M-50_LHEZpT_150-250' in samplename: Xsec  = 15.65
  # if 'DY2JetsToLL_M-50_LHEZpT_250-400' in samplename: Xsec  = 2.737
  # if 'DY2JetsToLL_M-50_LHEZpT_400-inf' in samplename: Xsec  = 0.4477
  # #############From XSDB#############

  ##gamma+jets are reupdated based on twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/SummaryTable1G25ns#Gamma_jets
  if 'GJets_HT-100To200'   in samplename: Xsec  =	9238
  if 'GJets_HT-200To400'   in samplename: Xsec  =	2305
  if 'GJets_HT-400To600'   in samplename: Xsec  =	274.4
  if 'GJets_HT-40To100'   in samplename: Xsec  =	20790
  if 'GJets_HT-600ToInf'   in samplename: Xsec  =	93.46
  # #############From XSDB#############
  # if 'GJets_HT-100To200'   in samplename: Xsec  =	9197.0
  # if 'GJets_HT-200To400'   in samplename: Xsec  =	2185.0
  # if 'GJets_HT-400To600'   in samplename: Xsec  =	259.9
  # if 'GJets_HT-40To100'   in samplename: Xsec  =	20660.0
  # if 'GJets_HT-600ToInf'   in samplename: Xsec  =	85.31
  # #############From XSDB#############


  if 'QCD_HT1000to1500'   in samplename: Xsec  =	1094
  if 'QCD_HT1500to2000'   in samplename: Xsec  =	98.99
  if 'QCD_HT2000toInf'   in samplename: Xsec  =	20.23
  if 'QCD_HT500to700'   in samplename: Xsec  =	29990
  if 'QCD_HT700to1000'   in samplename: Xsec  =	6351

  if 'QCD_HT100to200'   in samplename: Xsec  = 27540000.000000
  if 'QCD_HT200to300'   in samplename: Xsec  = 1735000.000000
  if 'QCD_HT300to500'   in samplename: Xsec  = 366800.000000

  if 'QCD_bEnriched_HT2000toInf' in samplename	:	Xsec = 0.6457
  if 'QCD_bEnriched_HT500to700' in samplename	:	Xsec = 1487.00
  if 'QCD_bEnriched_HT100to200' in	samplename	:	Xsec	=	1125000.00
  if 'QCD_bEnriched_HT700to1000' in	samplename	:	Xsec	=	297.40
  if 'QCD_bEnriched_HT1000to1500' in	samplename	:	Xsec	=	46.61
  if 'QCD_bEnriched_HT300to500' in	samplename	:	Xsec	=	16680.00
  if 'QCD_bEnriched_HT1500to2000' in	samplename	:	Xsec	=	3.74
  if 'QCD_bEnriched_HT200to300' in	samplename	:	Xsec	=	80200.00

  if 'QCD_HT100to200_BGenFilter' in samplename	  :	Xsec =   1274000.00
  if 'QCD_HT200to300_BGenFilter' in samplename	  :	Xsec =   111000.00
  if 'QCD_HT300to500_BGenFilter' in samplename	  :	Xsec =   27960.00
  if 'QCD_HT500to700_BGenFilter' in samplename	  :	Xsec =   3088.00
  if 'QCD_HT700to1000_BGenFilter' in samplename	  :	Xsec =   726.40
  if 'QCD_HT1000to1500_BGenFilter' in samplename	:	Xsec =   136.30
  if 'QCD_HT2000toInf_BGenFilter' in samplename	:	Xsec =   2.89

  # #############From XSDB#############
  # if 'QCD_bEnriched_HT2000toInf' in samplename	:	Xsec = 0.6462
  # if 'QCD_bEnriched_HT500to700' in samplename	:	Xsec = 1487.00
  # if 'QCD_bEnriched_HT100to200' in	samplename	:	Xsec	=	1117000.0
  # if 'QCD_bEnriched_HT700to1000' in	samplename	:	Xsec	=	296.5
  # if 'QCD_bEnriched_HT1000to1500' in	samplename	:	Xsec	=	46.61
  # if 'QCD_bEnriched_HT300to500' in	samplename	:	Xsec	=	16620.0
  # if 'QCD_bEnriched_HT1500to2000' in	samplename	:	Xsec	=	4.017
  # if 'QCD_bEnriched_HT200to300' in	samplename	:	Xsec	=	80220.0
  # #############From XSDB#############

  if 'QCD_HT1500to2000_BGenFilter' in	samplename	:	Xsec	=	13.61
  if 'QCD_HT300to500_BGenFilter' in	samplename	:	Xsec	=	27990.00
  if 'QCD_HT100to200_BGenFilter' in	samplename	:	Xsec	=	1278000.00
  if 'QCD_HT700to1000_BGenFilter' in	samplename	:	Xsec	=	723.80

  if 'TTTo2L2Nu'   in samplename: Xsec  =	87.5595 ##https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
  if 'TTToHadronic'   in samplename: Xsec  =	381.0923  ##https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
  if 'TTToSemiLeptonic'   in samplename: Xsec  =	365.2482  ##https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
  # #############From XSDB#############
  # if 'TTTo2L2Nu'   in samplename: Xsec  =	72.1455 #687.1*0.105
  # if 'TTToHadronic'   in samplename: Xsec  =	314.0047 #687.1*0.457
  # if 'TTToSemiLeptonic'   in samplename: Xsec  =	300.9498 #687.1*0.438
  # #############From XSDB#############

  if 'ST_s-channel_4f_leptonDecays'   in samplename: Xsec  =	10.32
  if 'ST_t-channel_antitop_4f'   in samplename: Xsec  =	80.95
  if 'ST_t-channel_top_4f'   in samplename: Xsec  =	 136.02
  if 'ST_tW_antitop_5f'   in samplename: Xsec  =	38.06
  if 'ST_tW_top_5f'   in samplename: Xsec  =	38.09
  # #############From XSDB#############
  # if 'ST_s-channel_4f_leptonDecays'   in samplename: Xsec  =	3.549
  # if 'ST_t-channel_antitop_4f'   in samplename: Xsec  =	67.91
  # if 'ST_t-channel_top_4f'   in samplename: Xsec  =	 113.3
  # if 'ST_tW_antitop_5f'   in samplename: Xsec  =	32.51
  # if 'ST_tW_top_5f'   in samplename: Xsec  =	32.45
  # #############From XSDB#############


  if 'WJetsToLNu_HT-100To200'   in samplename: Xsec  =	1395
  if 'WJetsToLNu_HT-200To400'   in samplename: Xsec  =	409.3
  if 'WJetsToLNu_HT-400To600'   in samplename: Xsec  =	57.91
  if 'WJetsToLNu_HT-600To800'   in samplename: Xsec  =	12.93
  if 'WJetsToLNu_HT-800To1200'   in samplename: Xsec  =	5.395
  if 'WJetsToLNu_HT-1200To2500'   in samplename: Xsec  =	1.08
  if 'WJetsToLNu_HT-2500ToInf'   in samplename: Xsec  =	0.008053

  if 'WJetsToLNu_TuneCUETP8M1' in samplename: Xsec = 60380

  if 'WJetsToLNu_Wpt-50To100' in samplename: Xsec = 3569.00
  if 'WJetsToLNu_Pt-50To100' in samplename: Xsec = 3569.00
  if 'WJetsToLNu_Pt-100To250' in samplename: Xsec = 769.8
  if 'WJetsToLNu_Pt-250To400' in samplename: Xsec = 27.62
  if 'WJetsToLNu_Pt-400To600' in samplename: Xsec = 3.591
  if 'WJetsToLNu_Pt-600ToInf' in samplename: Xsec = 0.549

  # #############From Deepak#############
  # if 'WJetsToLNu_Wpt-50To100' in samplename: Xsec = 3570.00
  # if 'WJetsToLNu_Pt-50To100' in samplename: Xsec = 3570.0
  # if 'WJetsToLNu_Pt-100To250' in samplename: Xsec = 779.1
  # if 'WJetsToLNu_Pt-250To400' in samplename: Xsec = 27.98
  # if 'WJetsToLNu_Pt-400To600' in samplename: Xsec = 3.604
  # if 'WJetsToLNu_Pt-600ToInf' in samplename: Xsec = 0.5545
  # #############From Deepak#############

  # #############From XSDB#############
  # if 'WJetsToLNu_Wpt-50To100' in samplename: Xsec = 3298.373338
  # if 'WJetsToLNu_Pt-50To100' in samplename: Xsec = 3570.0
  # if 'WJetsToLNu_Pt-100To250' in samplename: Xsec = 689.749632
  # if 'WJetsToLNu_Pt-250To400' in samplename: Xsec = 24.5069015
  # if 'WJetsToLNu_Pt-400To600' in samplename: Xsec = 3.110130566
  # if 'WJetsToLNu_Pt-600ToInf' in samplename: Xsec = 0.4683178368
  # #############From XSDB#############



  if 'W1JetsToLNu_LHEWpT_100-150' in samplename: Xsec  = 284.06
  if 'W1JetsToLNu_LHEWpT_150-250' in samplename: Xsec  = 71.71
  if 'W1JetsToLNu_LHEWpT_250-400' in samplename: Xsec  = 8.06
  if 'W1JetsToLNu_LHEWpT_400-inf' in samplename: Xsec  = 0.89

  if 'W2JetsToLNu_LHEWpT_100-150' in samplename: Xsec  = 281.63
  if 'W2JetsToLNu_LHEWpT_150-250' in samplename: Xsec  = 108.59
  if 'W2JetsToLNu_LHEWpT_250-400' in samplename: Xsec  = 18.03
  if 'W2JetsToLNu_LHEWpT_400-inf' in samplename: Xsec  = 3.0

  if 'WW' in samplename: Xsec  =	77.9187  ## https://indico.cern.ch/event/439995/contributions/1094416/attachments/1143460/1638648/diboson_final.pdf
  if 'WZ' in samplename: Xsec  =	46.781  ## https://indico.cern.ch/event/439995/contributions/1094416/attachments/1143460/1638648/diboson_final.pdf
  if 'ZZ'   in samplename: Xsec  =	23.1168 ## https://indico.cern.ch/event/439995/contributions/1094416/attachments/1143460/1638648/diboson_final.pdf
  # #############From XSDB#############
  # if 'WW' in samplename: Xsec  =	75.8
  # if 'WZ' in samplename: Xsec  =	27.59
  # if 'ZZ'   in samplename: Xsec  =	12.23
  # #############From XSDB#############

  if 'ZJetsToNuNu_HT-100To200'   in samplename: Xsec  =	304.5
  if 'ZJetsToNuNu_HT-200To400'   in samplename: Xsec  =	91.85
  if 'ZJetsToNuNu_HT-400To600'   in samplename: Xsec  =	13.11
  if 'ZJetsToNuNu_HT-600To800'   in samplename: Xsec  =	3.257
  if 'ZJetsToNuNu_HT-800To1200'   in samplename: Xsec  =	1.499
  if 'ZJetsToNuNu_HT-1200To2500'   in samplename: Xsec  =	0.343
  if 'ZJetsToNuNu_HT-2500ToInf'   in samplename: Xsec  =	0.005146

  if 'DYJetsToNuNu_PtZ-100To250' in samplename: Xsec = 54.91*3
  if 'DYJetsToNuNu_PtZ-250To400' in samplename: Xsec = 2.073*3
  if 'DYJetsToNuNu_PtZ-400To650' in samplename: Xsec = 0.2785*3
  if 'DYJetsToNuNu_PtZ-50To100' in samplename: Xsec = 236.5*3
  if 'DYJetsToNuNu_PtZ-650ToInf' in samplename: Xsec = 0.02604*3
  # #############From XSDB#############
  # if 'DYJetsToNuNu_PtZ-100To250' in samplename: Xsec = 54.86*3
  # if 'DYJetsToNuNu_PtZ-250To400' in samplename: Xsec = 2.073*3
  # if 'DYJetsToNuNu_PtZ-400To650' in samplename: Xsec = 0.2783*3
  # if 'DYJetsToNuNu_PtZ-50To100' in samplename: Xsec = 237.2*3
  # if 'DYJetsToNuNu_PtZ-650ToInf' in samplename: Xsec = 0.02603*3
  # #############From XSDB#############

  if 'Z1JetsToNuNu_M-50_LHEZpT_50-150' in samplename: Xsec  = 598.9
  if 'Z1JetsToNuNu_M-50_LHEZpT_150-250' in samplename: Xsec  = 18.04
  if 'Z1JetsToNuNu_M-50_LHEZpT_250-400' in samplename: Xsec  = 2.051
  if 'Z1JetsToNuNu_M-50_LHEZpT_400-inf' in samplename: Xsec  = 0.2251
  # #############From XSDB#############
  # if 'Z1JetsToNuNu_M-50_LHEZpT_50-150' in samplename: Xsec  = 596.1
  # if 'Z1JetsToNuNu_M-50_LHEZpT_150-250' in samplename: Xsec  = 18.0
  # if 'Z1JetsToNuNu_M-50_LHEZpT_250-400' in samplename: Xsec  = 2.068
  # if 'Z1JetsToNuNu_M-50_LHEZpT_400-inf' in samplename: Xsec  = 0.2243
  # #############From XSDB#############

  if 'Z2JetsToNuNu_M-50_LHEZpT_50-150' in samplename: Xsec  = 326.3
  if 'Z2JetsToNuNu_M-50_LHEZpT_150-250' in samplename: Xsec  = 29.6
  if 'Z2JetsToNuNu_M-50_LHEZpT_250-400' in samplename: Xsec  = 5.174
  if 'Z2JetsToNuNU_M-50_LHEZpT_400-inf' in samplename: Xsec  = 0.8472
  # #############From XSDB#############
  # if 'Z2JetsToNuNu_M-50_LHEZpT_50-150' in samplename: Xsec  = 326.5
  # if 'Z2JetsToNuNu_M-50_LHEZpT_150-250' in samplename: Xsec  = 29.87
  # if 'Z2JetsToNuNu_M-50_LHEZpT_250-400' in samplename: Xsec  = 5.219
  # if 'Z2JetsToNuNU_M-50_LHEZpT_400-inf' in samplename: Xsec  = 0.8433
  # #############From XSDB#############

  if 'ggZH_HToBB_ZToNuNu_M125'  in samplename: Xsec = 0.01222
  if 'ggZH_HToBB_ZToLL_M125'    in samplename: Xsec = 0.006185
  if 'WminusH_HToBB' in samplename: Xsec = 0.3654
  if 'WplusH_HToBB_WToLNu_M125' in samplename: Xsec = 0.2819
  if 'ZH_HToBB_ZToLL_M125'      in samplename and 'ggZH_HToBB_ZToLL_M125' not in samplename:   Xsec = 0.07924
  if 'ZH_HToBB_ZToNuNu_M125'    in samplename and 'ggZH_HToBB_ZToNuNu_M125' not in samplename: Xsec = 0.1565
  # #############From XSDB#############
  # if 'ggZH_HToBB_ZToNuNu_M125'  in samplename: Xsec = 0.01373
  # if 'ggZH_HToBB_ZToLL_M125'    in samplename: Xsec = 0.006954
  # if 'WminusH_HToBB' in samplename: Xsec = 0.1971
  # if 'WplusH_HToBB_WToLNu_M125' in samplename: Xsec = 0.2791
  # if 'ZH_HToBB_ZToLL_M125'      in samplename and 'ggZH_HToBB_ZToLL_M125' not in samplename:   Xsec = 0.07523
  # if 'ZH_HToBB_ZToNuNu_M125'    in samplename and 'ggZH_HToBB_ZToNuNu_M125' not in samplename: Xsec = 0.1482
  # #############From XSDB#############

  return Xsec
