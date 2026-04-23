import os
import sys
import datetime
import optparse
import ROOT as ROOT
import array
import string
import math
import numpy
import sampleXsecFiles.sample_xsec_run2_harmony as sample_xsec

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetFrameLineWidth(3)
#ROOT.gStyle.SetErrorX(0)
ROOT.gStyle.SetLineWidth(1)

#command  python StackPlotter_syst.py -d <DATASET_NAME> -m -y <Year> -D <histo_DIR> -S <signal_Dir>
usage = "usage: %prog [options] arg1 arg2"
parser = optparse.OptionParser(usage)

parser.add_option("-d", "--data", dest="datasetname")
parser.add_option("-D", "--pDir", type="string", dest="rootFileDir", help="histogram dir")
parser.add_option("-S", "--sigDir", type="string", dest="SIGrootFileDir", help="signal histogram")
parser.add_option("-v", "--version", type="string",dest="Version", help="version of histograms")
parser.add_option("-s", "--sr", action="store_true", dest="plotSIGNAL")
parser.add_option("-m", "--mu", action="store_true", dest="plotMuChannels")
parser.add_option("-e", "--ele", action="store_true", dest="plotEleChannels")
parser.add_option("-p", "--pho", action="store_true", dest="plotPhoChannels")
parser.add_option("-q", "--qcd", action="store_true", dest="plotQCDChannels")
parser.add_option("-y", "--year", dest="year", default="Year")
parser.add_option("-u", "--unblind", action="store_true", dest="unblinded")

(options, args) = parser.parse_args()

if options.plotSIGNAL == None:
    makeSIGplots = False
else:
    makeSIGplots = options.plotSIGNAL

if options.unblinded == None:
    datainSR = False
else:
    datainSR = options.unblinded

if options.plotMuChannels == None:
    makeMuCHplots = False
else:
    makeMuCHplots = options.plotMuChannels

if options.plotEleChannels == None:
    makeEleCHplots = False
else:
    makeEleCHplots = options.plotEleChannels

if options.plotPhoChannels == None:
    makePhoCRplots = False
else:
    makePhoCRplots = options.plotPhoChannels

if options.plotQCDChannels == None:
    makeQCDbCRplots = False
else:
    makeQCDbCRplots = options.plotQCDChannels

if options.datasetname.upper() == "SE":
    dtset = "SE"
elif options.datasetname.upper() == "SP":
    dtset = "SP"
elif options.datasetname.upper() == "SM":
    dtset = "SM"
else:
    dtset = "MET"

print("Using dataset "+dtset)

runOn2016 = False
runOn2017 = False
runOn2018 = False
if options.year == '2016':
    runOn2016 = True
elif options.year == '2017':
    runOn2017 = True
elif options.year == '2018':
    runOn2018 = True
else:
    print('Please provide on which year you want to run?')

if runOn2016:
    import sampleXsecFiles.sig_sample_xsec_2016 as sig_sample_xsec
    from systPythonFiles.uncert_v16_12_00_02 import syst_dict
    luminosity = 35.90 * 1000
    luminosity_ = '{0:.1f}'.format(35.90)
elif runOn2017:
    import sampleXsecFiles.sig_sample_xsec_2017 as sig_sample_xsec
    from systPythonFiles.uncert_v17_12_00_02 import syst_dict
    luminosity = 41.5 * 1000
    luminosity_ = '{0:.1f}'.format(41.50)
elif runOn2018:
    import sampleXsecFiles.sig_sample_xsec_2018 as sig_sample_xsec
    from systPythonFiles.uncert_v18_12_00_02 import syst_dict
    # luminosity = 13.90 * 1000 #A
    # luminosity_ = '{0:.2f}'.format(13.90)  # A
    # luminosity = 7.04 * 1000  # B
    # luminosity_ = '{0:.2f}'.format(7.04)  # AB
    # luminosity = 20.94 * 1000  # AB
    # luminosity_ = '{0:.2f}'.format(20.94) ##AB
    # luminosity = 38.70 * 1000  # CD
    # luminosity_ = '{0:.2f}'.format(38.70)  # CD
    luminosity = 59.64 * 1000
    luminosity_ = '{0:.1f}'.format(59.64)


datestr = str(datetime.date.today().strftime("%d%m%Y"))

if options.Version == None:
    print('Please provide which version of histograms are being plotted')
    sys.exit()
else:
    histVersion = options.Version

if options.rootFileDir == None:
    print('Please provide histogram directory name')
    sys.exit()
else:
    path = options.rootFileDir

sig_path = options.SIGrootFileDir

print("sig_path", sig_path)
if makeMuCHplots:
    yield_outfile_binwise = open('YieldsFiles/'+histVersion+'_Muon_binwise.txt','w')
    yield_outfile = open('YieldsFiles/'+histVersion+'_Muon.txt','w')
if makeEleCHplots:
    yield_outfile_binwise = open('YieldsFiles/'+histVersion+'_Electron_binwise.txt','w')
    yield_outfile = open('YieldsFiles/'+histVersion+'_Electron.txt', 'w')

alpha_list = list(string.ascii_uppercase)

syst_sources = ['CMSyear_eff_b', 'CMSyear_fake_b', 'EWK', 'CMSyear_Top', 'CMSyear_trig_met',
                'CMSyear_trig_ele', 'CMSyear_EleID', 'CMSyear_EleRECO', 'CMSyear_MuID', 'CMSyear_MuISO',
                'CMSyear_MuTRK','CMSyear_PU', 'JECAbsolute', 'JECAbsolute_year', 'JECBBEC1', 'JECBBEC1_year',
                'JECEC2', 'JECEC2_year', 'JECFlavorQCD', 'JECHF', 'JECHF_year', 'JECRelativeBal', 'JECRelativeSample_year', 'En'
]

def set_overflow(hist):
    bin_num = hist.GetXaxis().GetNbins()
    # Add overflow bin content to last bin
    hist.SetBinContent(bin_num, hist.GetBinContent(bin_num+1)+hist.GetBinContent(bin_num))
    hist.SetBinContent(bin_num+1, 0.)
    return hist


def addErrorForEmptyBins(hist, histMean):
    binerr_ = histMean #.Integral()
    for ibin in range(1, hist.GetXaxis().GetNbins()+1):
        if hist.GetBinContent(ibin) <= 0.0:
            hist.SetBinContent(ibin, 0.0)
            hist.SetBinError(ibin, binerr_*1.8)
    return hist

# limit_varSR = 'MET'
# limit_varCR = 'Recoil'
# minBin = 250
# maxBin = 1000
# var_legendCR = 'Recoil (GeV)'
# var_legendSR = 'p_{T}^{miss} (GeV)'

# limit_varSR = limit_varCR = 'bdtscore'
# minBin = -1
# maxBin = 1
# var_legendCR = var_legendSR ='BDT OutPut'

# limit_varSR = limit_varCR = 'ctsValue'
# minBin = 0
# maxBin = 1
# var_legendCR = var_legendSR = 'cos(#Theta)*'
def setVarBin(h_temp1, hist):
    # bins_ = np.linspace(-1.0, 1.0, num = 5) ####%%% for bdtSCORE ####%%% ## binset1
    # bins_ = [250,300,400,550,1000] ##default bining
    # bins_ = [250,280,340,460,1000]
    # bins_ = [250,270,320,400,1000]
    # bins_ = [250,275,325,400,1000]
    # bins_ = [250,265,325,425,1000]
    # bins_ = [250, 300, 325, 375, 1000]
    # bins_ = [250, 300, 312, 325, 1000]
    # bins_ = [250,260,300,350,1000]
    # bins_ = [250,300,350,400,500]
    # bins_ = [250,313,375,437,500]
    # bins_ =  [250,280,310,340,500]
    # bins_ = [250,260,270,280,500]
    # bins_ =  [250,260,270,280,1000]
    # bins_ = [250,260,270,280,300,350,400,500,1000]
    if 'ctsValue' in hist:
        bins_ = [0.0 , 0.25, 0.50 , 0.75, 1.0]  # 4 equal bins for CTS
        # bins_ = [0.0 , 0.45, 0.80 , 0.95, 1.0] # binset_v2
        # bins_ = [0.0 , 0.40, 0.75 , 0.90, 1.0]  # binset_v3
        # bins_ = [0.0 , 0.35, 0.55 , 0.75, 1.0]  # binset_v4
        # bins_ = [0.0 , 0.20, 0.50 , 0.80, 1.0] # binset_v5
        # bins_ = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0] # binset_v6
        # bins_ = [0.0, 0.17, 0.34, 0.51, 0.68, 0.84,1.0] # binset_v7
        # bins_ = [0., 0.125, 0.25, 0.375, 0.5, 0.625 ,0.75, 0.875, 1.] # binset_v8
        # bins_ = [0., 0.15, 0.25, 0.35, 0.45, 0.55 ,0.65, 0.80, 1.] # binset_v9
        # bins_ = [0., 0.10, 0.25, 0.40, 0.45, 0.60 ,0.75, 0.90, 1.] # binset_v10
        # bins_ = [0., 0.15, 0.25, 0.35, 0.45, 0.57 ,0.65, 0.80, 1.] # binset_v11
        # bins_ = [0., 0.15, 0.25, 0.35, 0.45, 0.57 ,0.70, 0.83, 1.] # binset_v12
        # bins_ = [0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        # bins_ = [0.    , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.85, 1.    ]
        # bins_ = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
        # bins_ =  [0., 0.08, 0.16, 0.25, 0.33, 0.41, 0.5, 0.58, 0.66, 0.75, 0.83, 0.91, 1.]
        # bins_ = [0., 0.06, 0.12, 0.18, 0.25, 0.31, 0.37, 0.43, 0.5, 0.56, 0.62, 0.68, 0.75, 0.81, 0.87, 0.93, 1.]
        # bins_ =  [0., 0.04, 0.08, 0.125, 0.16, 0.20, 0.25, 0.29, 0.33, 0.375, 0.41, 0.45, 0.5, 0.54, 0.58, 0.625, 0.66, 0.70, 0.75, 0.79, 0.83, 0.875, 0.91, 0.95, 1.]
        # bins_ = [0. , 0.25, 0.4  , 0.55, 0.70 , 0.85, 1.    ]
        h_temp = h_temp1.Rebin(len(bins_)-1, 'h_temp', array.array('d', bins_))
    elif ('_MET' in hist and 'SR' in hist and '_METPhi' not in hist) or ('Recoil' in hist and 'CR' in hist) or ('_MET' in hist and 'QCD' in hist):
        bins_ = [250, 300, 400, 550, 1000] #4 bins for MET
        # bins_ = [250, 300, 400, 1000]
        h_temp = h_temp1.Rebin(len(bins_)-1, 'h_temp', array.array('d', bins_))
        # h_temp = h_temp1.Rebin(30)
    else:
        h_temp = h_temp1
    return h_temp

def setHistStyle(h_temp2, hist, rebin):
    bins = h_temp2.GetNbinsX()
    if rebin > 1:
        if bins%rebin == 0:
            h_temp_ = h_temp2.Rebin(rebin)
        elif bins%(rebin+1) == 0:
            h_temp_ = h_temp2.Rebin(rebin+1)
        elif bins%(rebin-1) == 0:
            h_temp_ = h_temp2.Rebin(rebin-1)
    else:
        h_temp_ = h_temp2
    return h_temp_


def SetCMSAxis(h, xoffset=1., yoffset=1.):
    h.GetXaxis().SetTitleSize(0.047)
    h.GetYaxis().SetTitleSize(0.047)
    if type(h) is ((not ROOT.TGraphAsymmErrors) or (not ROOT.TGraph)):
        h.GetZaxis().SetTitleSize(0.047)
    h.GetXaxis().SetLabelSize(0.047)
    h.GetYaxis().SetLabelSize(0.047)
    if type(h) is ((not ROOT.TGraphAsymmErrors) or (not ROOT.TGraph)):
        h.GetZaxis().SetLabelSize(0.047)
    h.GetXaxis().SetTitleOffset(xoffset)
    h.GetYaxis().SetTitleOffset(yoffset)
    return h

def ExtraText(text_, x_, y_):
    if not text_: print("nothing provided as text to ExtraText, function crashing")
    ltx = ROOT.TLatex(x_, y_, text_)
    if len(text_) > 0:
        ltx.SetTextFont(42)
        ltx.SetTextSize(0.049)
        #ltx.Draw(x_,y_,text_)
        ltx.Draw('same')
    return ltx

def myCanvas1D():
    c = ROOT.TCanvas("myCanvasName", "The Canvas Title", 650, 600)
    c.SetBottomMargin(0.050)
    c.SetRightMargin(0.050)
    c.SetLeftMargin(0.050)
    c.SetTopMargin(0.050)
    return c

def SetLegend(coordinate_=[.50, .65, .90, .90], ncol=2):
    c_ = coordinate_
    legend = ROOT.TLegend(c_[0], c_[1], c_[2], c_[3])
    legend.SetBorderSize(0)
    legend.SetNColumns(ncol)
    legend.SetLineColor(1)
    legend.SetLineStyle(1)
    legend.SetLineWidth(1)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.045)
    return legend

def drawenergy1D(is2017, text_="Work in progress 2018", data=True):
    #pt = ROOT.TPaveText(0.0877181,0.9,0.9580537,0.96,"brNDC")
    pt = ROOT.TPaveText(0.0877181, 0.95, 0.9580537, 0.96, "brNDC")
    pt.SetBorderSize(0)
    pt.SetTextAlign(12)
    pt.SetFillStyle(0)
    pt.SetTextFont(52)
    cmstextSize = 0.07
    preliminarytextfize = cmstextSize * 0.7
    lumitextsize = cmstextSize * 0.7
    pt.SetTextSize(cmstextSize)
    text = pt.AddText(0.03, 0.57, "#font[61]{CMS}")
    #pt1 = ROOT.TPaveText(0.0877181,0.9,0.9580537,0.96,"brNDC")
    pt1 = ROOT.TPaveText(0.0877181, 0.936, 0.9580537, 0.96, "brNDC")
    pt1.SetBorderSize(0)
    pt1.SetTextAlign(12)
    pt1.SetFillStyle(0)
    pt1.SetTextFont(52)
    pt1.SetTextSize(preliminarytextfize)
    #text1 = pt1.AddText(0.215,0.4,text_)
    text1 = pt1.AddText(0.15, 0.40, text_)
    #pt2 = ROOT.TPaveText(0.0877181,0.9,0.9580537,0.96,"brNDC")
    pt2 = ROOT.TPaveText(0.0877181, 0.95, 0.9580537, 0.96, "brNDC")
    pt2.SetBorderSize(0)
    pt2.SetTextAlign(12)
    pt2.SetFillStyle(0)
    pt2.SetTextFont(52)
    pt2.SetTextFont(42)
    pt2.SetTextSize(lumitextsize)
    pavetext = ''
    if is2017 and data:
        pavetext = str(luminosity_)+' fb^{-1}'+" (13 TeV)"
    if (not is2017) and data:
        pavetext = str(luminosity_)+' fb^{-1}'+"(13 TeV)"
    if is2017 and not data:
        pavetext = "13 TeV"
    if (not is2017) and not data:
        pavetext = "13 TeV"
    if data:
        text3 = pt2.AddText(0.68, 0.5, pavetext)
    if not data:
        text3 = pt2.AddText(0.68, 0.5, pavetext)
    return [pt, pt1, pt2]

def makeplot(loc, hist, titleX, XMIN, XMAX, Rebin, ISLOG, NORATIOPLOT, reg, varBin, row=2):
    if 'ctsValue' in hist:
        limit_varSR = limit_varCR = 'ctsValue'
    elif ('_MET' in hist and 'SR' in hist and '_METPhi' not in hist) or ('Recoil' in hist and 'CR' in hist) and '_METPhi' not in hist or ('_MET' in hist and 'QCD' in hist and '_METPhi' not in hist):
        limit_varSR = 'MET'
        limit_varCR = 'Recoil'
    else:
        limit_varSR = 'NOT_VALID'
        limit_varCR = 'NOT_VALID'
    print('plotting histogram:    ', hist)
    isrebin = True  # bool(varBin)
    if runOn2016:
        files = open("sampleListFiles/samplelist_2016.txt", "r")
    elif runOn2017:
        files = open("sampleListFiles/samplelist_2017.txt", "r")
    elif runOn2018:
        files = open("sampleListFiles/samplelist_2018.txt", "r")
    ##=================================================================
    if 'preselR' in hist:
        histolabel = "Pre Selection"
    elif '_SR_1b' in hist:
        histolabel = "SR(1b)"
    elif '_SR_2b' in hist:
        histolabel = "SR(2b)"
    elif 'ZmumuCR_1b' in hist:
        histolabel = "Z(#mu#mu)+1b CR"
    elif 'ZeeCR_1b' in hist:
        histolabel = "Z(ee)+1b CR"
    elif 'ZmumuCR_2j' in hist:
        histolabel = "Z(#mu#mu)+2j CR"
    elif 'ZeeCR_2j' in hist:
        histolabel = "Z(ee)+2j CR"
    elif 'WmunuCR_1b' in hist:
        histolabel = "W(#mu#nu)+1b CR"
    elif 'WenuCR_1b' in hist:
        histolabel = "W(e#nu)+1b CR"
    elif 'TopmunuCR_1b' in hist:
        histolabel = "t#bar{t}(#mu#nu)+1b CR"
    elif 'TopenuCR_1b' in hist:
        histolabel = "t#bar{t}(e#nu)+1b CR"
    elif 'ZmumuCR_2b' in hist:
        histolabel = "Z(#mu#mu)+2b CR"
    elif 'ZeeCR_2b' in hist:
        histolabel = "Z(ee)+2b CR"
    elif 'ZmumuCR_3j' in hist:
        histolabel = "Z(#mu#mu)+3j CR"
    elif 'ZeeCR_3j' in hist:
        histolabel = "Z(ee)+3j CR"
    elif 'WmunuCR_2b' in hist:
        histolabel = "W(#mu#nu)+2b CR"
    elif 'WenuCR_2b' in hist:
        histolabel = "W(e#nu)+2b CR"
    elif 'TopmunuCR_2b' in hist:
        histolabel = "t#bar{t}(#mu#nu)+2b CR"
    elif 'TopenuCR_2b' in hist:
        histolabel = "t#bar{t}(e#nu)+2b CR"
    elif '_QCDbCR_1b' in hist:
        histolabel = "QCD(1b)"
    elif '_QCDbCR_2b' in hist:
        histolabel = "QCD(2b)"
    else:
        histolabel = "testing"
    ##=================================================================
    xsec = 1.0
    DIBOSON = ROOT.TH1F()
    Top = ROOT.TH1F()
    WJets = ROOT.TH1F()
    DYJets = ROOT.TH1F()
    ZJets = ROOT.TH1F()
    STop = ROOT.TH1F()
    GJets = ROOT.TH1F()
    QCD = ROOT.TH1F()
    SMH = ROOT.TH1F()
    ##=================================================================
    DYJets_Hists = []
    ZJets_Hists = []
    WJets_Hists = []
    GJets_Hists = []
    DIBOSON_Hists = []
    STop_Hists = []
    Top_Hists = []
    QCD_Hists = []
    SMH_Hists = []
    MET_Hist = []
    SE_Hist = []
    ##=================================================================
    DYJets_Hists_meanWeight = []
    ZJets_Hists_meanWeight = []
    WJets_Hists_meanWeight = []
    GJets_Hists_meanWeight = []
    DIBOSON_Hists_meanWeight = []
    STop_Hists_meanWeight = []
    Top_Hists_meanWeight = []
    QCD_Hists_meanWeight = []
    SMH_Hists_meanWeight = []
    ##=================================================================
    count = 0
    for file in files.readlines()[:]:
        myFile = path+'/'+file.rstrip()
        Str = str(count)
        exec("f"+Str+"=ROOT.TFile(myFile,'READ')", locals(), globals())
        exec("h_temp=f"+Str+".Get("+"\'"+str(hist)+"\'"+")", locals(), globals())
        exec("h_meanWeight=f"+Str+".Get("+"\'h_reg_"+str(reg)+"_meanWeight\'"+")", locals(), globals())
        exec("h_total_weight=f"+Str+".Get('h_total_mcweight')", locals(), globals())
        total_events = h_total_weight.Integral()
        if 'WJetsToLNu' in file or 'W1JetsToLNu' in file or 'W2JetsToLNu' in file or 'WJetsToLNu_Pt' in file:
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*luminosity)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                WJets_Hists.append(h_temp2)
            else:
                WJets_Hists.append(h_temp1)
            WJets_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif 'DYJetsToLL_M-50' in file or 'DYJetsToLL_Pt' in file or 'DY1JetsToLL' in file or 'DY2JetsToLL' in file or 'DYJetsToLL_Pt' in file:
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*luminosity)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                DYJets_Hists.append(h_temp2)
            else:
                DYJets_Hists.append(h_temp1)
            DYJets_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif 'ZJetsToNuNu' in file or 'Z1JetsToNuNu' in file or 'Z2JetsToNuNu' in file or 'Z2JetsToNuNU' in file or 'DYJetsToNuNu_PtZ' in file:
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*luminosity)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                ZJets_Hists.append(h_temp2)
            else:
                ZJets_Hists.append(h_temp1)
            ZJets_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif 'GJets_HT' in file:
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*xsec)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                GJets_Hists.append(h_temp2)
            else:
                GJets_Hists.append(h_temp1)
            GJets_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif ('WWTo' in file) or ('WZTo' in file) or ('ZZTo' in file) or ('WW_' in file) or ('ZZ_' in file) or ('WZ_' in file):
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*luminosity)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                DIBOSON_Hists.append(h_temp2)
            else:
                DIBOSON_Hists.append(h_temp1)
            DIBOSON_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif ('ST_t' in file) or ('ST_s' in file):
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*luminosity)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                STop_Hists.append(h_temp2)
            else:
                STop_Hists.append(h_temp1)
            STop_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif ('TTTo' in file) or ('TT_TuneCUETP8M2T4' in file):
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*luminosity)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                Top_Hists.append(h_temp2)
            else:
                Top_Hists.append(h_temp1)
            Top_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif ('QCD_HT' in file) or ('QCD_bEnriched_HT' in file or ('QCD' in file and 'BGenFilter' in file)):
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*luminosity)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                QCD_Hists.append(h_temp2)
            else:
                QCD_Hists.append(h_temp1)
            QCD_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif 'HToBB' in file:
            xsec = sample_xsec.getXsec(file, options.year)
            # print ('file', file ,'xsec', xsec,'total_events',total_events,'\n')
            if (total_events > 0):
                normlisation = (xsec*luminosity)/(total_events)
            else:
                normlisation = 0
            h_temp.Scale(normlisation)
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                SMH_Hists.append(h_temp2)
            else:
                SMH_Hists.append(h_temp1)
            SMH_Hists_meanWeight.append(h_meanWeight.Integral())
            # print('Yield: ', h_temp.Integral())
        elif 'combined_data_MET' in file:
            h_temp1 = setVarBin(h_temp,hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                MET_Hist.append(h_temp2)
            else:
                MET_Hist.append(h_temp1)
        elif 'combined_data_SE' in file:
            h_temp1 = setVarBin(h_temp, hist)
            if isrebin:
                h_temp2 = setHistStyle(h_temp1, hist, Rebin)
                SE_Hist.append(h_temp2)
            else:
                SE_Hist.append(h_temp1)
        count += 1
    ###====================================== add all the histograms regional based ======================================
    for i in range(len(WJets_Hists)):
        if i == 0:
            WJets = WJets_Hists[i]
        else:
            WJets.Add(WJets_Hists[i])
    # print('WJets_Hists_meanWeight', WJets_Hists_meanWeight)
    WJets = addErrorForEmptyBins(WJets, numpy.nanmean(WJets_Hists_meanWeight))
    WJets.Sumw2()
    for i in range(len(DYJets_Hists)):
        if i == 0:
            DYJets = DYJets_Hists[i]
        else:
            DYJets.Add(DYJets_Hists[i])
    # print('DYJets_Hists_meanWeight', DYJets_Hists_meanWeight)
    DYJets = addErrorForEmptyBins(DYJets, numpy.nanmean(DYJets_Hists_meanWeight))
    DYJets.Sumw2()
    for i in range(len(ZJets_Hists)):
        if i == 0:
            ZJets = ZJets_Hists[i]
        else:
            ZJets.Add(ZJets_Hists[i])
    # print('ZJets_Hists_meanWeight', ZJets_Hists_meanWeight)
    ZJets = addErrorForEmptyBins(ZJets, numpy.nanmean(ZJets_Hists_meanWeight))
    ZJets.Sumw2()
    for i in range(len(GJets_Hists)):
        if i == 0:
            GJets = GJets_Hists[i]
        else:
            GJets.Add(GJets_Hists[i])
    # print('GJets_Hists_meanWeight', GJets_Hists_meanWeight)
    GJets = addErrorForEmptyBins(GJets, numpy.nanmean(GJets_Hists_meanWeight))
    GJets.Sumw2()
    for i in range(len(DIBOSON_Hists)):
        if i == 0:
            DIBOSON = DIBOSON_Hists[i]
        else:
            DIBOSON.Add(DIBOSON_Hists[i])
    # print('DIBOSON_Hists_meanWeight', DIBOSON_Hists_meanWeight)
    DIBOSON = addErrorForEmptyBins(DIBOSON, numpy.nanmean(DIBOSON_Hists_meanWeight))
    DIBOSON.Sumw2()
    for i in range(len(STop_Hists)):
        if i == 0:
            STop = STop_Hists[i]
        else:
            STop.Add(STop_Hists[i])
    # print('STop_Hists_meanWeight', STop_Hists_meanWeight)
    STop = addErrorForEmptyBins(STop, numpy.nanmean(STop_Hists_meanWeight))
    STop.Sumw2()
    for i in range(len(Top_Hists)):
        if i == 0:
            Top = Top_Hists[i]
        else:
            Top.Add(Top_Hists[i])
    # print('Top_Hists_meanWeight', Top_Hists_meanWeight)
    Top = addErrorForEmptyBins(Top, numpy.nanmean(Top_Hists_meanWeight))
    Top.Sumw2()
    # if False:
    if (('ctsValue' in limit_varSR or 'ctsValue' in limit_varCR) and ('2b' in hist or '3j' in hist)) or (('MET' in limit_varSR or 'Recoil' in limit_varCR) and ('1b' in hist or '2j' in hist)):
    # if ('ctsValue' in limit_varSR and 'SR_2b' in hist ) or ('MET' in limit_varSR and 'SR_1b' in hist ):
        qcdFile = ROOT.TFile('qcdForSignal/step3_fitQCD_binwise_'+options.year+'.root',"READ")
        if 'SR' in hist:
            reg_ = reg.replace("SR","QCDbCR")
            qcdhist_temp = qcdFile.Get('qcd_'+reg_+'_sigReg')
        else:
            reg_ = reg.replace("CR","QCDCR")
            qcdhist_temp = qcdFile.Get('qcd_'+reg_+'_sigReg')
        QCD = qcdhist_temp.Clone()
        QCD.Sumw2()
    else:
        for i in range(len(QCD_Hists)):
            if i == 0:
                QCD = QCD_Hists[i]
            else:
                QCD.Add(QCD_Hists[i])
        # print('QCD_Hists_meanWeight', QCD_Hists_meanWeight)
        QCD = addErrorForEmptyBins(QCD, numpy.nanmean(QCD_Hists_meanWeight))
        QCD.Sumw2()
    for i in range(len(SMH_Hists)):
        if i == 0:
            SMH = SMH_Hists[i]
        else:
            SMH.Add(SMH_Hists[i])
    # print('SMH_Hists_meanWeight', SMH_Hists_meanWeight)
    SMH = addErrorForEmptyBins(SMH, numpy.nanmean(SMH_Hists_meanWeight))
    SMH.Sumw2()
    ##=================================================================
    ZJetsCount = ZJets.Integral()
    DYJetsCount = DYJets.Integral()
    WJetsCount = WJets.Integral()
    STopCount = STop.Integral()
    GJetsCount = GJets.Integral()
    TTCount = Top.Integral()
    VVCount = DIBOSON.Integral()
    QCDCount = QCD.Integral()
    SMHCount = SMH.Integral()

    print('ZJetsCount', 'DYJetsCount', 'WJetsCount', 'STopCount', 'GJetsCount', 'TTCount', 'VVCount', 'QCDCount')
    print(ZJetsCount , DYJetsCount , WJetsCount , STopCount , GJetsCount , TTCount , VVCount , QCDCount)
    mcsum = ZJetsCount + DYJetsCount + WJetsCount + STopCount + GJetsCount + TTCount + VVCount + QCDCount
    print('mcsum', mcsum)
    total_hists = WJets_Hists + DYJets_Hists + ZJets_Hists + GJets_Hists + DIBOSON_Hists + STop_Hists + Top_Hists + QCD_Hists

    if '_cutFlow' not in str(hist):
        for histo in total_hists:
            histo = set_overflow(histo)

    ROOT.gStyle.SetHistTopMargin(0.1)

    #============== CANVAS DECLARATION ===================
    #c12 = ROOT.TCanvas("Hist", "Hist", 0,0,1000,1000)
    c12 = myCanvas1D()

    #==================Stack==============================
    hs = ROOT.THStack("hs", " ")

    #============Colors for Histos
    DYJets.SetFillColor(ROOT.kGreen+1)
    DYJets.SetLineWidth(0)
    ZJets.SetFillColor(ROOT.kAzure-4)
    ZJets.SetLineWidth(0)
    DIBOSON.SetFillColor(ROOT.kBlue+1)
    DIBOSON.SetLineWidth(0)
    Top.SetFillColor(ROOT.kOrange-1)
    Top.SetLineWidth(0)
    WJets.SetFillColor(ROOT.kViolet-2)
    WJets.SetLineWidth(0)
    STop.SetFillColor(ROOT.kOrange+2)
    STop.SetLineWidth(0)
    GJets.SetFillColor(ROOT.kCyan-8)
    GJets.SetLineWidth(0)
    QCD.SetFillColor(ROOT.kGray+2)
    QCD.SetLineWidth(0)
    SMH.SetFillColor(ROOT.kRed-1)
    SMH.SetLineWidth(0)

    #=====================Stack all the histogram =========================

    ZJetsCount = ZJets.Integral()
    DYJetsCount = DYJets.Integral()
    WJetsCount = WJets.Integral()
    STopCount = STop.Integral()
    GJetsCount = GJets.Integral()
    TTCount = Top.Integral()
    VVCount = DIBOSON.Integral()
    QCDCount = QCD.Integral()
    SMHCount = SMH.Integral()
    counts_ = {ZJetsCount:ZJets, DYJetsCount:DYJets, WJetsCount:WJets, STopCount:STop, GJetsCount:GJets, TTCount:Top, VVCount:DIBOSON, QCDCount:QCD, SMHCount:SMH}
    # if 'QCD' in hist or '1b' in hist:
    #     counts_ = {ZJetsCount:ZJets, DYJetsCount:DYJets, WJetsCount:WJets, STopCount:STop, GJetsCount:GJets, TTCount:Top, VVCount:DIBOSON, QCDCount:QCD, SMHCount:SMH}
    # else:
    #     counts_ = {ZJetsCount:ZJets, DYJetsCount:DYJets, WJetsCount:WJets, STopCount:STop, GJetsCount:GJets, TTCount:Top, VVCount:DIBOSON, SMHCount:SMH}
    counts_sort = { key:counts_[key] for key in sorted(counts_)}
    for key in counts_sort:
        if key > 0:
            hs.Add(counts_sort[key],"hist")

    hasNoEvents = False

    Stackhist = hs.GetStack().Last()
    maxi = Stackhist.GetMaximum()
    Stackhist.SetLineWidth(2)
    if (Stackhist.Integral() == 0):
        hasNoEvents = True
        print('No events found! for '+hist+'\n')

    # =====================histogram for systematic/ statistical uncertainty ========================
    h_stat_err = Stackhist.Clone("h_stat_err")
    h_stat_err.Sumw2()
    h_stat_err.SetFillColor(ROOT.kGray+3)
    h_stat_err.SetLineColor(ROOT.kGray+3)
    h_stat_err.SetMarkerSize(0)
    h_stat_err.SetFillStyle(3013)
    h_stat_syst_err = h_stat_err.Clone("h_stat_syst_err")

    # =============================================

    if(NORATIOPLOT):
        c1_2 = ROOT.TPad("c1_2", "newpad", 0, 0.05, 1, 1)  # 0.993)
        c1_2.SetRightMargin(0.06)
    else:
        c1_2 = ROOT.TPad("c1_2", "newpad", 0, 0.20, 1, 1)
    c1_2.SetBottomMargin(0.09)
    c1_2.SetTopMargin(0.08)
    c1_2.SetLeftMargin(0.12)
    c1_2.SetRightMargin(0.06)
    c1_2.SetLogy(ISLOG)
    c1_2.Draw()
    c1_2.cd()
    for h in hs: h = SetCMSAxis(h)
    hs.Draw()
    if makeMuCHplots:
        noYieldHisto = bool(('weight' in hist) or ('_up' in hist)
                                or ('_down' in hist) or ('dPhiTrk' in hist) or ('dPhiCalo' in hist) or ('rJet1Pt' in hist))
    elif makeEleCHplots:
        noYieldHisto = bool(('weight' in hist) or ('_up' in hist)
                            or ('_down' in hist))
    if makeSIGplots:
        # if ('_'+limit_varSR in hist or 'cutFlow' in hist or 'nJets' in hist) and ('SR' in hist or 'preselR' in hist) and not noYieldHisto:
        if ('_'+limit_varSR in hist or 'cutFlow' in hist or 'nJets' in hist or 'nBJets' in hist) and ('SR' in hist or 'preselR' in hist) and not noYieldHisto:
        # if ('nJets' in hist) and ('SR' in hist):
            # how many signal points you want to include
            #ma_points = [10,50,100, 150, 200, 250,300, 350, 400, 500, 700, 750,1000]
            # ma_points = [10,50,100,150, 200, 250,300,350,400,450,500]
            ma_points = [50,150,350,500]
            mA_point = 600
            sig_leg = SetLegend([.15, .60, .45, .75], ncol=1)
            sig_leg.SetTextSize(0.030)
            sig_leg.SetTextFont(62)
            sig_leg.SetHeader("2HDM+a model")
            if runOn2016:
                signal_files_name = [name for name in os.listdir(sig_path) for mapoint in ma_points if 'Ma'+str(mapoint)+'_' in name and 'MA'+str(mA_point)+'_' in name]
                signal_files_name = sorted(signal_files_name, key=lambda item: (int(item.split('_')[4].strip('Ma')) if item.split('_')[4].strip('Ma').isdigit() else float('inf'), item))
                signal_files = {}
                for name in signal_files_name:
                    signal_files.update({name:ROOT.TFile(sig_path+'/'+name, 'READ')})
                total = {}
                sig_hist = {}
                sig_hist_list = []
                for key in signal_files:
                    total.update({key:signal_files[key].Get('h_total_mcweight')})
                    sig_hist.update({key: setVarBin(signal_files[key].Get(hist),hist)})
                    sig_hist[key].Scale(luminosity*sig_sample_xsec.getSigXsec_official(key)/total[key].Integral())
                    # sig_hist[key].Scale(luminosity*sig_sample_xsec.getSigXsec_150(key)/total[key].Integral())
            if runOn2017 or runOn2018:
                signal_files_name = [name for name in os.listdir(sig_path) for mapoint in ma_points if 'ma_'+str(mapoint)+'_' in name and 'mA_'+str(mA_point) in name]
                signal_files_name = sorted(signal_files_name, key=lambda item: (int(item.split('_')[-3]) if item.split('_')[-3].isdigit() else float('inf'), item))
                signal_files = {}
                for name in signal_files_name:
                    signal_files.update({name:ROOT.TFile(sig_path+'/'+name, 'READ')})
                total = {}
                sig_hist = {}
                sig_hist_list = []
                for key in signal_files:
                    total.update({key:signal_files[key].Get('h_total_mcweight')})
                    sig_hist.update({key: setVarBin(signal_files[key].Get(hist),hist)})
                    sig_hist_list.append(sig_hist[key].Scale(luminosity*sig_sample_xsec.getSigXsec_official(key)/total[key].Integral()))

            [(sig_hist[i].SetLineStyle(n), sig_hist[i].SetLineWidth(6), sig_hist[i].SetLineColor(n)) for i, n in zip(sig_hist, range(2, len(sig_hist)+2))]
            [(sig_hist[i].SetMarkerColor(n), sig_hist[i].SetMarkerStyle(n), sig_hist[i].SetMarkerSize(1.1)) for i, n in zip(sig_hist, range(2, len(sig_hist)+2))]
            if runOn2016:
                [sig_leg.AddEntry(sig_hist[his_list], "ma = "+filename.split('_')[4].strip('Ma')+" GeV, mA = "+filename.split('_')[6].strip('MA')+" GeV", "lp") for his_list, filename in zip(sig_hist, signal_files_name)]
            if runOn2017 or runOn2018:
                [sig_leg.AddEntry(sig_hist[his_list], "ma = "+filename.split('_')[-3]+" GeV, mA = "+filename.split('_')[-1].strip('.root')+" GeV", "lp") for his_list, filename in zip(sig_hist, signal_files_name)]
            # sqbkg = Stackhist.Clone()
            # # Loop over the signal histograms
            # for i, (key, value) in enumerate(sig_hist.items()):
            #     # Create a new histogram for the result
            #     result_hist = sig_hist[key].Clone(f"result_hist_{i}")
            #     result_hist.Reset()

            #     # Calculate s/sqrt(s+b) for each bin
            #     for j in range(1, sig_hist[key].GetNbinsX() + 1):
            #         s_val = sig_hist[key].GetBinContent(j)
            #         b_val = sqbkg.GetBinContent(j)
            #         sqrt_sb = ROOT.TMath.Sqrt(s_val + b_val)
            #         if sqrt_sb == 0:
            #             result_val = 0
            #         else:
            #             result_val = s_val / sqrt_sb
            #         result_hist.SetBinContent(j, result_val)
            #     sig_hist[key] = result_hist

            # print([sig_hist[i].Integral() for i in sig_hist])
            # [sig_hist[i].Scale(5) for i in sig_hist]
            [sig_hist[i].Draw("same Ehist") for i in sig_hist]
            sig_leg.Draw('same')
#####================================= data section =========================
    if 'SR' in reg:
        if datainSR:
            h_data = MET_Hist[0]
        else:
            h_data = hs.GetStack().Last()
    else:
        if dtset == "SE":
            h_data = SE_Hist[0]
        elif dtset == "MET":
            h_data = MET_Hist[0]
    # h_data = hs.GetStack().Last()
    h_data.Sumw2()
    h_data.SetLineColor(1)
    h_data.SetLineWidth(2)
    h_data.SetMarkerSize(1.3)
    h_data.SetMarkerStyle(20)
    h_data = SetCMSAxis(h_data)
    if '_cutFlow' not in str(hist):
        h_data = set_overflow(h_data)
    if(not NORATIOPLOT):
        h_data.Draw("same p e1")
    if (ISLOG):
        if '_cutFlow' in str(hist):
            hs.SetMaximum(1000000000)
            hs.SetMinimum(100)
        else:
            hs.SetMaximum(maxi * 1000)
            hs.SetMinimum(0.1)
    else:
        hs.SetMaximum(maxi * 1.75)
        hs.SetMinimum(0)
    # print ('Data Integral',h_data.Integral())
    ##=============================== hs setting section =====================
    #
    if (not hasNoEvents):
        hs.GetXaxis().SetNdivisions(508)
        if(NORATIOPLOT):
            hs.GetXaxis().SetTitleOffset(1.05)
            hs.GetXaxis().SetTitleFont(42)
            hs.GetXaxis().SetLabelFont(42)
            hs.GetXaxis().SetLabelSize(.03)
            hs.GetXaxis().SetTitle(str(titleX))
            hs.GetXaxis().SetTitleFont(42)
            hs.GetXaxis().SetLabelOffset(.01)
            hs.GetYaxis().SetTitleOffset(0.7)
            hs.GetYaxis().SetTitle("Events/bin")
            hs.GetYaxis().SetTitleSize(0.08)
            hs.GetYaxis().SetTitleFont(42)
            hs.GetYaxis().SetLabelFont(42)
            hs.GetYaxis().SetLabelSize(.04)
        else:
            hs.GetXaxis().SetTitle(str(titleX))
            hs.GetXaxis().SetTitleOffset(0.00)
            hs.GetXaxis().SetTitleFont(42)
            hs.GetXaxis().SetTitleSize(0.05)
            hs.GetXaxis().SetLabelFont(42)
            hs.GetXaxis().SetLabelOffset(.01)
            hs.GetXaxis().SetLabelSize(0.04)
            hs.GetYaxis().SetTitle("Events/bin")
            hs.GetYaxis().SetTitleSize(0.08)
            hs.GetYaxis().SetTitleOffset(0.7)
            hs.GetYaxis().SetTitleFont(42)
            hs.GetYaxis().SetLabelFont(42)
            hs.GetYaxis().SetLabelSize(.05)

        # if not isrebin:
        hs.GetXaxis().SetRangeUser(XMIN, XMAX)
        hs.GetXaxis().SetNdivisions(508)

    # print([(h.GetNbinsX(), h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax()) for h in hs ])

    #=============================  legend section =========================================
    DYLegend = "Z(ll)+jets "
    WLegend = "W(l#nu)+jets "
    GLegend = "#gamma+jets "
    ZLegend = "Z(#nu#nu)+jets "
    STLegend = "Single t "
    TTLegend = "t#bar{t} "
    VVLegend = "WW/WZ/ZZ "
    QCDLegend = "QCD "
    SMHLegend = "SMH "

    legend = SetLegend([.50, .58, .93, .92], ncol=2)

    if(not NORATIOPLOT):
        if 'SR' in reg:
            if datainSR:
                legend.AddEntry(h_data, "Data", "PEL")
            else:
                legend.AddEntry(h_data, "bkgSum", "PEL")
        else:
            legend.AddEntry(h_data, "Data", "PEL")
    legend.AddEntry(Top, TTLegend, "f")
    legend.AddEntry(STop, STLegend, "f")
    legend.AddEntry(WJets, WLegend, "f")
    legend.AddEntry(DIBOSON, VVLegend, "f")
    if GJetsCount > 0:
        legend.AddEntry(GJets, GLegend, "f")
    if ZJetsCount > 0:
        legend.AddEntry(ZJets, ZLegend, "f")
    legend.AddEntry(DYJets, DYLegend, "f")
    # if 'QCD' in hist or '1b' in hist:
    #     legend.AddEntry(QCD, QCDLegend, "f")
    legend.AddEntry(QCD, QCDLegend, "f")
    # if QCDCount > 0:
    #     legend.AddEntry(QCD, QCDLegend, "f")
    legend.AddEntry(SMH, SMHLegend, "f")

    legend.Draw('same')

    #=================================================latex section =====================
    t2d = ExtraText(str(histolabel), 0.20, 0.80)
    t2d.SetTextSize(0.06)

    t2d.SetTextAlign(12)
    t2d.SetNDC(ROOT.kTRUE)
    t2d.SetTextFont(42)
    t2d.Draw("same")

    # pt = drawenergy1D(True, text_="Internal", data=True)
    pt = drawenergy1D(True, text_="Work in progress", data=True)
    for ipt in pt:
        ipt.Draw()
    #======================================== ratio log ================

    ratioleg = SetLegend([.72, .80, .90, .90], 1)
    ratioleg.SetTextSize(0.15)

    #============================================= statistical error section ======================

    ratiostaterr = h_stat_err.Clone("ratiostaterr")
    ratiostaterr.Sumw2()
    ratiostaterr.SetStats(0)
    ratiostaterr.SetMinimum(0)
    ratiostaterr.SetMarkerSize()
    ratiostaterr.SetFillColor(ROOT.kBlack)
    ratiostaterr.SetFillStyle(3013)
    for i in range(h_stat_err.GetNbinsX()+2):
        ratiostaterr.SetBinContent(i, 0.0)
        if (h_stat_err.GetBinContent(i) > 1e-6):
            binerror = h_stat_err.GetBinError(i)/h_stat_err.GetBinContent(i)
            ratiostaterr.SetBinError(i, binerror)
            # h_stat_err.SetBinError(i, binerror)
        else:
            ratiostaterr.SetBinError(i, 999.)
            # h_stat_err.SetBinError(i, 999.)
    #============================================= systematic error section ======================
    if limit_varSR in hist or limit_varCR in hist:
        ratiosysterr = h_stat_err.Clone("ratiosysterr")
        ratiosysterr.Sumw2()
        ratiosysterr.SetStats(0)
        ratiosysterr.SetMinimum(0)
        ratiosysterr.SetMarkerSize(0)
        ratiosysterr.SetFillColor(ROOT.kBlack)
        ratiosysterr.SetFillStyle(3013)
        if 'SR' in reg and '_'+limit_varSR in hist:
            main_var = '_'+limit_varSR
            for i in range(h_stat_err.GetNbinsX()):
                binerror2 = 0.0
                ratiosysterr.SetBinContent(i, 0.0)
                if (h_stat_err.GetBinContent(i) > 1e-6):
                    binerror2 = (pow(h_stat_err.GetBinError(i), 2)
                                # + pow(syst_dict['CMSyear_fake_b_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_eff_b_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_trig_met_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # # + pow(syst_dict['CMSyear_Top_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_PU_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECAbsolute_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECAbsolute_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECBBEC1_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECBBEC1_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECEC2_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECEC2_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECFlavorQCD_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECHF_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECHF_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECRelativeBal_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECRelativeSample_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # # + pow(syst_dict['Res_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['En_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                )
                    binerror = math.sqrt(binerror2)
                    ratiosysterr.SetBinError(
                        i, binerror/h_stat_err.GetBinContent(i))
                    h_stat_syst_err.SetBinError(
                        i, binerror/h_stat_err.GetBinContent(i))
                else:
                    ratiosysterr.SetBinError(i, 999.)
                    h_stat_syst_err.SetBinError(i, 999.)
        elif 'CR' in reg and '_'+limit_varCR in hist:
            main_var = '_'+limit_varCR
            for i in range(1, h_stat_err.GetNbinsX()+1):
                binerror2 = 0.0
                ratiosysterr.SetBinContent(i, 0.0)
                if (h_stat_err.GetBinContent(i) > 1e-6):
                    binerror2 = (pow(h_stat_err.GetBinError(i), 2)
                                # + pow(syst_dict['CMSyear_fake_b_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_eff_b_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # # + pow(syst_dict['CMSyear_Top_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_trig_met_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_PU_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECAbsolute_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECAbsolute_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECBBEC1_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECBBEC1_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECEC2_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECEC2_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECFlavorQCD_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECHF_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECHF_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECRelativeBal_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['JECRelativeSample_year_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # # + pow(syst_dict['Res_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['En_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_trig_ele_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_EleID_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_EleRECO_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_MuID_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_MuISO_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                # + pow(syst_dict['CMSyear_MuTRK_syst_'+reg+main_var][i-1]*h_stat_err.GetBinContent(i), 2)
                                )
                    binerror = math.sqrt(binerror2)
                    ratiosysterr.SetBinError(i, binerror/h_stat_err.GetBinContent(i))
                    h_stat_syst_err.SetBinError(i, binerror/h_stat_err.GetBinContent(i))
                else:
                    ratiosysterr.SetBinError(i, 999.)
                    h_stat_syst_err.SetBinError(i, 999.)
    if limit_varSR in hist or limit_varCR in hist :
        # ratioleg.AddEntry(ratiosysterr, "stat + syst", "f")
        ratioleg.AddEntry(ratiosysterr, "stat", "f")
    else:
        ratioleg.AddEntry(ratiostaterr, "stat", "f")

    if(not NORATIOPLOT):
        if limit_varSR in hist or limit_varCR in hist :
            h_stat_err.Draw("same E2")
        else:
            h_stat_syst_err.Draw("same E2")

    #============================================= Lower Tpad Decalaration ====================================
    if(not NORATIOPLOT):
        c12.cd()
        DataMC = h_data.Clone()
        DataMC.Add(Stackhist, -1)    # remove for data/mc
        #DataMCPre = h_data.Clone()
        DataMC.Divide(Stackhist)
        DataMC.GetYaxis().SetTitle("#frac{Data-Pred}{Pred}")
        DataMC.GetYaxis().SetTitleSize(0.12)
        DataMC.GetYaxis().SetTitleOffset(0.42)
        DataMC.GetYaxis().SetTitleFont(42)
        DataMC.GetYaxis().SetLabelSize(0.12)
        DataMC.GetYaxis().CenterTitle()
        DataMC.GetXaxis().SetTitle(str(titleX))
        DataMC.GetXaxis().SetLabelSize(0.14)
        DataMC.GetXaxis().SetTitleSize(0.16)
        DataMC.GetXaxis().SetTitleOffset(1)
        DataMC.GetXaxis().SetTitleFont(42)
        DataMC.GetXaxis().SetTickLength(0.07)
        DataMC.GetXaxis().SetLabelFont(42)
        DataMC.GetYaxis().SetLabelFont(42)

    c1_1 = ROOT.TPad("c1_1", "newpad", 0, 0.00, 1, 0.3)
    if (not NORATIOPLOT):
        c1_1.Draw()
    c1_1.cd()
    c1_1.Range(-7.862408, -629.6193, 53.07125, 486.5489)
    c1_1.SetFillColor(0)
    c1_1.SetTicky(1)
    c1_1.SetLeftMargin(0.12)
    c1_1.SetRightMargin(0.06)
    c1_1.SetTopMargin(0.00)
    c1_1.SetBottomMargin(0.42)
    c1_1.SetFrameFillStyle(0)
    c1_1.SetFrameBorderMode(0)
    c1_1.SetFrameFillStyle(0)
    c1_1.SetFrameBorderMode(0)
    c1_1.SetLogy(0)

    if(not NORATIOPLOT):
        if (0):  # if(VARIABLEBINS)
            c1_1.SetLogx(0)
            DataMC.GetXaxis().SetMoreLogLabels()
            DataMC.GetXaxis().SetNoExponent()
            DataMC.GetXaxis().SetNdivisions(508)
        DataMC.GetXaxis().SetRangeUser(XMIN, XMAX)
        DataMC.SetMarkerSize(1.5)
        DataMC.SetMarkerStyle(20)
        DataMC.SetMarkerColor(1)
        if 'QCD' in hist:
            DataMC.SetMinimum(-1.18)
            DataMC.SetMaximum(1.18)
        else:
            DataMC.SetMinimum(-0.68)
            DataMC.SetMaximum(0.68)
        DataMC.GetXaxis().SetNdivisions(508)
        DataMC.GetYaxis().SetNdivisions(505)
        DataMC.Draw("P e1")
        if limit_varSR in hist or limit_varCR in hist:
            ratiosysterr.Draw("e2 same")
        else:
            ratiostaterr.Draw("e2 same")
        DataMC.Draw("P e1 same")
        # line1 = ROOT.TLine(XMIN, 0.2, XMAX, 0.2)
        # line2 = ROOT.TLine(XMIN, -0.2, XMAX, -0.2)
        # line1.SetLineStyle(2)
        # line1.SetLineColor(2)
        # line1.SetLineWidth(2)
        # line2.SetLineStyle(2)
        # line2.SetLineColor(2)
        # line2.SetLineWidth(2)
        # line1.Draw("same")
        # line2.Draw("same")
        ratioleg.Draw("same")
    # c12.Draw()
    plot = str(hist)
    noPdfPng = True
    if ('Up' in str(hist) or 'Down' in str(hist) or 'CMSyear' in str(hist)):
        noPdfPng = False
    if not os.path.exists('plots_norm/'+histVersion+'/bbDMPng/'+reg):
        os.makedirs('plots_norm/'+histVersion+'/bbDMPng/'+reg)
    if not os.path.exists('plots_norm/'+histVersion+'/bbDMPdf/'+reg):
        os.makedirs('plots_norm/'+histVersion+'/bbDMPdf/'+reg)
    if not os.path.exists('plots_norm/'+histVersion+'/bbDMRoot/'):
        os.makedirs('plots_norm/'+histVersion+'/bbDMRoot/')
    if not os.path.exists('plots_norm/'+histVersion+'/bbDMTxt/'+reg):
        os.makedirs('plots_norm/'+histVersion+'/bbDMTxt/'+reg)
    # if not datainSR:
    if datainSR:
        if (ISLOG == 0) and noPdfPng:
            c12.SaveAs('plots_norm/'+histVersion+'/bbDMPdf/'+reg+'/'+plot+'.pdf')
            c12.SaveAs('plots_norm/'+histVersion+'/bbDMPng/'+reg+'/'+plot+'.png')
            hist_yield = open('plots_norm/'+histVersion+'/bbDMTxt/'+reg+'/'+plot+'.txt','w')
            print("Saved. \n")
        if (ISLOG == 1) and noPdfPng:
            c12.SaveAs('plots_norm/'+histVersion+'/bbDMPdf/'+reg+'/'+plot+'_log.pdf')
            c12.SaveAs('plots_norm/'+histVersion+'/bbDMPng/'+reg+'/'+plot+'_log.png')
            hist_yield = open('plots_norm/'+histVersion+'/bbDMTxt/'+reg+'/'+plot+'_log.txt','w')
    fshape = ROOT.TFile('plots_norm/'+histVersion+'/bbDMRoot/'+plot+'.root', "RECREATE")
    fshape.cd()
    Stackhist.SetNameTitle("bkgSum", "bkgSum")
    Stackhist.Write()
    DIBOSON.SetNameTitle("DIBOSON", "DIBOSON")
    DIBOSON.Write()
    ZJets.SetNameTitle("ZJets", "ZJets")
    ZJets.Write()
    GJets.SetNameTitle("GJets", "GJets")
    GJets.Write()
    QCD.SetNameTitle("QCD", "QCD")
    QCD.Write()
    SMH.SetNameTitle("SMH", "SMH")
    SMH.Write()
    STop.SetNameTitle("STop", "STop")
    STop.Write()
    Top.SetNameTitle("Top", "Top")
    Top.Write()
    WJets.SetNameTitle("WJets", "WJets")
    WJets.Write()
    DYJets.SetNameTitle("DYJets", "DYJets")
    DYJets.Write()
    data_obs = h_data
    data_obs.SetNameTitle("data_obs", "data_obs")
    data_obs.Write()
    # bkg_list = { 'Top': Top, 'STop': STop, 'WJets': WJets, 'DIBOSON': DIBOSON, 'GJets': GJets, 'ZJets': ZJets, 'DYJets': DYJets, 'QCD': QCD, 'SMH': SMH,'Total_Bkg': Stackhist,'data_obs': h_data}
    bkg_list = { 'Top': Top, 'STop': STop, 'WJets': WJets, 'DIBOSON': DIBOSON, 'GJets': GJets, 'ZJets': ZJets, 'DYJets': DYJets, 'SMH': SMH,'QCD': QCD,'Total_Bkg': Stackhist,'data_obs': h_data}
    bkg_list_name = { 'Top': "$t\\bar{t}$", 'STop': "Single-$t$", 'WJets': "W(l$\\nu$)+jets", 'DIBOSON':"WW/WZ/ZZ", 'GJets': "$\\gamma$+jets", 'ZJets': "Z($\\nu\\nu$)+jets", 'DYJets': "Z(ll)+jets", 'QCD': "QCD", 'SMH': "SM-H" ,'Total_Bkg': "Total_Bkg",'data_obs': "data_obs",}

    # if noPdfPng and not datainSR:
    if noPdfPng and datainSR:
        for key in bkg_list:
            hist_yield.write(str(bkg_list_name[key])+'    '+str.format('{0:.2f}', bkg_list[key].GetBinContent(1))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(1))+'    '+str.format('{0:.2f}', bkg_list[key].GetBinContent(2))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(2))+'    '+str.format('{0:.2f}', bkg_list[key].GetBinContent(3))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(3))+'    '+str.format('{0:.2f}', bkg_list[key].GetBinContent(4))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(4))+'\n')
        hist_yield.close()
    # if makeSIGplots and ('_'+limit_varSR in hist) and ('SR' in hist) and not noYieldHisto and ('SR_1b' in reg or 'SR_2b' in reg):
    if makeSIGplots and ('_'+limit_varSR in hist or 'cutFlow' in hist or 'nJets' in hist or 'nBJets' in hist) and ('SR' in hist or 'preselR' in hist) and not noYieldHisto:
    # if makeSIGplots and ('nJets' in hist) and ('SR' in hist) and ('SR_1b' in reg or 'SR_2b' in reg):
        if runOn2016: [sig_hist[h_key].SetNameTitle('ma_'+h_key.split('5f_Ma')[-1].split('_MChi1')[0]+'_mA_'+h_key.split('MChi1_MA')[-1].split('_tanb35')[0],'ma_'+h_key.split('5f_Ma')[-1].split('_MChi1')[0]+'_mA_'+h_key.split('MChi1_MA')[-1].split('_tanb35')[0]) for h_key in sig_hist]
        else: [sig_hist[h_key].SetNameTitle(h_key.partition('_pythia8_')[-1].partition('.root')[0], h_key.partition('_pythia8_')[-1].partition('.root')[0]) for h_key in  sig_hist]
        [sig_hist[h_key].Write() for h_key in sig_hist]
    fshape.Write()
    fshape.Close()
    c12.Close()
    print('\n')
    syst_Unc = ('Up' not in hist) or ('Down' not in hist)
    if (('_'+limit_varSR in hist and 'SR' in hist ) or (limit_varCR in hist) or ('_'+limit_varSR in hist and 'QCD' in hist )) and syst_Unc and not noYieldHisto:
        if 'SR_1b' in str(hist): reg_name = 'SR-(1b) nJet > 2'
        elif 'SR_2b' in str(hist): reg_name = 'SR-(2b) nJet > 3'
        elif 'ZmumuCR_1b' in str(hist): reg_name = 'Z$\\mu\\mu$-CR-(1b)'
        elif 'ZmumuCR_2b' in str(hist): reg_name = 'Z$\\mu\\mu$-CR-(2b)'
        elif 'ZmumuCR_2j' in str(hist): reg_name = 'Z$\\mu\\mu$-CR-(2j)'
        elif 'ZmumuCR_3j' in str(hist): reg_name = 'Z$\\mu\\mu$-CR-(3j)'
        elif 'TopmunuCR_1b' in str(hist): reg_name = 't$\\bar{t}(\\mu)$-CR-(1b)'
        elif 'TopmunuCR_2b' in str(hist): reg_name = 't$\\bar{t}(\\mu)$-CR-(2b)'
        elif 'WmunuCR_1b' in str(hist): reg_name =  'W-$(\\mu\\nu)$-CR-(1b)'
        elif 'WmunuCR_2b' in str(hist): reg_name = 'W-$(\\mu\\nu)$-CR-(2b)'
        elif 'ZeeCR_1b' in str(hist): reg_name = 'Zee-CR-(1b)'
        elif 'ZeeCR_2b' in str(hist): reg_name = 'Zee-CR-(2b)'
        elif 'ZeeCR_2j' in str(hist): reg_name = 'Zee-CR-(2j)'
        elif 'ZeeCR_3j' in str(hist): reg_name = 'Zee-CR-(3j)'
        elif 'TopenuCR_1b' in str(hist): reg_name = 't$\\bar{t}(e)$-CR-(1b)'
        elif 'TopenuCR_2b' in str(hist): reg_name = 't$\\bar{t}(e)$-CR-(2b)'
        elif 'WenuCR_1b' in str(hist): reg_name = 'W-$(e\\nu)$-CR-(1b)'
        elif 'WenuCR_2b' in str(hist): reg_name = 'W-$(e\\nu)$-CR-(2b)'
        elif 'QCDbCR_1b' in str(hist): reg_name = 'QCD-CR-(1b)'
        elif 'QCDbCR_2b' in str(hist): reg_name = 'QCD-CR-(2b)'
        yield_outfile.write('region '+str(reg_name)+'\n')
        yield_outfile_binwise.write('region '+str(hist)+'\n')
        if makeSIGplots and ('MET' in hist or 'ctsValue' in hist) and ('SR' in hist) and not noYieldHisto and ('SR_1b' in reg or 'SR_2b' in reg):
            for h_key in sig_hist:
                if runOn2016:
                    bkg_list.update({'ma_'+h_key.split('5f_Ma')[-1].split('_MChi1')[0]+'_mA_'+h_key.split('MChi1_MA')[-1].split('_tanb35')[0]: sig_hist[h_key]})
                    bkg_list_name.update({'ma_'+h_key.split('5f_Ma')[-1].split('_MChi1')[0]+'_mA_'+h_key.split('MChi1_MA')[-1].split('_tanb35')[0]:'ma_'+h_key.split('5f_Ma')[-1].split('_MChi1')[0]+'_mA_'+h_key.split('MChi1_MA')[-1].split('_tanb35')[0]})
                else:
                    bkg_list.update({h_key.partition('_pythia8_')[-1].partition('.root')[0]: sig_hist[h_key]})
                    bkg_list_name.update({h_key.partition('_pythia8_')[-1].partition('.root')[0]: h_key.partition('_pythia8_')[-1].partition('.root')[0]})
        yield_outfile_binwise.write('        Bin1    Bin2    Bin3    Bin4\n')
        for key in bkg_list:
            bin_cont = [bkg_list[key].GetBinContent(i)>0.0 for i in range(1,5)]
            if any(bin_cont):
                yield_outfile_binwise.write(str(bkg_list_name[key])+'    '+str.format('{0:.2f}', bkg_list[key].GetBinContent(1))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(1))+'    '+str.format('{0:.2f}', bkg_list[key].GetBinContent(2))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(2))+'    '+str.format('{0:.2f}', bkg_list[key].GetBinContent(3))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(3))+'    '+str.format('{0:.2f}', bkg_list[key].GetBinContent(4))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(4))+'\n')
                # yield_outfile_binwise.write(str(bkg_list_name[key])+'    '+str.format('{0:.3f}', bkg_list[key].GetBinContent(1))+'    '+str.format('{0:.3f}', bkg_list[key].GetBinContent(2))+'    '+str.format('{0:.3f}', bkg_list[key].GetBinContent(3))+'    '+str.format('{0:.3f}', bkg_list[key].GetBinContent(4))+'\n')
        for key in bkg_list:
            binerror = 0.00
            bkg_list[key].Rebin(bkg_list[key].GetNbinsX())
            binerror = (bkg_list[key].GetBinError(1))
            # if bkg_list[key].GetBinContent(1) > 0.0:
            yield_outfile.write(str(bkg_list_name[key])+' '+str.format('{0:.2f}', bkg_list[key].GetBinContent(1))+'\xb1'+str.format('{0:.2f}', bkg_list[key].GetBinError(1))+'\n')
        yield_outfile_binwise.write('\n')
        yield_outfile.write('\n')

#=======================================================================


######################################################################

regions = []
PUreg = []
if makeMuCHplots:
    # regions = ['SR_1b', 'SR_2b']
    regions = ['SR_1b', 'SR_2b', 'ZmumuCR_2j', 'ZmumuCR_3j', 'TopmunuCR_2b', 'WmunuCR_1b',] #'QCDbCR_1b', 'QCDbCR_2b']
    # regions = ['SR_2b']
    # regions = ['WmunuCR_2b', 'WmunuCR_1b','TopmunuCR_1b']
    # regions = ['ZmumuCR_2j', 'ZmumuCR_3j']
    # regions = ['QCDbCR_1b', 'QCDbCR_2b']
    # regions = ['WmunuCR_1b', ]
    # regions = ['SR_2b',  'ZmumuCR_3j', 'TopmunuCR_2b','QCDbCR_2b'    ]
    # regions = ['SR_1b', 'SR_2b','ZmumuCR_1b', 'ZmumuCR_2b']
    # regions = ['preselR']
if makeEleCHplots:
    # regions = ['WenuCR_1b', 'TopenuCR_2b']
    # regions = ['ZeeCR_1b', 'ZeeCR_2b']
    regions = ['ZeeCR_2j', 'ZeeCR_3j', 'TopenuCR_2b', 'WenuCR_1b']
    # regions = ['ZeeCR_2j', 'ZeeCR_3j','TopenuCR_2b','WenuCR_1b', 'WenuCR_2b']
    # regions = ['ZeeCR_2j', 'ZeeCR_3j', 'TopenuCR_1b','TopenuCR_2b','WenuCR_2b']

# makeplot("reg_SR_2b_MET",'h_reg_SR_2b_MET',var_legendSR,250.,1000.,1,1,0,'SR_2b',varBin=False)
# makeplot('reg_SR_2b_nJets', 'h_reg_SR_2b_nJets','nJets', 0., 10., rebin, isLog, 0, 'SR_2b', varBin=False)


allplots = False
systHistoFiles = True
for reg in regions:
    if '_2b' in reg or '_3j' in reg:
        limit_varSR = limit_varCR = 'ctsValue'
        minBin = 0; maxBin = 1
        var_legendCR = var_legendSR = 'cos(#Theta)*'
        rebin = 1; isLog = 1
    elif '_1b' in reg or '_2j' in reg or 'preselR' in reg:
        limit_varSR = 'MET'
        limit_varCR = 'Recoil'
        minBin = 250; maxBin = 1000
        var_legendCR = 'Recoil (GeV)'
        var_legendSR = 'p_{T}^{miss} (GeV)'
        rebin = 1; isLog = 1
    # try:
    if 'SR_' in reg or 'preselR' in reg or 'QCDbCR' in reg:
        makeplot('reg_'+reg+'_'+limit_varSR, 'h_reg_'+reg+'_'+limit_varSR, var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
        # makeplot('reg_'+reg+'_prod_cat', 'h_reg_'+reg+'_prod_cat', 'Jet Flavour', 0, 3, 1, 1, 0, reg, varBin=False)
        makeplot('reg_'+reg+'_min_dPhi', 'h_reg_'+reg+'_min_dPhi', 'min_dPhi', 0.0, 3.2, 1, 1, 0, reg, varBin=False)
        if ('SR_2b' in reg):
            makeplot('reg_'+reg+'_MET', 'h_reg_'+reg+'_MET','p_{T}^{miss} (GeV)', 250, 1000, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nBJets', 'h_reg_'+reg+'_nBJets','nBJets', 0., 10., rebin, 1, 0, reg, varBin=False)
        if systHistoFiles:
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_eff_bUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_eff_bUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_eff_bDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_eff_bDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_fake_bUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_fake_bUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_fake_bDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_fake_bDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_EWKUp', 'h_reg_'+reg+'_'+limit_varSR+'_EWKUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_EWKDown', 'h_reg_'+reg+'_'+limit_varSR+'_EWKDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_TopUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_TopUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_TopDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_TopDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_trig_metUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_trig_metUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_trig_metDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_trig_metDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_PUUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_PUUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_PUDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_PUDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECUp', 'h_reg_'+reg+'_'+limit_varSR+'_JECUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECDown', 'h_reg_'+reg+'_'+limit_varSR+'_JECDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECAbsoluteUp', 'h_reg_'+reg+'_'+limit_varSR+'_JECAbsoluteUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECAbsolute_yearUp', 'h_reg_'+reg + '_'+limit_varSR+'_JECAbsolute_yearUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECBBEC1Up', 'h_reg_'+reg+'_'+limit_varSR+'_JECBBEC1Up', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECBBEC1_yearUp', 'h_reg_'+reg+'_'+limit_varSR+'_JECBBEC1_yearUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECEC2Up', 'h_reg_'+reg+'_'+limit_varSR+'_JECEC2Up', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECEC2_yearUp', 'h_reg_'+reg+'_'+limit_varSR+'_JECEC2_yearUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECFlavorQCDUp', 'h_reg_'+reg+'_'+limit_varSR+'_JECFlavorQCDUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECHFUp', 'h_reg_'+reg+'_'+limit_varSR+'_JECHFUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECHF_yearUp', 'h_reg_'+reg+'_'+limit_varSR+'_JECHF_yearUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECRelativeBalUp', 'h_reg_'+reg+'_'+limit_varSR+'_JECRelativeBalUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECRelativeSample_yearUp', 'h_reg_'+reg + '_'+limit_varSR+'_JECRelativeSample_yearUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECAbsoluteDown', 'h_reg_'+reg+'_'+limit_varSR+'_JECAbsoluteDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECAbsolute_yearDown', 'h_reg_'+reg + '_'+limit_varSR+'_JECAbsolute_yearDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECBBEC1Down', 'h_reg_'+reg+'_'+limit_varSR+'_JECBBEC1Down', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECBBEC1_yearDown', 'h_reg_'+reg + '_'+limit_varSR+'_JECBBEC1_yearDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECEC2Down', 'h_reg_'+reg+'_'+limit_varSR+'_JECEC2Down', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECEC2_yearDown', 'h_reg_'+reg+'_'+limit_varSR+'_JECEC2_yearDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECFlavorQCDDown', 'h_reg_'+reg+'_'+limit_varSR+'_JECFlavorQCDDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECHFDown', 'h_reg_'+reg+'_'+limit_varSR+'_JECHFDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECHF_yearDown', 'h_reg_'+reg+'_'+limit_varSR+'_JECHF_yearDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECRelativeBalDown', 'h_reg_'+reg + '_'+limit_varSR+'_JECRelativeBalDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_JECRelativeSample_yearDown', 'h_reg_'+reg + '_'+limit_varSR+'_JECRelativeSample_yearDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_ResUp', 'h_reg_'+reg+'_'+limit_varSR+'_ResUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_ResDown', 'h_reg_'+reg+'_'+limit_varSR+'_ResDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_jesUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_jesUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_jesDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_jesDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scaleUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scaleUp',
            #         var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scaleDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scaleDown',
            #         var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_wjetsUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_wjetsUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_wjetsDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_wjetsDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_dyjetsUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_dyjetsUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_dyjetsDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_dyjetsDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_zjetsUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_zjetsUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_zjetsDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_zjetsDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_dibosonUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_dibosonUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_dibosonDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_dibosonDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_singletUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_singletUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_singletDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_singletDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_ttUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_ttUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_ttDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_ttDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_qcdUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_qcdUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_qcdDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_qcdDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_smhlUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_smhUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_smhDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_smhDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_signalUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_signalUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_signalDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_signalDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_miscUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_miscUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_miscDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_mu_scale_miscDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_pdfUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_pdfUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_pdfDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_pdfDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_prefireUp', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_prefireUp', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varSR+'_CMSyear_prefireDown', 'h_reg_'+reg+'_'+limit_varSR+'_CMSyear_prefireDown', var_legendSR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
        if allplots:
            makeplot('reg_'+reg+'_cutFlow', 'h_reg_'+reg+'_cutFlow', 'CutFlow', 0, 8, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nJets', 'h_reg_'+reg+'_nJets', 'nJets', 0., 10., rebin, 1, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_METPhi', 'h_reg_'+reg+'_METPhi', ' p_{T}^{miss} #phi', -3, 3, 1, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_dPhiTrk_pfMET', 'h_reg_'+reg+'_dPhiTrk_pfMET', '#Delta#phi (Trkp_{T}^{miss} - p_{T}^{miss})', -3.2, 3.2, rebin, 1, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_dPhiCalo_pfMET', 'h_reg_'+reg+'_dPhiCalo_pfMET', '#Delta#phi(Calop_{T}^{miss} - pfp_{T}^{miss})', -3.2, 3.2, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1Pt', 'h_reg_'+reg+'_Jet1Pt', 'JET1 p_{T} (GeV)', 30., 800., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_delta_pfCalo', 'h_reg_'+reg+'_delta_pfCalo', 'PFp_{T}^{miss}-Calop_{T}^{miss}/Recoil', 0., 1.5, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1Eta', 'h_reg_'+reg+'_Jet1Eta', 'JET1 #eta', -2.5, 2.5, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1Phi', 'h_reg_'+reg+'_Jet1Phi', 'JET1 #phi', -3.14, 3.14, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1deepCSV', 'h_reg_'+reg+'_Jet1deepCSV', 'JET1 deepCSV', 0, 1.2, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1NHadEF', 'h_reg_'+reg+'_Jet1NHadEF', 'Jet1NHadEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1CHadEF', 'h_reg_'+reg+'_Jet1CHadEF', 'Jet1CHadEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1CEmEF', 'h_reg_'+reg+'_Jet1CEmEF', 'Jet1CEmEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1NEmEF', 'h_reg_'+reg+'_Jet1NEmEF', 'Jet1NEmEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1CMulti', 'h_reg_'+reg+'_Jet1CMulti', 'Jet1CMulti', 0, 50, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1NMultiplicity', 'h_reg_'+reg+'_Jet1NMultiplicity', 'Jet1NMultiplicity', 0, 50, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2Pt', 'h_reg_'+reg+'_Jet2Pt', 'JET2 p_{T} (GeV)', 30., 800., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2Eta', 'h_reg_'+reg+'_Jet2Eta', 'JET2 #eta', -2.5, 2.5, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2Phi', 'h_reg_'+reg+'_Jet2Phi', 'JET2 #phi', -3.14, 3.14, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2deepCSV', 'h_reg_'+reg+'_Jet2deepCSV', 'JET2 deepCSV', 0, 1.2, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2NHadEF', 'h_reg_'+reg+'_Jet2NHadEF', 'Jet2NHadEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2CHadEF', 'h_reg_'+reg+'_Jet2CHadEF', 'Jet2CHadEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2CEmEF', 'h_reg_'+reg+'_Jet2CEmEF', 'Jet2CEmEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2NEmEF', 'h_reg_'+reg+'_Jet2NEmEF', 'Jet2NEmEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2CMulti', 'h_reg_'+reg+'_Jet2CMulti', 'Jet2CMulti', 0, 50, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet2NMultiplicity', 'h_reg_'+reg+'_Jet2NMultiplicity', 'Jet2NMultiplicity', 0, 50, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nBJets', 'h_reg_'+reg+'_nBJets', 'nBJets', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_NEle', 'h_reg_'+reg+'_NEle', 'NEle', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_NMu', 'h_reg_'+reg+'_NMu', 'NMu', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nPho', 'h_reg_'+reg+'_nPho', 'nPho', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_NTau', 'h_reg_'+reg+'_NTau', 'NTau', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_ratioPtJet21', 'h_reg_'+reg+'_ratioPtJet21', 'JET2 p_{T}/JET1 p_{T} ', 0., 1.0, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_dPhiJet12', 'h_reg_'+reg+'_dPhiJet12', 'JET1#phi - JET2#phi', -7.5, 7.5, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_dEtaJet12', 'h_reg_'+reg+'_dEtaJet12', 'JET1#eta - JET2#eta', -7.5, 7.5, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nPV', 'h_reg_'+reg+'_nPV', 'Before PU reweighting', 0., 70., 2, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_PUnPV', 'h_reg_'+reg+'_PUnPV', 'After PU reweighting', 0., 70., 2, 1, 0, reg, varBin=False)
            if ('SR_1b' in reg):
                makeplot('reg_'+reg+'_isjet1EtaMatch', 'h_reg_'+reg+'_isjet1EtaMatch',     'JET1#eta X JET2#eta', 0, 1, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_M_Jet1Jet2', 'h_reg_'+reg+'_M_Jet1Jet2',     'Inv Mass(Jet1, Jet2)', 0, 2000, 5, 1, 0, reg, varBin=False)
            elif ('SR_2b' in reg):
                # makeplot('reg_'+reg+'_MET', 'h_reg_'+reg+'_MET',     # 'p_{T}^{miss} (GeV)', 250, 1000, rebin, isLog, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_isjet2EtaMatch', 'h_reg_'+reg+'_isjet2EtaMatch',     'JET1#eta X JET3#eta', -1, 1, rebin, isLog, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_M_Jet1Jet3', 'h_reg_'+reg+'_M_Jet1Jet3',     'Inv Mass(Jet1, Jet3)', 0, 2000, 10, 1, 0, reg, varBin=False)
                # makeplot('reg_'+reg+'_M_Jet1Jet2', 'h_reg_'+reg+'_M_Jet1Jet2',
                #          'Inv Mass(Jet1, Jet2)', 0, 2000, 5, 1, 0, reg, varBin=False)
            elif ('preselR' in reg):
                makeplot('reg_'+reg+'_isjet1EtaMatch', 'h_reg_'+reg+'_isjet1EtaMatch',     'JET1#eta X JET3#eta', -1, 1,rebin, isLog, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_M_Jet1Jet3', 'h_reg_'+reg+'_M_Jet1Jet3',     'Inv Mass(Jet1, Jet2)', 0, 2000, 5, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_isjet2EtaMatch', 'h_reg_'+reg+'_isjet2EtaMatch',     'JET1#eta X JET3#eta', -1, 1, rebin, isLog, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_M_Jet1Jet3', 'h_reg_'+reg+'_M_Jet1Jet3',     'Inv Mass(Jet1, Jet3)', 0, 2000, 5, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_rJet1PtMET', 'h_reg_'+reg+'_rJet1PtMET', 'Jet1 p_{T}/MET', 0, 20, 5, 1, 0, reg, varBin=False)
    else:
        makeplot('reg_'+reg+'_'+limit_varCR, 'h_reg_'+reg+'_'+limit_varCR, var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
        makeplot('reg_'+reg+'_lep1_pT', 'h_reg_'+reg+'_lep1_pT', 'lepton1 p_{T}', 30, 500, rebin, 1, 0, reg, varBin=False)
        makeplot('reg_'+reg+'_min_dPhi', 'h_reg_'+reg+'_min_dPhi', 'min_dPhi', 0, 3.2, 1, 1, 0, reg, varBin=False)
        # if ('_2b' in reg or '_3j' in reg):
            # makeplot('reg_'+reg+'_Recoil', 'h_reg_'+reg+'_Recoil', 'Recoil (GeV)', 250, 1000, rebin, isLog, 0, reg, varBin=False)
        # makeplot('reg_'+reg+'_prod_cat', 'h_reg_'+reg+'_prod_cat', 'Jet Flavour', 0, 3, 1, 1, 0, reg, varBin=False)
        if systHistoFiles:
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_eff_bUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_eff_bUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_eff_bDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_eff_bDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_fake_bUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_fake_bUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_fake_bDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_fake_bDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_EWKUp', 'h_reg_'+reg+'_'+limit_varCR+'_EWKUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_EWKDown', 'h_reg_'+reg+'_'+limit_varCR+'_EWKDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_TopUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_TopUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_TopDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_TopDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_trig_metUp', 'h_reg_'+reg + '_'+limit_varCR+'_CMSyear_trig_metUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_trig_metDown', 'h_reg_'+reg + '_'+limit_varCR+'_CMSyear_trig_metDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_PUUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_PUUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_PUDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_PUDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varCR+'_JECUp', 'h_reg_'+reg+'_'+limit_varCR+'_JECUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varCR+'_JECDown', 'h_reg_'+reg+'_'+limit_varCR+'_JECDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECAbsoluteUp', 'h_reg_'+reg + '_'+limit_varCR+'_JECAbsoluteUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECAbsolute_yearUp', 'h_reg_'+reg + '_'+limit_varCR+'_JECAbsolute_yearUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECBBEC1Up', 'h_reg_'+reg + '_'+limit_varCR+'_JECBBEC1Up', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECBBEC1_yearUp', 'h_reg_'+reg + '_'+limit_varCR+'_JECBBEC1_yearUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECEC2Up', 'h_reg_'+reg + '_'+limit_varCR+'_JECEC2Up', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECEC2_yearUp', 'h_reg_'+reg + '_'+limit_varCR+'_JECEC2_yearUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECFlavorQCDUp', 'h_reg_'+reg + '_'+limit_varCR+'_JECFlavorQCDUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECHFUp', 'h_reg_'+reg+'_'+limit_varCR+'_JECHFUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECHF_yearUp', 'h_reg_'+reg + '_'+limit_varCR+'_JECHF_yearUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECRelativeBalUp', 'h_reg_'+reg + '_'+limit_varCR+'_JECRelativeBalUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECRelativeSample_yearUp', 'h_reg_'+reg + '_'+limit_varCR+'_JECRelativeSample_yearUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECAbsoluteDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECAbsoluteDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECAbsolute_yearDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECAbsolute_yearDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECBBEC1Down', 'h_reg_'+reg + '_'+limit_varCR+'_JECBBEC1Down', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECBBEC1_yearDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECBBEC1_yearDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECEC2Down', 'h_reg_'+reg + '_'+limit_varCR+'_JECEC2Down', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECEC2_yearDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECEC2_yearDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECFlavorQCDDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECFlavorQCDDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECHFDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECHFDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECHF_yearDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECHF_yearDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECRelativeBalDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECRelativeBalDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_JECRelativeSample_yearDown', 'h_reg_'+reg + '_'+limit_varCR+'_JECRelativeSample_yearDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varCR+'_ResUp', 'h_reg_'+reg+'_'+limit_varCR+'_ResUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varCR+'_ResDown', 'h_reg_'+reg+'_'+limit_varCR+'_ResDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varCR+'_EnUp', 'h_reg_'+reg+'_'+limit_varCR+'_EnUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varCR+'_EnDown', 'h_reg_'+reg+'_'+limit_varCR+'_EnDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_jesUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_jesUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_jesDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_jesDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scaleUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scaleUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scaleDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scaleDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_wjetsUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_wjetsUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_wjetsDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_wjetsDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_dyjetsUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_dyjetsUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_dyjetsDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_dyjetsDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_zjetsUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_zjetsUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_zjetsDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_zjetsDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_dibosonUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_dibosonUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_dibosonDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_dibosonDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_singletUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_singletUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_singletDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_singletDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_ttUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_ttUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_ttDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_ttDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_qcdUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_qcdUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_qcdDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_qcdDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_smhUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_smhUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_smhDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_smhDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_signalUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_signalUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_signalDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_signalDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_miscUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_miscUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_miscDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_mu_scale_miscDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_pdfUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_pdfUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_pdfDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_pdfDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_prefireUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_prefireUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_prefireDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_prefireDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_trig_eleUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_trig_eleUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_trig_eleDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_trig_eleDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_EleIDUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_EleIDUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_EleIDDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_EleIDDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_EleRECOUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_EleRECOUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_EleRECODown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_EleRECODown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_MuIDUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_MuIDUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_MuIDDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_MuIDDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_MuISOUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_MuISOUp',     var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_MuISODown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_MuISODown',     var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_MuTRKUp', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_MuTRKUp', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_'+limit_varCR+'_CMSyear_MuTRKDown', 'h_reg_'+reg+'_'+limit_varCR+'_CMSyear_MuTRKDown', var_legendCR, minBin, maxBin, rebin, isLog, 0, reg, varBin=False)
        if allplots:
            makeplot('reg_'+reg+'_cutFlow', 'h_reg_'+reg+'_cutFlow', 'CutFlow', 0, 12, rebin, isLog, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_MET', 'h_reg_'+reg+'_MET', 'Real p_{T}^{miss} (GeV)', 0., 700., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_METPhi', 'h_reg_'+reg+'_METPhi', ' p_{T}^{miss} #phi', -3, 3, rebin, isLog, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_dPhiTrk_pfMET', 'h_reg_'+reg+'_dPhiTrk_pfMET', '#Delta#phi (Trkp_{T}^{miss} - p_{T}^{miss})', -3.2, 3.2, rebin, 1, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_dPhiCalo_pfMET', 'h_reg_'+reg+'_dPhiCalo_pfMET', '#Delta#phi(Calop_{T}^{miss} - pfp_{T}^{miss})', -3.2, 3.2, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1Pt', 'h_reg_'+reg+'_Jet1Pt', 'JET1 p_{T} (GeV)', 30., 800., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_delta_pfCalo', 'h_reg_'+reg+'_delta_pfCalo', 'PFp_{T}^{miss}-Calop_{T}^{miss}/Recoil', 0., 1.5, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1Eta', 'h_reg_'+reg+'_Jet1Eta', 'JET1 #eta', -2.5, 2.5, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_dPhi_lep1_MET', 'h_reg_'+reg+'_dPhi_lep1_MET', '#Delta(lepton1,p_{T}^{Miss})', 0, 5, rebin, 1, 0, reg, varBin=False)
            if ('2b' in reg or '3j' in reg) and 'Z' in reg:
                makeplot('reg_'+reg+'_dPhi_lep2_MET', 'h_reg_'+reg+'_dPhi_lep2_MET',     '#Delta(lepton2,p_{T}^{Miss})', 0, 5, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1Phi', 'h_reg_'+reg+'_Jet1Phi', 'JET1 #phi', -3.14, 3.14, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1deepCSV', 'h_reg_'+reg+'_Jet1deepCSV', 'JET1 deepCSV', 0, 1.2, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1NHadEF', 'h_reg_'+reg+'_Jet1NHadEF', 'Jet1NHadEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1CHadEF', 'h_reg_'+reg+'_Jet1CHadEF', 'Jet1CHadEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1CEmEF', 'h_reg_'+reg+'_Jet1CEmEF', 'Jet1CEmEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1NEmEF', 'h_reg_'+reg+'_Jet1NEmEF', 'Jet1NEmEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1CMulti', 'h_reg_'+reg+'_Jet1CMulti', 'Jet1CMulti', 0, 50, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_Jet1NMultiplicity', 'h_reg_'+reg+'_Jet1NMultiplicity', 'Jet1NMultiplicity', 0, 50, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_lep1_pT', 'h_reg_'+reg+'_lep1_pT', 'lepton1 p_{T}', 30, 500, rebin, 1, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_lep1_pT_CMSyear_EleIDUp', 'h_reg_'+reg+'_lep1_pT_CMSyear_EleIDUp', 'lepton1 p_{T}', 0, 500, rebin, 1, 0, reg, varBin=False)
            # makeplot('reg_'+reg+'_lep1_pT_CMSyear_EleIDDown', 'h_reg_'+reg+'_lep1_pT_CMSyear_EleIDDown', 'lepton1 p_{T}', 0, 500, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_lep1_Phi', 'h_reg_'+reg+'_lep1_Phi', 'lepton1 #phi', -3.14, 3.14, rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nJets', 'h_reg_'+reg+'_nJets', 'nJets', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nBJets', 'h_reg_'+reg+'_nBJets', 'nBJets', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_NEle', 'h_reg_'+reg+'_NEle', 'NEle', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_NMu', 'h_reg_'+reg+'_NMu', 'NMu', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nPho', 'h_reg_'+reg+'_nPho', 'nPho', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_NTau', 'h_reg_'+reg+'_NTau', 'NTau', 0., 10., rebin, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_nPV', 'h_reg_'+reg+'_nPV', 'Before PU reweighting', 0., 70., 2, 0, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_PUnPV', 'h_reg_'+reg+'_PUnPV', 'After PU reweighting', 0., 70., 2, 0, 0, reg, varBin=False)
            if ('W' in reg) or ('Top' in reg):
                makeplot('reg_'+reg+'_Wmass', 'h_reg_'+reg+'_Wmass',     'W candidate mass (GeV)', 0., 165., rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_WpT', 'h_reg_'+reg+'_WpT',     'W candidate p_{T} (GeV)', 0., 700., rebin, 1, 0, reg, varBin=False)
            if ('Z' in reg):
                makeplot('reg_'+reg+'_Zmass', 'h_reg_'+reg+'_Zmass',     'Z candidate mass (GeV)', 70., 110., 1, 0, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_ZpT', 'h_reg_'+reg+'_ZpT',     'Z candidate p_{T} (GeV)', 0., 700., rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_lep2_pT', 'h_reg_'+reg+'_lep2_pT',     'lepton2 p_{T}', 30, 500, 1, 1, 0, reg, varBin=False)
            if ('WmunuCR_1b' not in reg) and ('WenuCR_1b' not in reg):
                makeplot('reg_'+reg+'_Jet2Pt', 'h_reg_'+reg+'_Jet2Pt',     'JET2 p_{T} (GeV)', 30., 800., rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2Eta', 'h_reg_'+reg+'_Jet2Eta',     'JET2 #eta', -2.5, 2.5, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2Phi', 'h_reg_'+reg+'_Jet2Phi',     'JET2 #phi', -3.14, 3.14, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2deepCSV', 'h_reg_'+reg+'_Jet2deepCSV',     'JET2 deepCSV', 0, 1.2, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2NHadEF', 'h_reg_'+reg+'_Jet2NHadEF',     'Jet2NHadEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2CHadEF', 'h_reg_'+reg+'_Jet2CHadEF',     'Jet2CHadEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2CEmEF', 'h_reg_'+reg+'_Jet2CEmEF',     'Jet2CEmEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2NEmEF', 'h_reg_'+reg+'_Jet2NEmEF',     'Jet2NEmEF', 0, 1.1, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2CMulti', 'h_reg_'+reg+'_Jet2CMulti',     'Jet2CMulti', 0, 50, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_Jet2NMultiplicity', 'h_reg_'+reg+'_Jet2NMultiplicity',     'Jet2NMultiplicity', 0, 50, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_ratioPtJet21', 'h_reg_'+reg+'_ratioPtJet21',     'JET2 p_{T}/JET1 p_{T} ', 0., 1.0, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_dPhiJet12', 'h_reg_'+reg+'_dPhiJet12',     'JET1#phi - JET2#phi', -7.5, 7.5, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_dEtaJet12', 'h_reg_'+reg+'_dEtaJet12',     'JET1#eta - JET2#eta', -7.5, 7.5, rebin, 1, 0, reg, varBin=False)
            if ('1b' in reg or '2j' in reg) and ('WmunuCR_1b' not in reg) and ('WenuCR_1b' not in reg):
                makeplot('reg_'+reg+'_isjet1EtaMatch', 'h_reg_'+reg+'_isjet1EtaMatch',     'JET1#eta X JET2#eta', 0, 1, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_M_Jet1Jet2', 'h_reg_'+reg+'_M_Jet1Jet2',     'Inv Mass(Jet1, Jet2)', 0, 2000, rebin, 1, 0, reg, varBin=False)
            elif ('WmunuCR_2b' in reg or 'WenuCR_2b' in reg):
                makeplot('reg_'+reg+'_isjet1EtaMatch', 'h_reg_'+reg+'_isjet1EtaMatch',     'JET1#eta X JET2#eta', 0, 1, rebin, 1, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_M_Jet1Jet2', 'h_reg_'+reg+'_M_Jet1Jet2',     'Inv Mass(Jet1, Jet2)', 0, 2000, 5, 1, 0, reg, varBin=False)
            elif ('2b' in reg or '3j' in reg):
                makeplot('reg_'+reg+'_Recoil', 'h_reg_'+reg+'_Recoil',     'Recoil (GeV)', 250, 1000, rebin, isLog, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_isjet2EtaMatch', 'h_reg_'+reg+'_isjet2EtaMatch',     'JET1#eta X JET3#eta', -1, 1, rebin, isLog, 0, reg, varBin=False)
                makeplot('reg_'+reg+'_M_Jet1Jet3', 'h_reg_'+reg+'_M_Jet1Jet3',     'Inv Mass(Jet1, Jet3)', 0, 2000, 5, 1, 0, reg, varBin=False)
            makeplot('reg_'+reg+'_rJet1PtMET', 'h_reg_'+reg+'_rJet1PtMET', 'Jet1 p_{T}/MET', 0, 10, rebin, isLog, 0, reg, varBin=False)
    # except:
    #     pass
yield_outfile.close()
yield_outfile_binwise.close()
