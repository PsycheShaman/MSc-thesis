//
// Configuration for PWG3 barrel open charm and beauty PbPb 2010
//
// 1 HIJING event 2.76 TeV with b<30fm and central dNch/deta=2000
// +
// N (20) PYTHIA pp 2.76 TeV Perugia0
// 20% ccbar pair per event
//     at least one in |y|<1.5
//     D mesons decay hadronically
// 20% bbbar pair per event
//     at least one in |y|<1.5
//     D mesons decay hadronically
// 20% ccbar pair per event
//     decays not forced
//     at least one electron from charm in |y|<1.2
// 20% bbbar pair per event
//     decays not forced
//     at least one electron from charm or beauty in |y|<1.2
//  10% J/psi(|y|<1.0)->e+e-
//  10% B(|y|<2.0)->J/psi(|y|<2.0)->e+e-
//
// One can use the configuration macro in compiled mode by
// root [0] gSystem->Load("libgeant321");
// root [0] gSystem->SetIncludePath("-I$ROOTSYS/include -I$ALICE_ROOT/include\
//                   -I$ALICE_ROOT -I$ALICE/geant3/TGeant3");
// root [0] .x grun.C(1,"Config.C++")
//
//
#if !defined(__CINT__) || defined(__MAKECINT__)
#include <Riostream.h>
#include <TRandom.h>
#include <TDatime.h>
#include <TSystem.h>
#include <TVirtualMC.h>
#include <TGeant3TGeo.h>
#include "STEER/AliRunLoader.h"
#include "STEER/AliRun.h"
#include "STEER/AliConfig.h"
#include "PYTHIA6/AliDecayerPythia.h"
#include "PYTHIA6/AliGenPythia.h"
#include "TDPMjet/AliGenDPMjet.h"
#include "STEER/AliMagFCheb.h"
#include "STRUCT/AliBODY.h"
#include "STRUCT/AliMAG.h"
#include "STRUCT/AliABSOv3.h"
#include "STRUCT/AliDIPOv3.h"
#include "STRUCT/AliHALLv3.h"
#include "STRUCT/AliFRAMEv2.h"
#include "STRUCT/AliSHILv3.h"
#include "STRUCT/AliPIPEv3.h"
#include "ITS/AliITSv11.h"
#include "TPC/AliTPCv2.h"
#include "TOF/AliTOFv6T0.h"
#include "HMPID/AliHMPIDv3.h"
#include "ZDC/AliZDCv3.h"
#include "TRD/AliTRDv1.h"
#include "TRD/AliTRDgeometry.h"
#include "FMD/AliFMDv1.h"
#include "MUON/AliMUONv1.h"
#include "PHOS/AliPHOSv1.h"
#include "PHOS/AliPHOSSimParam.h"
#include "PMD/AliPMDv1.h"
#include "T0/AliT0v1.h"
#include "EMCAL/AliEMCALv2.h"
#include "ACORDE/AliACORDEv1.h"
#include "VZERO/AliVZEROv7.h"
#endif


enum PDC06Proc_t
{
  kPythia6, kPythia6D6T, kPythia6ATLAS, kPythia6ATLAS_Flat, kPythiaPerugia0, kPhojet, kHijing, kHijing2000, kHijing2000HF, kHijingNuclei, kHydjet, kDpmjet, kRunMax
};

const char * pprRunName[] = {
  "kPythia6", "kPythia6D6T", "kPythia6ATLAS", "kPythia6ATLAS_Flat", "kPythiaPerugia0", "kPhojet", "kHijing", "kHijing2000", "kHijing2000HF", "kHijingNuclei", "kHydjet", "kDpmjet"
};

enum Mag_t
{
  kNoField, k5kG, kFieldMax
};

const char * pprField[] = {
  "kNoField", "k5kG"
};

enum PprTrigConf_t
{
    kDefaultPPTrig, kDefaultPbPbTrig
};

const char * pprTrigConfName[] = {
    "p-p","Pb-Pb"
};

//--- Functions ---
class AliGenPythia;
AliGenerator *MbPythia();
AliGenerator *MbPythiaTuneD6T();
AliGenerator *MbPhojet();
AliGenerator *Hijing();
AliGenerator *Hijing2000();
AliGenerator *HijingNuclei();
AliGenerator *Hijing2000HF(Int_t typeHF, Float_t ptHmin, Float_t ptHmax, Float_t ptHmin, Float_t ptHmax);
AliGenerator *Hydjet();
AliGenerator *Dpmjet();
AliGenerator *Ampt();
AliGenerator* HijingPlusJets();

void ProcessEnvironmentVars();

// Geterator, field, beam energy
static PDC06Proc_t   proc     = kHijingNuclei;
static Int_t         hproc    = -1;
static Mag_t         mag      = k5kG;
static Float_t       energy   = 2760.; // energy in CMS
static Float_t       bMin     = 29.;
static Float_t       bMax =   = 30.;
static PprTrigConf_t strig = kDefaultPbPbTrig; // default pp trigger configuration
static Double_t      JpsiPol  = 0; // Jpsi polarisation
static Bool_t        JpsiHarderPt = kFALSE; // Jpsi harder pt spectrum (8.8 TeV)
//========================//
// Set Random Number seed //
//========================//
TDatime dt;
static UInt_t seed    = dt.Get();

// Comment line
static TString comment;

void Config()
{
  // Get settings from environment variables
  ProcessEnvironmentVars();

  gRandom->SetSeed(seed);
  cerr<<"Seed for random number generation= "<<seed<<endl;

  // Libraries required by geant321
#if defined(__CINT__)
  gSystem->AddIncludePath("-I$ALICE_ROOT/HIJING -I$ALICE_ROOT/THijing -I$ALICE_ROOT/PYTHIA6 -I$ALICE_ROOT/PYTHIA8");
  gSystem->Load("liblhapdf");      // Parton density functions
  gSystem->Load("libEG");
  gSystem->Load("libEGPythia6");   // TGenerator interface
  if (proc == kPythia6 || proc == kPhojet || proc == kDpmjet) {
    gSystem->Load("libpythia6");        // Pythia 6.2
    gSystem->Load("libAliPythia6");     // ALICE specific implementations
  } else if (proc != kHydjet) {
    //    gSystem->Load("libpythia6.4.21");   // Pythia 6.4
    gSystem->Load("libpythia6");
    gSystem->Load("libAliPythia6");     // ALICE specific implementations
  }

  if (proc == kHijing || proc == kHijing2000 || proc == kHijingNuclei || proc == kHijing2000HF ||  proc == kHijingPlusJets1 || proc == kHijingPlusJets2 || proc == kHijingPlusJets3 ||  proc == kHijingPlusJetsPlusBox ) {
	  gSystem->Load("libHIJING");
  	  gSystem->Load("libTHijing");
  } else if (proc == kHydjet)  {
	  gSystem->Load("libTUHKMgen");
  } else if (proc == kDpmjet) {
	  gSystem->Load("libdpmjet");
          gSystem->Load("libTDPMjet");
  } else if (proc == kAmptHF || proc == kAmpt) {
	  gSystem->Load("libampt");
       	  gSystem->Load("libTAmpt");
  }

  gSystem->Load("libgeant321");

#endif

  new TGeant3TGeo("C++ Interface to Geant3");

  //=======================================================================
  //  Create the output file


  AliRunLoader* rl=0x0;

  cout<<"Config.C: Creating Run Loader ..."<<endl;
  rl = AliRunLoader::Open("galice.root",
			  AliConfig::GetDefaultEventFolderName(),
			  "recreate");
  if (rl == 0x0)
    {
      gAlice->Fatal("Config.C","Can not instatiate the Run Loader");
      return;
    }
  rl->SetCompressionLevel(2);
  rl->SetNumberOfEventsPerFile(1000);
  gAlice->SetRunLoader(rl);

    // Set the trigger configuration
    AliSimulation::Instance()->SetTriggerConfig(pprTrigConfName[strig]);
    cout<<"Trigger configuration is set to  "<<pprTrigConfName[strig]<<endl;

  //
  //=======================================================================
  // ************* STEERING parameters FOR ALICE SIMULATION **************
  // --- Specify event type to be tracked through the ALICE setup
  // --- All positions are in cm, angles in degrees, and P and E in GeV


    gMC->SetProcess("DCAY",1);
    gMC->SetProcess("PAIR",1);
    gMC->SetProcess("COMP",1);
    gMC->SetProcess("PHOT",1);
    gMC->SetProcess("PFIS",0);
    gMC->SetProcess("DRAY",0);
    gMC->SetProcess("ANNI",1);
    gMC->SetProcess("BREM",1);
    gMC->SetProcess("MUNU",1);
    gMC->SetProcess("CKOV",1);
    gMC->SetProcess("HADR",1);
    gMC->SetProcess("LOSS",2);
    gMC->SetProcess("MULS",1);
    gMC->SetProcess("RAYL",1);

    Float_t cut = 1.e-3;        // 1MeV cut by default
    Float_t tofmax = 1.e10;

    gMC->SetCut("CUTGAM", cut);
    gMC->SetCut("CUTELE", cut);
    gMC->SetCut("CUTNEU", cut);
    gMC->SetCut("CUTHAD", cut);
    gMC->SetCut("CUTMUO", cut);
    gMC->SetCut("BCUTE",  cut);
    gMC->SetCut("BCUTM",  cut);
    gMC->SetCut("DCUTE",  cut);
    gMC->SetCut("DCUTM",  cut);
    gMC->SetCut("PPCUTM", cut);
    gMC->SetCut("TOFMAX", tofmax);



    Float_t randHF = gRandom->Rndm();
    Int_t typeHF = -1;
    if (hproc == -1) {
      // RANDOM SELECTION OF ONE OF THE SEVEN GENERATION TYPES
      //
       if(randHF < 0.16) {
	typeHF=0;
      } else if (randHF >= 0.16 && randHF < 0.32) {
	typeHF=1;
      } else if (randHF >= 0.32 && randHF < 0.48) {
	typeHF=2;
      } else if (randHF >= 0.48 && randHF < 0.64) {
	typeHF=3;
      } else if (randHF >= 0.64 && randHF < 0.72) {
	typeHF=4;
      } else if (randHF >= 0.72 && randHF < 0.80) {
	typeHF=5;
      } else {
	typeHF=6;
      }
    } else {
       if (hproc == 1) typeHF = (randHF < 0.5)? 0:1;
       if (hproc == 2) typeHF = (randHF < 0.5)? 2:3;
       if (hproc == 3) typeHF = (randHF < 0.5)? 4:5;
       if (hproc == 4) typeHF = 6;
    }

    printf("subtype typeHF %5d \n", typeHF);

    TString sproc("typeHF_");
    sproc += Form("%d.proc", typeHF);
    TFile* file = new TFile(sproc, "recreate");
    file->Close();
    //
    // pt Hard bin for HF production
    Float_t ptHmin[3]  =  {2.76,  20.,   50.};
    Float_t ptHmax[3]  =  {20.,   50., 1000.};

    Float_t ranPTH = gRandom->Rndm();
    Int_t ptHbin =  -1;
    if (ranPTH < 0.7) {
      ptHbin = 0;
    } else if (ranPTH < 0.9) {
      ptHbin = 1;
    } else {
      ptHbin = 2;
    }

    //
    // pt Hard bin for Jet production
    Float_t ptHJmin[3]  =  {15.,  50.,   80.};
    Float_t ptHJmax[3]  =  {50.,  80., 1000.};

    Int_t ptHJbin = 3. * gRandom->Rndm();

/*
    //======================//
    // Set External decayer //
    //======================//
    if (proc != kHydjet) {
      TVirtualMCDecayer* decayer = new AliDecayerPythia();
      if(proc == kHijing2000HF && (typeHF==0 || typeHF==1)) {
	decayer->SetForceDecay(kHadronicDWithout4Bodies);
      } else {
	decayer->SetForceDecay(kAll);
      }
      decayer->Init();
      gMC->SetExternalDecayer(decayer);
    }
*/
  //=========================//
  // Generator Configuration //
  //=========================//
  AliGenerator* gener = 0x0;

  if (proc == kPythia6) {
      gener = MbPythia();
  } else if (proc == kPythia6D6T) {
      gener = MbPythiaTuneD6T();
  } else if (proc == kPythia6ATLAS) {
      gener = MbPythiaTuneATLAS();
  } else if (proc == kPythiaPerugia0) {
      gener = MbPythiaTunePerugia0();
  } else if (proc == kPythia6ATLAS_Flat) {
      gener = MbPythiaTuneATLAS_Flat();
  } else if (proc == kPhojet) {
      gener = MbPhojet();
  } else if (proc == kHijing) {
      gener = Hijing();
  } else if (proc == kHijing2000) {
      gener = Hijing2000();
  } else if (proc == kHijingNuclei) {
      gener = HijingNuclei();
  } else if (proc == kHijing2000HF || proc == kAmptHF) {
    gener = Hijing2000HF(typeHF, ptHmin[ptHbin], ptHmax[ptHbin], ptHJmin[ptHJbin], ptHJmax[ptHJbin]);
  } else if (proc == kHydjet) {
      gener = Hydjet();
  } else if (proc == kDpmjet) {
      gener = Dpmjet();
  } else if (proc == kAmpt) {
      gener = Ampt();
  }  else if (proc == kHijingPlusJets1 || proc == kHijingPlusJets2 || proc == kHijingPlusJets3) {
      gener =HijingPlusJets();
  }
  else if (proc == kHijingPlusJetsPlusBox) {
      gener = HijingPlusJetsPlusBox();
  }
  else if (proc == kPythiaJets) {
     gener = PythiaJets();
  }
  else if (proc == kPythiaJetsPlusBox) {
     gener = PythiaJetsPlusBox();
  }



  //
  //
  // Size of the interaction diamond
  // Longitudinal
  Float_t sigmaz  = 5.4 / TMath::Sqrt(2.); // [cm]

  //
  // Transverse
  Float_t betast  = 3.5;                      // beta* [m]
  Float_t eps     = 3.75e-6;                   // emittance [m]
  Float_t gamma   = energy / 2.0 / 0.938272;  // relativistic gamma [1]
  Float_t sigmaxy = TMath::Sqrt(eps * betast / gamma) / TMath::Sqrt(2.) * 100.;  // [cm]

  printf("\n \n Diamond size x-y: %10.3e z: %10.3e\n \n", sigmaxy, sigmaz);

  gener->SetSigma(sigmaxy, sigmaxy, sigmaz);      // Sigma in (X,Y,Z) (cm) on IP position
  gener->SetVertexSmear(kPerEvent);
  gener->Init();

  printf("\n \n Comment: %s \n \n", comment.Data());

   //
   // FIELD
   //

  TGeoGlobalMagField::Instance()->SetField(new AliMagF("Maps","Maps", -1., -1., AliMagF::k5kG,
     	   	AliMagF::kBeamTypeAA, 1380.));


  rl->CdGAFile();

  Int_t iABSO  = 1;
  Int_t iACORDE= 0;
  Int_t iDIPO  = 1;
  Int_t iEMCAL = 1;
  Int_t iFMD   = 1;
  Int_t iFRAME = 1;
  Int_t iHALL  = 1;
  Int_t iITS   = 1;
  Int_t iMAG   = 1;
  Int_t iMUON  = 1;
  Int_t iPHOS  = 1;
  Int_t iPIPE  = 1;
  Int_t iPMD   = 1;
  Int_t iHMPID = 1;
  Int_t iSHIL  = 1;
  Int_t iT0    = 1;
  Int_t iTOF   = 1;
  Int_t iTPC   = 1;
  Int_t iTRD   = 1;
  Int_t iVZERO = 1;
  Int_t iZDC   = 1;


    //=================== Alice BODY parameters =============================
    AliBODY *BODY = new AliBODY("BODY", "Alice envelop");


    if (iMAG)
    {
        //=================== MAG parameters ============================
        // --- Start with Magnet since detector layouts may be depending ---
        // --- on the selected Magnet dimensions ---
        AliMAG *MAG = new AliMAG("MAG", "Magnet");
    }


    if (iABSO)
    {
        //=================== ABSO parameters ============================
        AliABSO *ABSO = new AliABSOv3("ABSO", "Muon Absorber");
    }

    if (iDIPO)
    {
        //=================== DIPO parameters ============================

        AliDIPO *DIPO = new AliDIPOv3("DIPO", "Dipole version 3");
    }

    if (iHALL)
    {
        //=================== HALL parameters ============================

        AliHALL *HALL = new AliHALLv3("HALL", "Alice Hall");
    }


    if (iFRAME)
    {
        //=================== FRAME parameters ============================

        AliFRAMEv2 *FRAME = new AliFRAMEv2("FRAME", "Space Frame");
	FRAME->SetHoles(1);
    }

    if (iSHIL)
    {
        //=================== SHIL parameters ============================

        AliSHIL *SHIL = new AliSHILv3("SHIL", "Shielding Version 3");
    }


    if (iPIPE)
    {
        //=================== PIPE parameters ============================

        AliPIPE *PIPE = new AliPIPEv3("PIPE", "Beam Pipe");
    }

    if (iITS)
    {
        //=================== ITS parameters ============================

	AliITS *ITS  = new AliITSv11("ITS","ITS v11");
    }

    if (iTPC)
    {
      //============================ TPC parameters =====================

        AliTPC *TPC = new AliTPCv2("TPC", "Default");
    }


    if (iTOF) {
        //=================== TOF parameters ============================

	AliTOF *TOF = new AliTOFv6T0("TOF", "normal TOF");
    }


    if (iHMPID)
    {
        //=================== HMPID parameters ===========================

        AliHMPID *HMPID = new AliHMPIDv3("HMPID", "normal HMPID");

    }


    if (iZDC)
    {
        //=================== ZDC parameters ============================

        AliZDC *ZDC = new AliZDCv3("ZDC", "normal ZDC");
	ZDC->SetSpectatorsTrack();
        ZDC->SetLumiLength(0.);
    }

    if (iTRD)
    {
        //=================== TRD parameters ============================

        AliTRD *TRD = new AliTRDv1("TRD", "TRD slow simulator");
        AliTRDgeometry *geoTRD = TRD->GetGeometry();
	// Partial geometry: modules at 0,1,7,8,9,10,11,15,16,17
	// starting at 3h in positive direction
	geoTRD->SetSMstatus(2,0);
	geoTRD->SetSMstatus(3,0);
	geoTRD->SetSMstatus(4,0);
        geoTRD->SetSMstatus(5,0);
	geoTRD->SetSMstatus(6,0);
        geoTRD->SetSMstatus(12,0);
        geoTRD->SetSMstatus(13,0);
        geoTRD->SetSMstatus(14,0);
    }

    if (iFMD)
    {
        //=================== FMD parameters ============================

	AliFMD *FMD = new AliFMDv1("FMD", "normal FMD");
   }

    if (iMUON)
    {
        //=================== MUON parameters ===========================
        // New MUONv1 version (geometry defined via builders)

        AliMUON *MUON = new AliMUONv1("MUON", "default");
	// activate trigger efficiency by cells
	MUON->SetTriggerEffCells(1); // not needed if raw masks
    }

    if (iPHOS)
    {
        //=================== PHOS parameters ===========================

     AliPHOS *PHOS = new AliPHOSv1("PHOS", "noCPV_Modules123");

    }


    if (iPMD)
    {
        //=================== PMD parameters ============================

        AliPMD *PMD = new AliPMDv1("PMD", "normal PMD");
    }

    if (iT0)
    {
        //=================== T0 parameters ============================
        AliT0 *T0 = new AliT0v1("T0", "T0 Detector");
    }

    if (iEMCAL)
    {
        //=================== EMCAL parameters ============================

        AliEMCAL *EMCAL = new AliEMCALv2("EMCAL", "EMCAL_COMPLETEV1");
    }

     if (iACORDE)
    {
        //=================== ACORDE parameters ============================

        AliACORDE *ACORDE = new AliACORDEv1("ACORDE", "normal ACORDE");
    }

     if (iVZERO)
    {
        //=================== ACORDE parameters ============================

        AliVZERO *VZERO = new AliVZEROv7("VZERO", "normal VZERO");
    }
}
//
//           PYTHIA
//

AliGenerator* MbPythia()
{
      comment = comment.Append(" pp: Pythia low-pt");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-12.,12.);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyMb);
      pythia->SetEnergyCMS(energy);

      return pythia;
}

AliGenerator* MbPythiaTuneD6T()
{
      comment = comment.Append(" pp: Pythia low-pt");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-12.,12.);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyMb);
      pythia->SetEnergyCMS(energy);
//    Tune
//    109     D6T : Rick Field's CDF Tune D6T (NB: needs CTEQ6L pdfs externally)
      pythia->SetTune(109); // F I X
      pythia->SetStrucFunc(kCTEQ6l);
//
      return pythia;
}

AliGenerator* MbPythiaTunePerugia0()
{
      comment = comment.Append(" pp: Pythia low-pt (Perugia0)");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-12.,12.);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyMb);
      pythia->SetEnergyCMS(energy);
//    Tune
//    320     Perugia 0
      pythia->SetTune(320);
      pythia->UseNewMultipleInteractionsScenario();
//
      return pythia;
}


AliGenerator* MbPythiaTuneATLAS()
{
      comment = comment.Append(" pp: Pythia low-pt");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-12.,12.);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyMb);
      pythia->SetEnergyCMS(energy);
//    Tune
//    C   306 ATLAS-CSC: Arthur Moraes' (new) ATLAS tune (needs CTEQ6L externally)
      pythia->SetTune(306);
      pythia->SetStrucFunc(kCTEQ6l);
//
      return pythia;
}

AliGenerator* PythiaJets()
{
      comment = comment.Append(" pp: Pythia low-pt");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-12., 12.);
      pythia->SetPtRange(0, 1000.);
      pythia->SetProcess(kPyJets);
      pythia->SetEnergyCMS(energy);
      pythia->SetStrucFunc(kCTEQ6l);
      pythia->SetJetEtaRange(-1.5, 1.5);
      pythia->SetJetEtRange(500., 1000.); // JMT
      pythia->SetPtHard(450., 1000.); // JMT
      // JMT pythia->SetJetEtRange(10., 800.);
      pythia->SetPycellParameters(2.2, 300, 432, 0., 4., 5., 0.7);
//
      pythia->SetForceDecay(kAll); // JMT

      return pythia;
}

AliGenerator* PythiaJetsPlusBox()
{
      comment = comment.Append(" pp: Pythia Jets + Box pt>50 GeV");

      AliGenCocktail *gener  = new AliGenCocktail();

//
//    Pythia
//
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-12., 12.);
      pythia->SetPtRange(0, 1000.);
      pythia->SetProcess(kPyJets);
      pythia->SetEnergyCMS(energy);
      pythia->SetStrucFunc(kCTEQ6l);
      pythia->SetJetEtaRange(-1.5, 1.5);
      pythia->SetJetEtRange(500., 1000.);
      pythia->SetPtHard(450., 1000.);
      pythia->SetPycellParameters(2.2, 300, 432, 0., 4., 5., 0.7);
//
      pythia->SetForceDecay(kAll);

     //
     // High pt charged particles
     // add 4 pions
     // add 2 protons
     // add 2 kaons
     //
     const  Double_t minPt = 50.;
     const  Double_t maxPt = 100.;

     Float_t thmin          = (180./TMath::Pi())*2.*atan(exp(-1.0));
     Float_t thmax          = (180./TMath::Pi())*2.*atan(exp( 1.0));

     AliGenBox* pionPlus = new AliGenBox(2);     // pions+
     pionPlus->SetPart(211);
     pionPlus->SetPtRange(minPt,maxPt);
     pionPlus->SetThetaRange(thmin, thmax);

     AliGenBox* pionMinus = new AliGenBox(2);     // pions-
     pionMinus->SetPart(-211);
     pionMinus->SetPtRange(minPt,maxPt);
     pionMinus->SetThetaRange(thmin, thmax);

     AliGenBox* kaonPlus = new AliGenBox(1);     // kaons+
     kaonPlus->SetPart(321);
     kaonPlus->SetPtRange(minPt,maxPt);
     kaonPlus->SetThetaRange(thmin, thmax);

     AliGenBox* kaonMinus = new AliGenBox(1);     // kaons-
     kaonMinus->SetPart(-321);
     kaonMinus->SetPtRange(minPt,maxPt);
     kaonMinus->SetThetaRange(thmin, thmax);

     AliGenBox* protonPlus = new AliGenBox(1);     // protons+
     protonPlus->SetPart(2212);
     protonPlus->SetPtRange(minPt,maxPt);
     protonPlus->SetThetaRange(thmin, thmax);

     AliGenBox* protonMinus = new AliGenBox(1);     // protons-
     protonMinus->SetPart(-2212);
     protonMinus->SetPtRange(minPt,maxPt);
     protonMinus->SetThetaRange(thmin, thmax);

     gener->AddGenerator(pythia,"pythiaJets", 1);
     gener->AddGenerator(pionMinus,"pi-", 1);
     gener->AddGenerator(pionPlus,"pi+", 1);
     gener->AddGenerator(kaonMinus,"K-", 1);
     gener->AddGenerator(kaonPlus,"K+", 1);
     gener->AddGenerator(protonMinus,"P-", 1);
     gener->AddGenerator(protonPlus,"P+", 1);

  return gener;
}



AliGenerator* MbPythiaTuneATLAS_Flat()
{
      AliGenPythia* pythia = MbPythiaTuneATLAS();

      comment = comment.Append("; flat multiplicity distribution");

      // set high multiplicity trigger
      // this weight achieves a flat multiplicity distribution
      TH1 *weight = new TH1D("weight","weight",201,-0.5,200.5);
      weight->SetBinContent(1,5.49443);
      weight->SetBinContent(2,8.770816);
      weight->SetBinContent(6,0.4568624);
      weight->SetBinContent(7,0.2919915);
      weight->SetBinContent(8,0.6674189);
      weight->SetBinContent(9,0.364737);
      weight->SetBinContent(10,0.8818444);
      weight->SetBinContent(11,0.531885);
      weight->SetBinContent(12,1.035197);
      weight->SetBinContent(13,0.9394057);
      weight->SetBinContent(14,0.9643193);
      weight->SetBinContent(15,0.94543);
      weight->SetBinContent(16,0.9426507);
      weight->SetBinContent(17,0.9423649);
      weight->SetBinContent(18,0.789456);
      weight->SetBinContent(19,1.149026);
      weight->SetBinContent(20,1.100491);
      weight->SetBinContent(21,0.6350525);
      weight->SetBinContent(22,1.351941);
      weight->SetBinContent(23,0.03233504);
      weight->SetBinContent(24,0.9574557);
      weight->SetBinContent(25,0.868133);
      weight->SetBinContent(26,1.030998);
      weight->SetBinContent(27,1.08897);
      weight->SetBinContent(28,1.251382);
      weight->SetBinContent(29,0.1391099);
      weight->SetBinContent(30,1.192876);
      weight->SetBinContent(31,0.448944);
      weight->SetBinContent(32,1);
      weight->SetBinContent(33,1);
      weight->SetBinContent(34,1);
      weight->SetBinContent(35,1);
      weight->SetBinContent(36,0.9999997);
      weight->SetBinContent(37,0.9999997);
      weight->SetBinContent(38,0.9999996);
      weight->SetBinContent(39,0.9999996);
      weight->SetBinContent(40,0.9999995);
      weight->SetBinContent(41,0.9999993);
      weight->SetBinContent(42,1);
      weight->SetBinContent(43,1);
      weight->SetBinContent(44,1);
      weight->SetBinContent(45,1);
      weight->SetBinContent(46,1);
      weight->SetBinContent(47,0.9999999);
      weight->SetBinContent(48,0.9999998);
      weight->SetBinContent(49,0.9999998);
      weight->SetBinContent(50,0.9999999);
      weight->SetBinContent(51,0.9999999);
      weight->SetBinContent(52,0.9999999);
      weight->SetBinContent(53,0.9999999);
      weight->SetBinContent(54,0.9999998);
      weight->SetBinContent(55,0.9999998);
      weight->SetBinContent(56,0.9999998);
      weight->SetBinContent(57,0.9999997);
      weight->SetBinContent(58,0.9999996);
      weight->SetBinContent(59,0.9999995);
      weight->SetBinContent(60,1);
      weight->SetBinContent(61,1);
      weight->SetBinContent(62,1);
      weight->SetBinContent(63,1);
      weight->SetBinContent(64,1);
      weight->SetBinContent(65,0.9999999);
      weight->SetBinContent(66,0.9999998);
      weight->SetBinContent(67,0.9999998);
      weight->SetBinContent(68,0.9999999);
      weight->SetBinContent(69,1);
      weight->SetBinContent(70,1);
      weight->SetBinContent(71,0.9999997);
      weight->SetBinContent(72,0.9999995);
      weight->SetBinContent(73,0.9999994);
      weight->SetBinContent(74,1);
      weight->SetBinContent(75,1);
      weight->SetBinContent(76,1);
      weight->SetBinContent(77,1);
      weight->SetBinContent(78,0.9999999);
      weight->SetBinContent(79,1);
      weight->SetBinContent(80,1);
      weight->SetEntries(526);

      Int_t limit = weight->GetRandom();
      pythia->SetTriggerChargedMultiplicity(limit, 1.4);

      comment = comment.Append(Form("; multiplicity threshold set to %d in |eta| < 1.4", limit));

      return pythia;
}

AliGenerator* MbPhojet()
{
      comment = comment.Append(" pp: Pythia low-pt");
//
//    DPMJET
#if defined(__CINT__)
  gSystem->Load("libdpmjet");      // Parton density functions
  gSystem->Load("libTDPMjet");      // Parton density functions
#endif
      AliGenDPMjet* dpmjet = new AliGenDPMjet(-1);
      dpmjet->SetMomentumRange(0, 999999.);
      dpmjet->SetThetaRange(0., 180.);
      dpmjet->SetYRange(-12.,12.);
      dpmjet->SetPtRange(0,1000.);
      dpmjet->SetProcess(kDpmMb);
      dpmjet->SetEnergyCMS(energy);

      return dpmjet;
}


AliGenerator* MbPythiaTunePerugia0chadr()
{
      comment = comment.Append(" pp: Pythia (Perugia0) chadr (1 ccbar per event, 1 c-hadron in |y|<1.5, chadrons decay to hadrons");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-1.,1.);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyCharmppMNRwmi);
      pythia->SetEnergyCMS(energy);
//    Tune
//    320     Perugia 0
      pythia->SetTune(320);
      pythia->UseNewMultipleInteractionsScenario();
//
//    decays
      pythia->SetForceDecay(kHadronicDWithout4Bodies);

//    write only HF sub event
      pythia->SetStackFillOpt(AliGenPythia::kHeavyFlavor);
      return pythia;
}

AliGenerator* MbPythiaTunePerugia0bchadr()
{
      comment = comment.Append(" pp: Pythia (Perugia0) bchadr (1 bbbar per event, 1 c-hadron in |y|<1.5, chadrons decay to hadrons");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-1.5,1.5);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyBeautyppMNRwmi);
      pythia->SetEnergyCMS(energy);
//    Tune
//    320     Perugia 0
      pythia->SetTune(320);
      pythia->UseNewMultipleInteractionsScenario();
//
//    decays
      pythia->SetForceDecay(kHadronicDWithout4Bodies);

//    write only HF sub event
      pythia->SetStackFillOpt(AliGenPythia::kHeavyFlavor);
      return pythia;
}

AliGenerator* MbPythiaTunePerugia0cele()
{
      comment = comment.Append(" pp: Pythia (Perugia0) cele (1 ccbar per event, 1 electron in |y|<1.2");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      //pythia->SetYRange(-2.,2.);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyCharmppMNRwmi);
      pythia->SetEnergyCMS(energy);
//    Tune
//    320     Perugia 0
      pythia->SetTune(320);
      pythia->UseNewMultipleInteractionsScenario();
//
//    decays
      pythia->SetCutOnChild(1);
      pythia->SetPdgCodeParticleforAcceptanceCut(11);
      pythia->SetChildYRange(-1.2,1.2);
      pythia->SetChildPtRange(0,10000.);

//    write only HF sub event
      pythia->SetStackFillOpt(AliGenPythia::kHeavyFlavor);
      return pythia;
}

AliGenerator* MbPythiaTunePerugia0bele()
{
      comment = comment.Append(" pp: Pythia (Perugia0) bele (1 bbbar per event, 1 electron in |y|<1.2");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      //pythia->SetYRange(-2.,2.);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyBeautyppMNRwmi);
      pythia->SetEnergyCMS(energy);
//    Tune
//    320     Perugia 0
      pythia->SetTune(320);
      pythia->UseNewMultipleInteractionsScenario();
//
//    decays
      pythia->SetCutOnChild(1);
      pythia->SetPdgCodeParticleforAcceptanceCut(11);
      pythia->SetChildYRange(-1.2,1.2);
      pythia->SetChildPtRange(0,10000.);
//    write only HF sub event
      pythia->SetStackFillOpt(AliGenPythia::kHeavyFlavor);

      return pythia;
}

AliGenerator* MbPythiaTunePerugia0Jpsi2e()
{
  comment = comment.Append("Jpsi forced to dielectrons");
  AliGenParam *jpsi=0x0;
  if(JpsiHarderPt) jpsi = new AliGenParam(1, AliGenMUONlib::kJpsi, "CDF pp 8.8", "Jpsi");  // 8.8 TeV
  else jpsi = new AliGenParam(1, AliGenMUONlib::kJpsi, "PbPb 2.76", "Jpsi");  // 7 TeV
  jpsi->SetPtRange(0.,999.);
  jpsi->SetYRange(-1.0, 1.0);
  jpsi->SetPhiRange(0.,360.);
  jpsi->SetForceDecay(kDiElectron);
  return jpsi;
}

AliGenerator* MbPythiaTunePerugia0BtoJpsi2e()
{
      comment = comment.Append(" pp: Pythia (Perugia0) BtoJpsi (1 bbbar per event, 1 b-hadron in |y|<2, 1 J/psi in |y|<2");
//
//    Pythia
      AliGenPythia* pythia = new AliGenPythia(-1);
      pythia->SetMomentumRange(0, 999999.);
      pythia->SetThetaRange(0., 180.);
      pythia->SetYRange(-2.,2.);
      pythia->SetPtRange(0,1000.);
      pythia->SetProcess(kPyBeautyppMNRwmi);
      pythia->SetEnergyCMS(energy);
//    Tune
//    320     Perugia 0
      pythia->SetTune(320);
      pythia->UseNewMultipleInteractionsScenario();
//
//    decays
      pythia->SetCutOnChild(1);
      pythia->SetPdgCodeParticleforAcceptanceCut(443);
      pythia->SetChildYRange(-2,2);
      pythia->SetChildPtRange(0,10000.);
      //
//    decays
      pythia->SetForceDecay(kBJpsiDiElectron);
//    write only HF sub event
      pythia->SetStackFillOpt(AliGenPythia::kHeavyFlavor);
      return pythia;
}

void ProcessEnvironmentVars()
{
    // Run type
    if (gSystem->Getenv("CONFIG_RUN_TYPE")) {
      for (Int_t iRun = 0; iRun < kRunMax; iRun++) {
	if (strcmp(gSystem->Getenv("CONFIG_RUN_TYPE"), pprRunName[iRun])==0) {
	  proc = (PDC06Proc_t)iRun;
	  cout<<"Run type set to "<<pprRunName[iRun]<<endl;
	}
      }
    }


    // Hard process
    if (gSystem->Getenv("CONFIG_RUN_HTYPE")) {
      hproc = atoi(gSystem->Getenv("CONFIG_RUN_HTYPE"));
      cout<<"Hard Process set to "<< hproc <<endl;
    }

    // Field
    if (gSystem->Getenv("CONFIG_FIELD")) {
      for (Int_t iField = 0; iField < kFieldMax; iField++) {
	if (strcmp(gSystem->Getenv("CONFIG_FIELD"), pprField[iField])==0) {
	  mag = (Mag_t)iField;
	  cout<<"Field set to "<<pprField[iField]<<endl;
	}
      }
    }

    // Energy
    if (gSystem->Getenv("CONFIG_ENERGY")) {
      energy = atoi(gSystem->Getenv("CONFIG_ENERGY"));
      cout<<"Energy set to "<<energy<<" GeV"<<endl;
    }

    // Random Number seed
    if (gSystem->Getenv("CONFIG_SEED")) {
      seed = atoi(gSystem->Getenv("CONFIG_SEED"));
    }

  // Impact param
    if (gSystem->Getenv("CONFIG_BMIN")) {
      bMin = atof(gSystem->Getenv("CONFIG_BMIN"));
    }

    if (gSystem->Getenv("CONFIG_BMAX")) {
      bMax = atof(gSystem->Getenv("CONFIG_BMAX"));
    }
    cout<<"Impact parameter in ["<<bMin<<","<<bMax<<"]"<<endl;
}

AliGenerator* Hijing()
{
    AliGenHijing *gener = new AliGenHijing(-1);
// centre of mass energy
    gener->SetEnergyCMS(2760.);
    gener->SetImpactParameterRange(bMin, bMax);
// reference frame
    gener->SetReferenceFrame("CMS");
// projectile
     gener->SetProjectile("A", 208, 82);
     gener->SetTarget    ("A", 208, 82);
// tell hijing to keep the full parent child chain
     gener->KeepFullEvent();
// enable jet quenching
     gener->SetJetQuenching(1);
// enable shadowing
     gener->SetShadowing(1);
// Don't track spectators
     gener->SetSpectators(0);
// kinematic selection
     gener->SetSelectAll(0);
     return gener;
}

AliGenerator* Hijing2000()
{
    AliGenHijing *gener = (AliGenHijing*) Hijing();
    gener->SetJetQuenching(0);
    gener->SetPtHardMin (2.3);
    return gener;
}

AliGenerator* HijingNuclei()
{
  AliGenCocktail *gener = new AliGenCocktail();
  /*gener->SetProjectile("A", 208, 82);
  gener->SetTarget    ("A", 208, 82);
  gener->SetEnergyCMS(2760.);
*/
  // 1. Hijing
  AliGenHijing *gh = (AliGenHijing*) Hijing();
  gh->SetImpactParameterRange(14.9, 30.0);

AliGenBox *box2 = new AliGenBox(100);
box2->SetPart(211);
//pythia->SetMomentumRange(0, 999999.)
box2->SetPtRange(1.6,20.);
box2->SetPhiRange(0., 360.);
box2->SetYRange(-1,1);

AliGenBox *box3 = new AliGenBox(100);
box3->SetPart(-211);
box3->SetPtRange(1.6,20.);
//box3->SetPtRange(1.5,20.);
box3->SetPhiRange(0., 360.);
box3->SetYRange(-1,1);

  // 2. Deuteron
  /*AliGenBox *box2 = new AliGenBox(10);
  box2->SetPart(1000010020);
  box2->SetPtRange(0., 10.);
  box2->SetPhiRange(0., 360.);
  box2->SetYRange(-1,1);

  // 3. Anti-Deuteron
  AliGenBox *box3 = new AliGenBox(10);
  box3->SetPart(-1000010020);
  box3->SetPtRange(0., 10.);
  box3->SetPhiRange(0., 360.);
  box3->SetYRange(-1,1);

  // 4. He-3
  AliGenBox *box4 = new AliGenBox(10);
  box4->SetPart(1000020030);
  box4->SetPtRange(0., 10.);
  box4->SetPhiRange(0., 360.);
  box4->SetYRange(-1,1);

  // 5. Anti-He-3
  AliGenBox *box5 = new AliGenBox(10);
  box5->SetPart(-1000020030);
  box5->SetPtRange(0., 10.);
  box5->SetPhiRange(0., 360.);
  box5->SetYRange(-1,1);

  // 6. Tritons
  AliGenBox *box6 = new AliGenBox(10);
  box6->SetPart(1000010030);
  box6->SetPtRange(0., 10.);
  box6->SetPhiRange(0., 360.);
  box6->SetYRange(-1,1);

  // 7. Anti-Tritons
  AliGenBox *box7 = new AliGenBox(10);
  box7->SetPart(-1000010030);
  box7->SetPtRange(0., 10.);
  box7->SetPhiRange(0., 360.);
  box7->SetYRange(-1,1);

  // 8. He-4
  AliGenBox *box8 = new AliGenBox(10);
  box8->SetPart(1000020040);
  box8->SetPtRange(0., 10.);
  box8->SetPhiRange(0., 360.);
  box8->SetYRange(-1,1);

  // 9. Anti-He-4
  AliGenBox *box9 = new AliGenBox(10);
  box9->SetPart(-1000020040);
  box9->SetPtRange(0., 10.);
  box9->SetPhiRange(0., 360.);
  box9->SetYRange(-1,1);

    // 10. Hyperhelium 5
    AliGenBox *box10 = new AliGenBox(20);
    box10->SetPart(1010020050);
    box10->SetPtRange(0., 10.);
    box10->SetPhiRange(0., 360.);
    box10->SetYRange(-1,1);

    // 11. Anti-Hyperhelium 5
    AliGenBox *box11 = new AliGenBox(20);
    box11->SetPart(-1010020050);
    box11->SetPtRange(0., 10.);
    box11->SetPhiRange(0., 360.);
    box11->SetYRange(-1,1);

    // 12. Double Hyper Hydrogen 4
    AliGenBox *box12 = new AliGenBox(20);
    box12->SetPart(1020010040);
    box12->SetPtRange(0., 10.);
    box12->SetPhiRange(0., 360.);
    box12->SetYRange(-1,1);

    // 13. Anti-Double Hyper Hydrogen 4
    AliGenBox *box13 = new AliGenBox(20);
    box13->SetPart(-1020010040);
    box13->SetPtRange(0., 10.);
    box13->SetPhiRange(0., 360.);
    box13->SetYRange(-1,1);

	// 14. HyperHydrogen3
    AliGenBox *box14 = new AliGenBox(20);
    box14->SetPart(1010010030);
    box14->SetPtRange(0., 10.);
    box14->SetPhiRange(0., 360.);
    box14->SetYRange(-1,1);

	// 15. AntiHyperHydrogen3
    AliGenBox *box15 = new AliGenBox(20);
    box15->SetPart(-1010010030);
    box15->SetPtRange(0., 10.);
    box15->SetPhiRange(0., 360.);
    box15->SetYRange(-1,1);

	// 16. HyperHydrogen4
    AliGenBox *box16 = new AliGenBox(20);
    box16->SetPart(1010010040);
    box16->SetPtRange(0., 10.);
    box16->SetPhiRange(0., 360.);
    box16->SetYRange(-1,1);

	// 17. AntiHyperHydrogen4
    AliGenBox *box17 = new AliGenBox(20);
    box17->SetPart(-1010010040);
    box17->SetPtRange(0., 10.);
    box17->SetPhiRange(0., 360.);
    box17->SetYRange(-1,1);

	// 18. Hyperhelium 4
    AliGenBox *box18 = new AliGenBox(20);
    box18->SetPart(1010020040);
    box18->SetPtRange(0., 10.);
    box18->SetPhiRange(0., 360.);
    box18->SetYRange(-1,1);

    // 19. Anti-Hyperhelium 4
    AliGenBox *box19 = new AliGenBox(20);
    box19->SetPart(-1010020040);
    box19->SetPtRange(0., 10.);
    box19->SetPhiRange(0., 360.);
    box19->SetYRange(-1,1);

    // 20. LambdaLambda
    AliGenBox *box20 = new AliGenBox(20);
    box20->SetPart(1020000021);
    box20->SetPtRange(0., 10.);
    box20->SetPhiRange(0., 360.);
    box20->SetYRange(-1,1);

    // 21. AntiLambdaLambda
    AliGenBox *box21 = new AliGenBox(20);
    box21->SetPart(-1020000021);
    box21->SetPtRange(0., 10.);
    box21->SetPhiRange(0., 360.);
    box21->SetYRange(-1,1);

     // 22. OmegaOmega
    AliGenBox *box22 = new AliGenBox(20);
    box22->SetPart(1060020020);
    box22->SetPtRange(0., 10.);
    box22->SetPhiRange(0., 360.);
    box22->SetYRange(-1,1);

    // 23. AntiOmegaOmega
    AliGenBox *box23 = new AliGenBox(20);
    box23->SetPart(-1060020020);
    box23->SetPtRange(0., 10.);
    box23->SetPhiRange(0., 360.);
    box23->SetYRange(-1,1);

    // 24. OmegaProton
    AliGenBox *box24 = new AliGenBox(20);
    box24->SetPart(1030000020);
    box24->SetPtRange(0., 10.);
    box24->SetPhiRange(0., 360.);
    box24->SetYRange(-1,1);

    // 25. AntiOmegaProton
    AliGenBox *box25 = new AliGenBox(20);
    box25->SetPart(-1030000020);
    box25->SetPtRange(0., 10.);
    box25->SetPhiRange(0., 360.);
    box25->SetYRange(-1,1);

    // 26. LambdaNeutronNeutron
    AliGenBox *box26 = new AliGenBox(20);
    box26->SetPart(1010000030);
    box26->SetPtRange(0., 10.);
    box26->SetPhiRange(0., 360.);
    box26->SetYRange(-1,1);

    // 27. AntiLambdaNeutronNeutron
    AliGenBox *box27 = new AliGenBox(20);
    box27->SetPart(-1010000030);
    box27->SetPtRange(0., 10.);
    box27->SetPhiRange(0., 360.);
    box27->SetYRange(-1,1);

     // 28. Xi0P
    AliGenBox *box28 = new AliGenBox(20);
    box28->SetPart(1020010020);
    box28->SetPtRange(0., 10.);
    box28->SetPhiRange(0., 360.);
    box28->SetYRange(-1,1);

    // 29. AntiXi0P
    AliGenBox *box29 = new AliGenBox(20);
    box29->SetPart(-1020010020);
    box29->SetPtRange(0., 10.);
    box29->SetPhiRange(0., 360.);
    box29->SetYRange(-1,1);

    // 30. LambdaN
    AliGenBox *box30 = new AliGenBox(20);
    box30->SetPart(1010000020);
    box30->SetPtRange(0., 10.);
    box30->SetPhiRange(0., 360.);
    box30->SetYRange(-1,1);

     // 31. AntiLambdaN
    AliGenBox *box31 = new AliGenBox(20);
    box31->SetPart(-1010000020);
    box31->SetPtRange(0., 10.);
    box31->SetPhiRange(0., 360.);
    box31->SetYRange(-1,1);

*/


  gener->AddGenerator(gh,"hijing",1);
  gener->AddGenerator(box2,"fbox2",1);
  gener->AddGenerator(box3,"fbox3",1);
  /*gener->AddGenerator(box4,"fbox4",1);
  gener->AddGenerator(box5,"fbox5",1);
  gener->AddGenerator(box6,"fbox6",1);
  gener->AddGenerator(box7,"fbox7",1);
  gener->AddGenerator(box8,"fbox8",1);
  gener->AddGenerator(box9,"fbox9",1);
  gener->AddGenerator(box10,"fbox10",1);
  gener->AddGenerator(box11,"fbox11",1);
  gener->AddGenerator(box12,"fbox12",1);
  gener->AddGenerator(box13,"fbox13",1);
  gener->AddGenerator(box14,"fbox14",1);
  gener->AddGenerator(box15,"fbox15",1);
  gener->AddGenerator(box16,"fbox16",1);
  gener->AddGenerator(box17,"fbox17",1);
  gener->AddGenerator(box18,"fbox18",1);
  gener->AddGenerator(box19,"fbox19",1);
  gener->AddGenerator(box20,"fbox20",1);
  gener->AddGenerator(box21,"fbox21",1);
  gener->AddGenerator(box22,"fbox22",1);
  gener->AddGenerator(box23,"fbox23",1);
  gener->AddGenerator(box24,"fbox24",1);
  gener->AddGenerator(box25,"fbox25",1);
  gener->AddGenerator(box26,"fbox26",1);
  gener->AddGenerator(box27,"fbox27",1);
  gener->AddGenerator(box28,"fbox28",1);
  gener->AddGenerator(box29,"fbox29",1);
  gener->AddGenerator(box30,"fbox30",1);
  gener->AddGenerator(box31,"fbox31",1);*/

  return gener;
}

AliGenerator* Hijing2000HF(Int_t typeHF, Float_t ptHmin, Float_t ptHmax, Float_t ptHJmin, Float_t ptHJmax)
{
  comment = comment.Append(" PbPb: Hjing2000 + pythia events for HF signals");

  AliGenCocktail *cocktail = new AliGenCocktail();
  cocktail->SetProjectile("A", 208, 82);
  cocktail->SetTarget    ("A", 208, 82);
  cocktail->SetEnergyCMS(energy);
  //
  // 1 Hijing event
  TFormula* one    = new TFormula("one",    "1.");
  // provides underlying event and collision geometry
  if  (proc == kHijing2000HF) {
  	AliGenHijing *hijing = Hijing2000();
  	cocktail->AddGenerator(hijing,"hijing",1);
  }
  if  (proc == kAmptHF) {
  	AliGenAmpt *ampt = Ampt();
  	cocktail->AddGenerator(ampt,"ampt",1);
  }
  //
  // N Pythia Heavy Flavor events
  // N is determined from impact parameter according to the following formula
  TFormula* formula = new TFormula("Signals",
				   "60. * (x < 5.) + 80. * (1. - x/20.)*(x>5.)");
  //
  AliGenerator* pythiaHF = 0x0;
  switch(typeHF) {
  case 0:
    pythiaHF = MbPythiaTunePerugia0chadr();
    break;
  case 1:
    pythiaHF = MbPythiaTunePerugia0bchadr();
    break;
  case 2:
    pythiaHF = MbPythiaTunePerugia0cele();
    break;
  case 3:
    pythiaHF = MbPythiaTunePerugia0bele();
    break;
  case 4:
    pythiaHF = MbPythiaTunePerugia0Jpsi2e();
    break;
  case 5:
    pythiaHF = MbPythiaTunePerugia0BtoJpsi2e();
    break;
  case 6:
    pythiaHF = PythiaJets();
    break;
  default:
    pythiaHF = MbPythiaTunePerugia0chadr();
    break;
  }
  if (typeHF != 6) {
        cocktail->AddGenerator(pythiaHF, "pythiaHF",   1, formula);
  } else {
  	cocktail->AddGenerator(pythiaHF, "pythiaJets", 1, one);
  }
  if (typeHF  < 4) ((AliGenPythia*)pythiaHF)->SetPtHard(ptHmin,  ptHmax);
  if (typeHF == 6) ((AliGenPythia*)pythiaHF)->SetPtHard(ptHJmin, ptHJmax);
  //
  // Pi0
  // Flat pt spectrum in range 0..20
  // Set pseudorapidity range from -1.2 to 1.2
  //
  Float_t thmin          = (180./TMath::Pi())*2.*atan(exp(-1.2));
  Float_t thmax          = (180./TMath::Pi())*2.*atan(exp( 1.2));

  TFormula* neutralsF    = new TFormula("neutrals",    "20.+ 80.*exp(- 0.5 * x * x / 5.12 / 5.12)");
  AliGenPHOSlib *plib = new AliGenPHOSlib();

  AliGenParam *genPi0 = new AliGenParam(1, plib, AliGenPHOSlib::kPi0Flat);
  genPi0->SetPhiRange(0., 360.) ;
  genPi0->SetYRange(-1.2, 1.2) ;
  genPi0->SetPtRange(0., 50.) ;
  cocktail->AddGenerator(genPi0, "pi0", 1., neutralsF);

  AliGenParam *genEta = new AliGenParam(1, plib, AliGenPHOSlib::kEtaFlat);
  genEta->SetPhiRange(0., 360.) ;
  genEta->SetYRange(-1.2, 1.2) ;
  genEta->SetPtRange(0., 50.) ;
  cocktail->AddGenerator(genEta, "eta", 1., neutralsF);

  AliGenBox* genGamma = new AliGenBox(1);
  genGamma->SetPart(111);
  genGamma->SetPtRange(0, 50.);
  genGamma->SetThetaRange(thmin, thmax);
  cocktail->AddGenerator(genGamma, "gamma", 1., neutralsF);
  //
  // Jpsi->mu+ mu-
  //
  jpsi2m = new AliGenParam(1, AliGenMUONlib::kJpsi, "PbPb 2.76", "Jpsi");  // 7 TeV
  jpsi2m->SetPtRange(0.,999.);
  jpsi2m->SetYRange(-4.2, -2.3);
  jpsi2m->SetPhiRange(0.,360.);
  jpsi2m->SetForceDecay(kDiMuon);
  jpsi2m->SetCutOnChild(1);
  jpsi2m->SetChildPhiRange(0.,360.);
  jpsi2m->SetChildThetaRange(168.5,178.5);
  cocktail->AddGenerator(jpsi2m, "Jpsi2M", 1, one);
  //
  // Chi_c -> J/Psi + gamma, J/Psi -> e+e-
  //
  AliGenParam* genChic = new AliGenParam(1, AliGenMUONlib::kChic,"default","Chic");
  genChic->SetMomentumRange(0, 999.);        // Wide cut on the momentum
  genChic->SetPtRange(0, 100.);              // Wide cut on Pt
  genChic->SetYRange(-2.5, 2.5);
  genChic->SetCutOnChild(1);                 // Enable cuts on decay products
  genChic->SetChildPhiRange(0., 360.);
  genChic->SetChildThetaRange(thmin, thmax); // In the acceptance of the Central Barrel
  genChic->SetForceDecay(kChiToJpsiGammaToElectronElectron); // Chi_c -> J/Psi + gamma, J/Psi -> e+e-
  cocktail->AddGenerator(genChic, "Chi_c", 1, one);
  //
  // Strangeness Cocktail
  //
  Int_t prt=0;
  AliGenSTRANGElib *lib = new AliGenSTRANGElib();

  prt = AliGenSTRANGElib::kLambda;
  AliGenParam *gl = new AliGenParam(2 * 4, prt,  // 4 Lambda and 4 anti-Lambda
				    lib->GetPt(prt),lib->GetY(prt),lib->GetIp(prt)); // mt-scaled pt
  gl->SetPtRange(0, 999);
  gl->SetYRange(-1.2, 1.2);
  gl->SetForceDecay(kNoDecay);

  AliGenBox *gk0 = new AliGenBox(1);         // 1 K0s, flat pt
  gk0->SetPart(310);
  gk0->SetPtRange(0, 30);
  gk0->SetYRange(-1.2, 1.2);

  AliGenBox *gfla = new AliGenBox(1);        // 1 Lambda, flat pt
  gfla->SetPart(3122);
  gfla->SetPtRange(0, 30);
  gfla->SetYRange(-1.2, 1.2);

  AliGenBox *gxi = new AliGenBox(3);         // 3 Xi-, flat pt
  gxi->SetPart(3312);
  gxi->SetPtRange(0, 15);
  gxi->SetYRange(-1.2, 1.2);

  AliGenBox *gx0 = new AliGenBox(3);         // 3 Xi0, flat pt
  gx0->SetPart(3322);
  gx0->SetPtRange(0, 15);
  gx0->SetYRange(-1.2, 1.2);

  AliGenBox *gom = new AliGenBox(2);         // 2 Omega-, flat pt
  gom->SetPart(3334);
  gom->SetPtRange(0, 12);
  gom->SetYRange(-1.2, 1.2);

  cocktail->AddGenerator(gl,  "mt-scaled lambda"   , 1, one);
  cocktail->AddGenerator(gk0, "flat pt k0s"        , 1, one);
  cocktail->AddGenerator(gfla,"flat pt lambda"     , 1, one);
  cocktail->AddGenerator(gxi, "flat pt xi-"        , 1, one);
  cocktail->AddGenerator(gx0, "flat pt xi0"        , 1, one);
  cocktail->AddGenerator(gom, "flat pt omega-"     , 1, one);
//
// High pt charged particles
//

  AliGenBox* pionPlus = new AliGenBox(1); // pions+
  pionPlus->SetPart(211);
  pionPlus->SetPtRange(6., 50.);
  pionPlus->SetYRange(-1.2, 1.2);

  AliGenBox* pionMinus = new AliGenBox(1); // pions-
  pionMinus->SetPart(-211);
  pionMinus->SetPtRange(6., 50.);
  pionMinus->SetYRange(-1.2, 1.2);

  AliGenBox* kaonPlus = new AliGenBox(1); // kaons+
  kaonPlus->SetPart(321);
  kaonPlus->SetPtRange(6., 50.);
  kaonPlus->SetYRange(-1.2, 1.2);

  AliGenBox* kaonMinus = new AliGenBox(1); // kaons-
  kaonMinus->SetPart(-321);
  kaonMinus->SetPtRange(6., 50.);
  kaonMinus->SetYRange(-1.2, 1.2);

  AliGenBox* protonPlus = new AliGenBox(1); // protons+
  protonPlus->SetPart(2212);
  protonPlus->SetPtRange(6., 50.);
  protonPlus->SetYRange(-1.2, 1.2);

  AliGenBox* protonMinus = new AliGenBox(1); // protons-
  protonMinus->SetPart(-2212);
  protonMinus->SetPtRange(6., 50.);
  protonMinus->SetYRange(-1.2, 1.2);

  TString centralityFormula("(5. * (x < 5.) + 20. / 3. * (1. - x / 20.) * (x > 5.))");

  // weight roughly according to abundance
  TFormula* formulaHPTPions   = new TFormula("High Pt", centralityFormula + " * 6");
  TFormula* formulaHPTKaons   = new TFormula("High Pt", centralityFormula + " * 3");
  TFormula* formulaHPTProtons = new TFormula("High Pt", centralityFormula + " * 1");

  cocktail->AddGenerator(pionPlus,    "flat pion+" ,   1, formulaHPTPions);
  cocktail->AddGenerator(pionMinus,   "flat pion-" ,   1, formulaHPTPions);
  cocktail->AddGenerator(kaonPlus,    "flat kaon+" ,   1, formulaHPTKaons);
  cocktail->AddGenerator(kaonMinus,   "flat kaon-" ,   1, formulaHPTKaons);
  cocktail->AddGenerator(protonPlus,  "flat proton+" , 1, formulaHPTProtons);
  cocktail->AddGenerator(protonMinus, "flat proton-" , 1, formulaHPTProtons);

  return cocktail;
}

AliGenerator* Hydjet()
{
  AliGenUHKM *genHi = new AliGenUHKM(-1);
  genHi->SetAllParametersLHC();
  genHi->SetProjectile("A", 208, 82);
  genHi->SetTarget    ("A", 208, 82);
  genHi->SetEcms(2760);
  genHi->SetEnergyCMS(2760.);
  genHi->SetBmin(bMin);
  genHi->SetBmax(bMax);
  genHi->SetPyquenPtmin(9);
  return genHi;
}

AliGenerator* Dpmjet()
{
  AliGenDPMjet* dpmjet = new AliGenDPMjet(-1);
  dpmjet->SetEnergyCMS(energy);
  dpmjet->SetProjectile("A", 208, 82);
  dpmjet->SetTarget    ("A", 208, 82);
  dpmjet->SetImpactParameterRange(bMin, bMax);
  dpmjet->SetPi0Decay(0);
  return dpmjet;
}

AliGenerator* Ampt()
{

  AliGenAmpt *genHi = new AliGenAmpt(-1);
  genHi->SetEnergyCMS(2760);
  genHi->SetReferenceFrame("CMS");
  genHi->SetProjectile("A", 208, 82);
  genHi->SetTarget    ("A", 208, 82);
  genHi->SetPtHardMin (2);
  genHi->SetImpactParameterRange(bMin,bMax);
  genHi->SetJetQuenching(0); // enable jet quenching
  genHi->SetShadowing(1);    // enable shadowing
  genHi->SetDecaysOff(1);    // neutral pion and heavy particle decays switched off
  genHi->SetSpectators(0);   // track spectators
  genHi->KeepFullEvent();
  genHi->SetSelectAll(0);
  return genHi;
}

AliGenerator* HijingPlusJetsPlusBox()
{
    AliGenCocktail *gener  = new AliGenCocktail();

    AliGenHijing *hijing = new AliGenHijing(-1);
// Centre of mass energy
    hijing->SetEnergyCMS(2760.);
    hijing->SetImpactParameterRange(0. , 30.);
// reference frame
    hijing->SetReferenceFrame("CMS");
// projectile
     hijing->SetProjectile("A", 208, 82);
     hijing->SetTarget    ("A", 208, 82);
// tell hijing to keep the full parent child chain
     hijing->KeepFullEvent();
// enable jet quenching
     hijing->SetJetQuenching(0);
// enable shadowing
     hijing->SetShadowing(1);
// neutral pion and heavy particle decays switched off
     hijing->SetDecaysOff(1);
// Don't track spectators
     hijing->SetSpectators(0);
     hijing->SetSpectators(0);
// kinematic selection
     hijing->SetSelectAll(0);
//
     hijing->SetPtHardMin (2.3);


     AliGenPythia * pythia = new AliGenPythia(-1);
     pythia->SetEnergyCMS(2760.);        //        Centre of mass energy
     pythia->SetProcess(kPyJets);        //        Process type
     pythia->SetJetEtaRange(-1.5, 1.5);  //        Final state kinematic cuts
     pythia->SetStrucFunc(kCTEQ6l);
     pythia->SetJetEtRange(150., 1000.);
     pythia->SetPtHard(135., 1000.);
     pythia->SetPycellParameters(2.2, 300, 432, 0., 4., 5., 0.7);
     pythia->SetForceDecay(kAll);

     //
     // High pt charged particles
     // add 100 pions, 2 protons, 2 kaons
     //
     const  Double_t minPt = 50.;
     const  Double_t maxPt = 100.;

     Float_t thmin          = (180./TMath::Pi())*2.*atan(exp(-1.0));
     Float_t thmax          = (180./TMath::Pi())*2.*atan(exp( 1.0));

     AliGenBox* pionPlus = new AliGenBox(50);     // pions+
     pionPlus->SetPart(211);
     pionPlus->SetPtRange(minPt,maxPt);
     pionPlus->SetThetaRange(thmin, thmax);

     AliGenBox* pionMinus = new AliGenBox(50);     // pions-
     pionMinus->SetPart(-211);
     pionMinus->SetPtRange(minPt,maxPt);
     pionMinus->SetThetaRange(thmin, thmax);

     AliGenBox* kaonPlus = new AliGenBox(1);     // kaons+
     kaonPlus->SetPart(321);
     kaonPlus->SetPtRange(minPt,maxPt);
     kaonPlus->SetThetaRange(thmin, thmax);

     AliGenBox* kaonMinus = new AliGenBox(1);     // kaons-
     kaonMinus->SetPart(-321);
     kaonMinus->SetPtRange(minPt,maxPt);
     kaonMinus->SetThetaRange(thmin, thmax);

     AliGenBox* protonPlus = new AliGenBox(1);     // protons+
     protonPlus->SetPart(2212);
     protonPlus->SetPtRange(minPt,maxPt);
     protonPlus->SetThetaRange(thmin, thmax);

     AliGenBox* protonMinus = new AliGenBox(1);     // protons-
     protonMinus->SetPart(-2212);
     protonMinus->SetPtRange(minPt,maxPt);
     protonMinus->SetThetaRange(thmin, thmax);

     gener->AddGenerator(hijing,  "HIJING PbPb",    1);
     gener->AddGenerator(pythia,  "Pythia Jets",    1);
     gener->AddGenerator(pionPlus,  "pi+",     1);
     gener->AddGenerator(pionMinus,  "pi-",    1);
     gener->AddGenerator(kaonPlus,  "K+",    1);
     gener->AddGenerator(kaonMinus,  "K-",    1);
     gener->AddGenerator(protonPlus,  "P+",    1);
     gener->AddGenerator(protonMinus,  "P-",    1);


     return gener;
}


AliGenerator* HijingPlusJets()
{
    AliGenCocktail *gener  = new AliGenCocktail();

    AliGenHijing *hijing = new AliGenHijing(-1);
// Centre of mass energy
    hijing->SetEnergyCMS(2760.);
    hijing->SetImpactParameterRange(0. , 30.);
// reference frame
    hijing->SetReferenceFrame("CMS");
// projectile
     hijing->SetProjectile("A", 208, 82);
     hijing->SetTarget    ("A", 208, 82);
// tell hijing to keep the full parent child chain
     hijing->KeepFullEvent();
// enable jet quenching
     hijing->SetJetQuenching(0);
// enable shadowing
     hijing->SetShadowing(1);
// neutral pion and heavy particle decays switched off
     hijing->SetDecaysOff(1);
// Don't track spectators
     hijing->SetSpectators(0);
     hijing->SetSpectators(0);
// kinematic selection
     hijing->SetSelectAll(0);
//
     hijing->SetPtHardMin (2.3);


     AliGenPythia * pythia = new AliGenPythia(-1);
     pythia->SetEnergyCMS(2760.);        //        Centre of mass energy
     pythia->SetProcess(kPyJets);        //        Process type
     pythia->SetJetEtaRange(-1.5, 1.5);  //        Final state kinematic cuts
     pythia->SetStrucFunc(kCTEQ6l);
     if (proc == kHijingPlusJets1) {
        pythia->SetJetEtRange(50., 80.);
	pythia->SetPtHard(45., 1000.);
     	pythia->SetPycellParameters(2.2, 300, 432, 0., 4., 5., 0.7);
     } else if (proc == kHijingPlusJets2) {
        pythia->SetJetEtRange(80., 1000.);
	pythia->SetPtHard(72., 1000.);
     	pythia->SetPycellParameters(2.2, 300, 432, 0., 4., 5., 0.7);
     } else if (proc == kHijingPlusJets3) {
        pythia->SetJetEtRange(150., 1000.);
	pythia->SetPtHard(135., 1000.);
     	pythia->SetPycellParameters(2.2, 300, 432, 0., 4., 5., 0.7);
     }

     pythia->SetForceDecay(kAll);
//
//
//
     AliGenPythia * pythia1 = new AliGenPythia(-1);
     pythia1->SetEnergyCMS(2760.);        //        Centre of mass energy
     pythia1->SetProcess(kPyJets);        //        Process type
     pythia1->SetJetEtaRange(-1.5, 1.5);  //        Final state kinematic cuts
     pythia1->SetStrucFunc(kCTEQ6l);
     if (proc == kHijingPlusJets3) {
        pythia1->SetJetEtRange(200., 1000.);
	pythia1->SetPtHard(180., 1000.);
     	pythia1->SetPycellParameters(2.2, 300, 432, 0., 4., 5., 0.7);
     }
     pythia1->SetForceDecay(kAll);
//
//
//
     AliGenPythia * pythia2 = new AliGenPythia(-1);
     pythia2->SetEnergyCMS(2760.);        //        Centre of mass energy
     pythia2->SetProcess(kPyJets);        //        Process type
     pythia2->SetJetEtaRange(-1.5, 1.5);  //        Final state kinematic cuts
     pythia2->SetStrucFunc(kCTEQ6l);
     if (proc == kHijingPlusJets3) {
        pythia2->SetJetEtRange(250., 1000.);
	pythia2->SetPtHard(225., 1000.);
     	pythia2->SetPycellParameters(2.2, 300, 432, 0., 4., 5., 0.7);
     }
     pythia2->SetForceDecay(kAll);

     gener->AddGenerator(hijing,  "HIJING PbPb",    1);
     gener->AddGenerator(pythia,  "Pythia Jet 1",    1);
     gener->AddGenerator(pythia1,  "Pythia Jet 2",    1);
     gener->AddGenerator(pythia2,  "Pythia Jet 3",    1);

     return gener;
}
