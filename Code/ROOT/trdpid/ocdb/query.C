//
// created:       2016-10-07
// last changed:  2016-10-07
//
// based on AliTRDcheckConfig.C
//


#if !defined(__CINT__) || defined(__MAKECINT__)

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <TFile.h>
#include <TString>
#include <TSubString>

#include <AliCDBEntry.h>
#include <AliTRDCalDCS.h>
#include <AliTRDCalDCSv2.h>
#include <AliCDBManager.h>

#endif

using std::cout;
using std::endl;

//Int_t   calVer;
//Int_t   configtag;
//TString configversion;
//TString configname;
//

struct runinfo_t
{

  // GRP info
  TTimeStamp start_time;
  TTimeStamp end_time;
  
  float  cavern_pressure;
  //float  surface_pressure;
  //int    detector_mask;

  float anode_voltage[540];
  float drift_voltage[540];

  float gain[540];
  float vdrift[540];
  float ExB[540];
  
};



//__________________________________________________________________________
void query(Int_t year=2016, Int_t run=265377)
{

  // set up the connection to the OCDB
  AliCDBManager* man = AliCDBManager::Instance();

  if (0) {
    man->SetDefaultStorage
      (Form("local:///cvmfs/alice-ocdb.cern.ch/calibration/data/%d/OCDB/",year));
  } else {
    man->SetDefaultStorage
      (Form("alien://folder=/alice/data/%d/OCDB/",year));
  }
  
  man->SetCacheFlag(kTRUE);
  man->SetRun(run);

  runinfo_t runinfo;

  // -----------------------------------------------------------------------
  // Get the GRP data. Only runs with a corresponding GRP entry in the OCDB
  // will be processed
  AliCDBEntry *entry = man->Get("GRP/GRP/Data",run);
  AliGRPObject* grp = dynamic_cast<AliGRPObject*>entry->GetObject();

  if (!grp) return;

  TTimeStamp start (grp->GetTimeStart());
  TTimeStamp end   (grp->GetTimeEnd());
  
  runinfo.start_time = start;
  runinfo.end_time = end;
  runinfo.cavern_pressure =
    grp->GetCavernAtmosPressure()->MakeGraph()->GetMean();
  //runinfo.surface_pressure = grp->GetSurfaceAtmosPressure();
  //runinfo.detector_mask = grp->GetDetectorMask();
  
  if (0) {
    TCanvas* c = new TCanvas("cavern pressure", "cavern pressure");
    grp->GetCavernAtmosPressure()->MakeGraph()->Draw("alp");
  }
  
  // -----------------------------------------------------------------------
  // HV data

  AliCDBEntry *entry = man->Get("TRD/Calib/trd_hvAnodeUmon",run);
  AliTRDSensorArray* arr = dynamic_cast<AliTRDSensorArray*>entry->GetObject();

  for (int i=0;i<arr->NumSensors();i++) {
    
    //cout << arr->GetSensorNum(i)->GetIdDCS() << "  "
    //<< arr->GetSensorNum(i)->GetStringID() << endl;

    //cout << arr->GetSensorNum(i)->Eval(start,true) << endl;;

    // somehow, this array does not produce a TGraph, therefore we use
    // the value at the start of the run
    runinfo.anode_voltage[i] = arr->GetSensorNum(i)->Eval(start,true);
  }
  
  AliCDBEntry *entry = man->Get("TRD/Calib/trd_hvDriftUmon",run);
  AliTRDSensorArray* arr = dynamic_cast<AliTRDSensorArray*>entry->GetObject();

  for (int i=0;i<arr->NumSensors();i++) {
    // somehow, this array does not produce a TGraph, therefore we use
    // the value at the start of the run
    runinfo.drift_voltage[i] = arr->GetSensorNum(i)->Eval(start,true);
  }
  

  
//    // time step for time dependent information (change this if you
//    // need something else)
//    UInt_t dTime = TMath::Max((endTime-startTime)/20, Long_t(5*60));
//
//    // get monitoring information
//    AliTRDSensorArray *anodeISensors = 0;
//    AliTRDSensorArray *anodeUSensors = 0;
//    AliTRDSensorArray *driftISensors = 0;
//    AliTRDSensorArray *driftUSensors = 0;
//    AliTRDSensorArray *temperatureSensors = 0;
//    AliTRDSensorArray *chamberStatusSensors = 0;
//    AliTRDSensorArray *overpressureSensors = 0;
//    AliTRDSensorArray *gasCO2Sensors = 0;
//    AliTRDSensorArray *gasH2OSensors = 0;
//    AliTRDSensorArray *gasO2Sensors = 0;
//    //  AliTRDSensorArray *adcClkPhaseSensors = 0;
//
//    if(getHVInfo) {
//      // anode hv currents (per chamber)
//      if((entry = GetCDBentry("TRD/Calib/trd_hvAnodeImon"))) anodeISensors = (AliTRDSensorArray*)entry->GetObject();
//      // anode hv voltages (per chamber)
//      if((entry = GetCDBentry("TRD/Calib/trd_hvAnodeUmon"))) anodeUSensors = (AliTRDSensorArray*)entry->GetObject();
//      // drift hv currents (per chamber)
//      if((entry = GetCDBentry("TRD/Calib/trd_hvDriftImon"))) driftISensors = (AliTRDSensorArray*)entry->GetObject();
//      // drift hv voltages (per chamber)
//      if((entry = GetCDBentry("TRD/Calib/trd_hvDriftUmon"))) driftUSensors = (AliTRDSensorArray*)entry->GetObject();
//    }  // end if(getHVInfo)
//
//    if(getStatusInfo) {
//      // chamber status (from sensors)
//      if((entry = GetCDBentry("TRD/Calib/trd_chamberStatus"))) chamberStatusSensors = (AliTRDSensorArray*)entry->GetObject();
//    }   // end if(getStatusInfo)
//
//    if(getGasInfo) {
//      // temperatures from chamber sensors (per chamber)
//      if((entry = GetCDBentry("TRD/Calib/trd_envTemp"))) temperatureSensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // gas overpressure (per whole TRD)
//      if((entry = GetCDBentry("TRD/Calib/trd_gasOverpressure"))) overpressureSensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // gas CO2 fraction (whole TRD)
//      if((entry = GetCDBentry("TRD/Calib/trd_gasCO2"))) gasCO2Sensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // gas H2O fraction (whole TRD)
//      if((entry = GetCDBentry("TRD/Calib/trd_gasH2O"))) gasH2OSensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // gas O2 fraction (whole TRD)
//      if((entry = GetCDBentry("TRD/Calib/trd_gasO2"))) gasO2Sensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // ADC Clk phase (whole TRD)
//  /*
//      entry = manager->Get("TRD/Calib/trd_adcClkPhase");
//      if(entry) {
//  entry->SetOwner(kTRUE);
//  adcClkPhaseSensors = (AliTRDSensorArray*)entry->GetObject();
//      }
//  */
//    }  // end if getGasInfo
//
//

  
  // get calibration information

  AliCDBEntry *entry = man->Get("TRD/Calib/ChamberGainFactor",run);
  AliTRDCalDet* g = dynamic_cast<AliTRDCalDet*>entry->GetObject();
  
  for (int i=0; i<540; i++) {
    runinfo.gain[i] = g->GetValue(i);
  }
  
  AliCDBEntry *entry = man->Get("TRD/Calib/ChamberVdrift",run);
  AliTRDCalDet* vd = dynamic_cast<AliTRDCalDet*>entry->GetObject();
  
  for (int i=0; i<540; i++) {
    runinfo.vdrift[i] = vd->GetValue(i);
  }
  
  AliCDBEntry *entry = man->Get("TRD/Calib/ChamberExB",run);
  AliTRDCalDet* exb = dynamic_cast<AliTRDCalDet*>entry->GetObject();
  
  for (int i=0; i<540; i++) {
    runinfo.ExB[i] = exb->GetValue(i);
  }
  

//    if(getCalibrationInfo) {
//      if((entry = GetCDBentry("TRD/Calib/ChamberGainFactor", 0))) chamberGainFactor = (AliTRDCalDet*)entry->GetObject();
//
//      if((entry = GetCDBentry("TRD/Calib/LocalGainFactor", 0))) padGainFactor = (AliTRDCalPad*)entry->GetObject();
//
//      ProcessTRDCalibArray(chamberGainFactor, padGainFactor,
//          parName,
//          runMeanGain, runRMSGain,
//          chamberMeanGain, chamberRMSGain,
//          smMeanGain, smRMSGain);
//    }
//
//    // process pedestals
//    AliTRDCalDet *chamberNoise = 0;
//    AliTRDCalPad *padNoise = 0;
//    Double_t runMeanNoise=0.0, runRMSNoise=0.0;
//    TVectorD chamberMeanNoise(AliTRDcalibDB::kNdet);
//    TVectorD chamberRMSNoise(AliTRDcalibDB::kNdet);
//    TVectorD smMeanNoise(AliTRDcalibDB::kNsector);
//    TVectorD smRMSNoise(AliTRDcalibDB::kNsector);
//    parName = "Noise";
//    if(getCalibrationInfo) {
//      if((entry = GetCDBentry("TRD/Calib/DetNoise", 0))) chamberNoise = (AliTRDCalDet*)entry->GetObject();
//
//      if((entry = GetCDBentry("TRD/Calib/PadNoise", 0))) padNoise = (AliTRDCalPad*)entry->GetObject();
//
//      ProcessTRDCalibArray(chamberNoise, padNoise,
//          parName,
//          runMeanNoise, runRMSNoise,
//          chamberMeanNoise, chamberRMSNoise,
//          smMeanNoise, smRMSNoise);
//    }
//
//    // process drift velocity
//    AliTRDCalDet *chamberVdrift = 0;
//    AliTRDCalPad *padVdrift = 0;
//    Double_t runMeanVdrift=0.0, runRMSVdrift=0.0;
//    TVectorD chamberMeanVdrift(AliTRDcalibDB::kNdet);
//    TVectorD chamberRMSVdrift(AliTRDcalibDB::kNdet);
//    TVectorD smMeanVdrift(AliTRDcalibDB::kNsector);
//    TVectorD smRMSVdrift(AliTRDcalibDB::kNsector);
//    parName = "Vdrift";
//    if(getCalibrationInfo) {
//      if((entry = GetCDBentry("TRD/Calib/ChamberVdrift", 0))) chamberVdrift = (AliTRDCalDet*)entry->GetObject();
//
//      if((entry = GetCDBentry("TRD/Calib/LocalVdrift", 0))) padVdrift = (AliTRDCalPad*)entry->GetObject();
//
//      ProcessTRDCalibArray(chamberVdrift, padVdrift,
//          parName,
//          runMeanVdrift, runRMSVdrift,
//          chamberMeanVdrift, chamberRMSVdrift,
//          smMeanVdrift, smRMSVdrift);
//    }
//
//    // process T0
//    AliTRDCalDet *chamberT0 = 0;
//    AliTRDCalPad *padT0 = 0;
//    Double_t runMeanT0=0.0, runRMST0=0.0;
//    TVectorD chamberMeanT0(AliTRDcalibDB::kNdet);
//    TVectorD chamberRMST0(AliTRDcalibDB::kNdet);
//    TVectorD smMeanT0(AliTRDcalibDB::kNsector);
//    TVectorD smRMST0(AliTRDcalibDB::kNsector);
//    parName = "T0";
//    if(getCalibrationInfo) {
//      if((entry = GetCDBentry("TRD/Calib/ChamberT0", 0))) chamberT0 = (AliTRDCalDet*)entry->GetObject();
//
//      if((entry = GetCDBentry("TRD/Calib/LocalT0", 0))) padT0 = (AliTRDCalPad*)entry->GetObject();
//
//      ProcessTRDCalibArray(chamberT0, padT0,
//          parName,
//          runMeanT0, runRMST0,
//          chamberMeanT0, chamberRMST0,
//          smMeanT0, smRMST0);
//    }
//
//    // process pad and chamber status
//    AliTRDCalChamberStatus* chamberStatus = 0;
//    AliTRDCalPadStatus *padStatus = 0;
//    Float_t runBadPadFraction=0.0;
//    TVectorD chamberBadPadFraction(AliTRDcalibDB::kNdet);
//    TVectorD chamberStatusValues(AliTRDcalibDB::kNdet);
//    if(getCalibrationInfo) {
//      if((entry = GetCDBentry("TRD/Calib/ChamberStatus", 0))) chamberStatus = (AliTRDCalChamberStatus*)entry->GetObject();
//
//      if((entry = GetCDBentry("TRD/Calib/PadStatus", 0))) padStatus = (AliTRDCalPadStatus*)entry->GetObject();
//
//      ProcessTRDstatus(chamberStatus, padStatus,
//            runBadPadFraction, chamberBadPadFraction,
//            chamberStatusValues);
//    }
//
//    // get Goofie information
//    AliTRDSensorArray *goofieGainSensors = 0x0;
//    AliTRDSensorArray *goofieHvSensors = 0x0;
//    AliTRDSensorArray *goofiePressureSensors = 0x0;
//    AliTRDSensorArray *goofieTempSensors = 0x0;
//    AliTRDSensorArray *goofieVelocitySensors = 0x0;
//    AliTRDSensorArray *goofieCO2Sensors = 0x0;
//    AliTRDSensorArray *goofieN2Sensors = 0x0;
//
//    if(getGoofieInfo) {
//      // goofie gain
//      if((entry = GetCDBentry("TRD/Calib/trd_goofieGain"))) goofieGainSensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // goofie HV
//      if((entry = GetCDBentry("TRD/Calib/trd_goofieHv"))) goofieHvSensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // goofie pressure
//      if((entry = GetCDBentry("TRD/Calib/trd_goofiePressure"))) goofiePressureSensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // goofie temperature
//      if((entry = GetCDBentry("TRD/Calib/trd_goofieTemp"))) goofieTempSensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // goofie drift velocity
//      if((entry = GetCDBentry("TRD/Calib/trd_goofieVelocity"))) goofieVelocitySensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // goofie CO2
//      if((entry = GetCDBentry("TRD/Calib/trd_goofieCO2"))) goofieCO2Sensors = (AliTRDSensorArray*)entry->GetObject();
//
//      // goofie N2
//      if((entry = GetCDBentry("TRD/Calib/trd_goofieN2"))) goofieN2Sensors = (AliTRDSensorArray*)entry->GetObject();
//    }   // end if getGoofieInfo
//
//    // process the DCS FEE arrays
//    Int_t nSB1 = 0; Int_t nSB2 = 0; Int_t nSB3 = 0; Int_t nSB4 = 0; Int_t nSB5 = 0;
//    Int_t nChanged = 0;
//    Bool_t sorAndEor = kFALSE;
//    TVectorD statusArraySOR(AliTRDcalibDB::kNdet);
//    TVectorD statusArrayEOR(AliTRDcalibDB::kNdet);
//    Int_t dcsFeeGlobalNTimeBins = -1;
//    Int_t dcsFeeGlobalConfigTag = -1;
//    Int_t dcsFeeGlobalSingleHitThres = -1;
//    Int_t dcsFeeGlobalThreePadClustThres = -1;
//    Int_t dcsFeeGlobalSelectiveNoSZ = -1;
//    Int_t dcsFeeGlobalTCFilterWeight = -1;
//    Int_t dcsFeeGlobalTCFilterShortDecPar = -1;
//    Int_t dcsFeeGlobalTCFilterLongDecPar = -1;
//    Int_t dcsFeeGlobalModeFastStatNoise = -1;
//    TObjString dcsFeeGlobalConfigVersion("");
//    TObjString dcsFeeGlobalConfigName("");
//    TObjString dcsFeeGlobalFilterType("");
//    TObjString dcsFeeGlobalReadoutParam("");
//    TObjString dcsFeeGlobalTestPattern("");
//    TObjString dcsFeeGlobalTrackletMode("");
//    TObjString dcsFeeGlobalTrackletDef("");
//    TObjString dcsFeeGlobalTriggerSetup("");
//    TObjString dcsFeeGlobalAddOptions("");
//    if(getDCSInfo) {
//      TObjArray *objArrayCDB = 0;
//      TObject* calDCSsor = 0x0;
//      TObject* calDCSeor = 0x0;
//      if((entry = GetCDBentry("TRD/Calib/DCS"))) objArrayCDB = (TObjArray*)entry->GetObject();
//      if(objArrayCDB) {
//        objArrayCDB->SetOwner(kTRUE);
//        calDCSsor = objArrayCDB->At(0);
//        calDCSeor = objArrayCDB->At(1);
//
//        ProcessTRDCalDCSFEE(calDCSsor, calDCSeor,
//          nSB1, nSB2, nSB3, nSB4, nSB5,
//          nChanged, sorAndEor, statusArraySOR, statusArrayEOR);
//      }
//      if(calDCSsor || calDCSeor) {
//        TObject *caldcs = 0;
//        if(calDCSsor) caldcs = calDCSsor;
//        else caldcs = calDCSeor;
//	Int_t calVer = 0;
//	if (!strcmp(caldcs->ClassName(),"AliTRDCalDCS"))   calVer = 1;
//	if (!strcmp(caldcs->ClassName(),"AliTRDCalDCSv2")) calVer = 2;
//        if (calVer == 1) {
//	  dcsFeeGlobalNTimeBins           = ((AliTRDCalDCS*)caldcs)->GetGlobalNumberOfTimeBins();
//	  dcsFeeGlobalConfigTag           = ((AliTRDCalDCS*)caldcs)->GetGlobalConfigTag();
//	  dcsFeeGlobalSingleHitThres      = ((AliTRDCalDCS*)caldcs)->GetGlobalSingleHitThres();
//	  dcsFeeGlobalThreePadClustThres  = ((AliTRDCalDCS*)caldcs)->GetGlobalThreePadClustThres();
//	  dcsFeeGlobalSelectiveNoSZ       = ((AliTRDCalDCS*)caldcs)->GetGlobalSelectiveNoZS();
//	  dcsFeeGlobalTCFilterWeight      = ((AliTRDCalDCS*)caldcs)->GetGlobalTCFilterWeight();
//	  dcsFeeGlobalTCFilterShortDecPar = ((AliTRDCalDCS*)caldcs)->GetGlobalTCFilterShortDecPar();
//	  dcsFeeGlobalTCFilterLongDecPar  = ((AliTRDCalDCS*)caldcs)->GetGlobalTCFilterLongDecPar();
//	  dcsFeeGlobalModeFastStatNoise   = ((AliTRDCalDCS*)caldcs)->GetGlobalModeFastStatNoise();
//	  dcsFeeGlobalConfigVersion       = ((AliTRDCalDCS*)caldcs)->GetGlobalConfigVersion().Data();
//	  dcsFeeGlobalConfigName          = ((AliTRDCalDCS*)caldcs)->GetGlobalConfigName().Data();
//	  dcsFeeGlobalFilterType          = ((AliTRDCalDCS*)caldcs)->GetGlobalFilterType().Data();
//	  dcsFeeGlobalReadoutParam        = ((AliTRDCalDCS*)caldcs)->GetGlobalReadoutParam().Data();
//	  dcsFeeGlobalTestPattern         = ((AliTRDCalDCS*)caldcs)->GetGlobalTestPattern().Data();
//	  dcsFeeGlobalTrackletMode        = ((AliTRDCalDCS*)caldcs)->GetGlobalTrackletMode().Data();
//	  dcsFeeGlobalTrackletDef         = ((AliTRDCalDCS*)caldcs)->GetGlobalTrackletDef().Data();
//	  dcsFeeGlobalTriggerSetup        = ((AliTRDCalDCS*)caldcs)->GetGlobalTriggerSetup().Data();
//	  dcsFeeGlobalAddOptions          = ((AliTRDCalDCS*)caldcs)->GetGlobalAddOptions().Data();
//	}
//        if (calVer == 2) {
//	  dcsFeeGlobalNTimeBins           = ((AliTRDCalDCSv2*)caldcs)->GetGlobalNumberOfTimeBins();
//	  dcsFeeGlobalConfigTag           = ((AliTRDCalDCSv2*)caldcs)->GetGlobalConfigTag();
//	  dcsFeeGlobalSingleHitThres      = ((AliTRDCalDCSv2*)caldcs)->GetGlobalSingleHitThres();
//	  dcsFeeGlobalThreePadClustThres  = ((AliTRDCalDCSv2*)caldcs)->GetGlobalThreePadClustThres();
//	  dcsFeeGlobalSelectiveNoSZ       = ((AliTRDCalDCSv2*)caldcs)->GetGlobalSelectiveNoZS();
//	  dcsFeeGlobalTCFilterWeight      = ((AliTRDCalDCSv2*)caldcs)->GetGlobalTCFilterWeight();
//	  dcsFeeGlobalTCFilterShortDecPar = ((AliTRDCalDCSv2*)caldcs)->GetGlobalTCFilterShortDecPar();
//	  dcsFeeGlobalTCFilterLongDecPar  = ((AliTRDCalDCSv2*)caldcs)->GetGlobalTCFilterLongDecPar();
//	  dcsFeeGlobalModeFastStatNoise   = ((AliTRDCalDCSv2*)caldcs)->GetGlobalModeFastStatNoise();
//	  dcsFeeGlobalConfigVersion       = ((AliTRDCalDCSv2*)caldcs)->GetGlobalConfigVersion().Data();
//	  dcsFeeGlobalConfigName          = ((AliTRDCalDCSv2*)caldcs)->GetGlobalConfigName().Data();
//	  dcsFeeGlobalFilterType          = ((AliTRDCalDCSv2*)caldcs)->GetGlobalFilterType().Data();
//	  dcsFeeGlobalReadoutParam        = ((AliTRDCalDCSv2*)caldcs)->GetGlobalReadoutParam().Data();
//	  dcsFeeGlobalTestPattern         = ((AliTRDCalDCSv2*)caldcs)->GetGlobalTestPattern().Data();
//	  dcsFeeGlobalTrackletMode        = ((AliTRDCalDCSv2*)caldcs)->GetGlobalTrackletMode().Data();
//	  dcsFeeGlobalTrackletDef        = ((AliTRDCalDCSv2*)caldcs)->GetGlobalTrackletDef().Data();
//	  dcsFeeGlobalTriggerSetup        = ((AliTRDCalDCSv2*)caldcs)->GetGlobalTriggerSetup().Data();
//	  dcsFeeGlobalAddOptions          = ((AliTRDCalDCSv2*)caldcs)->GetGlobalAddOptions().Data();
//	}
//      }
//      if(objArrayCDB) objArrayCDB->RemoveAll();
//    }   // end if(getDCSInfo)
//
//
//    // loop over time steps
//    for(UInt_t iTime = (getGRPInfo ? startTime : 0); iTime<=(getGRPInfo ? endTime : 0); iTime += (getGRPInfo ? dTime : 1)) {
//      // time stamp
//      TTimeStamp iStamp(iTime);
//      cout << "time step  " << iStamp.GetDate()/10000 << "/"
//      << (iStamp.GetDate()/100)-(iStamp.GetDate()/10000)*100 << "/"
//      << iStamp.GetDate()%100 << "   "
//      << iStamp.GetTime()/10000 << ":"
//      << (iStamp.GetTime()/100)-(iStamp.GetTime()/10000)*100 << ":"
//      << iStamp.GetTime()%100 << endl;
//
//      // cavern pressure
//      Float_t pressure = -99.;
//      Bool_t inside=kFALSE;
//      if(cavern_pressure) pressure = cavern_pressure->Eval(iStamp,inside);
//
//      // surface pressure
//      Float_t surfacePressure = -99.;
//      if(surface_pressure) surfacePressure = surface_pressure->Eval(iStamp,inside);
//
//      // anode I sensors
//      TVectorD anodeIValues(AliTRDcalibDB::kNdet);
//      if(anodeISensors) ProcessTRDSensorArray(anodeISensors, iStamp, anodeIValues);
//
//      // anode U sensors
//      TVectorD anodeUValues(AliTRDcalibDB::kNdet);
//      if(anodeUSensors) ProcessTRDSensorArray(anodeUSensors, iStamp, anodeUValues);
//
//      // drift I sensors
//      TVectorD driftIValues(AliTRDcalibDB::kNdet);
//      if(driftISensors) ProcessTRDSensorArray(driftISensors, iStamp, driftIValues);
//
//      // drift U sensors
//      TVectorD driftUValues(AliTRDcalibDB::kNdet);
//      if(driftUSensors) ProcessTRDSensorArray(driftUSensors, iStamp, driftUValues);
//
//      // chamber temperatures
//      TVectorD envTempValues(AliTRDcalibDB::kNdet);
//      if(temperatureSensors) ProcessTRDSensorArray(temperatureSensors, iStamp, envTempValues);
//
//      // chamber status sensors
//      TVectorD statusValues(AliTRDcalibDB::kNdet);
//      if(chamberStatusSensors) ProcessTRDSensorArray(chamberStatusSensors, iStamp, statusValues);
//
//      // gas overpressure
//      TVectorD overpressureValues(overpressureSensors ? overpressureSensors->NumSensors() : 0);
//      if(overpressureSensors) ProcessTRDSensorArray(overpressureSensors, iStamp, overpressureValues);
//
//      // gas CO2
//      TVectorD gasCO2Values(gasCO2Sensors ? gasCO2Sensors->NumSensors() : 0);
//      if(gasCO2Sensors) ProcessTRDSensorArray(gasCO2Sensors, iStamp, gasCO2Values);
//
//      // gas H2O
//      TVectorD gasH2OValues(gasH2OSensors ? gasH2OSensors->NumSensors() : 0);
//      if(gasH2OSensors) ProcessTRDSensorArray(gasH2OSensors, iStamp, gasH2OValues);
//
//      // gas O2
//      TVectorD gasO2Values(gasO2Sensors ? gasO2Sensors->NumSensors() : 0);
//      if(gasO2Sensors) ProcessTRDSensorArray(gasO2Sensors, iStamp, gasO2Values);
//
//      // ADC Clk phase
//      //TVectorD adcClkPhaseValues(adcClkPhaseSensors ? adcClkPhaseSensors->NumSensors() : 0);
//      //if(adcClkPhaseSensors) ProcessTRDSensorArray(adcClkPhaseSensors, iStamp, adcClkPhaseValues);
//
//      // goofie gain
//      TVectorD goofieGainValues(goofieGainSensors ? goofieGainSensors->NumSensors() : 0);
//      if(goofieGainSensors) ProcessTRDSensorArray(goofieGainSensors, iStamp, goofieGainValues);
//
//      // goofie HV
//      TVectorD goofieHvValues(goofieHvSensors ? goofieHvSensors->NumSensors() : 0);
//      if(goofieHvSensors) ProcessTRDSensorArray(goofieHvSensors, iStamp, goofieHvValues);
//
//      // goofie pressure
//      TVectorD goofiePressureValues(goofiePressureSensors ? goofiePressureSensors->NumSensors() : 0);
//      if(goofiePressureSensors) ProcessTRDSensorArray(goofiePressureSensors, iStamp, goofiePressureValues);
//
//      // goofie temperature
//      TVectorD goofieTempValues(goofieTempSensors ? goofieTempSensors->NumSensors() : 0);
//      if(goofieTempSensors) ProcessTRDSensorArray(goofieTempSensors, iStamp, goofieTempValues);
//
//      // goofie drift velocity
//      TVectorD goofieVelocityValues(goofieVelocitySensors ? goofieVelocitySensors->NumSensors() : 0);
//      if(goofieVelocitySensors) ProcessTRDSensorArray(goofieVelocitySensors, iStamp, goofieVelocityValues);
//
//      // goofie CO2
//      TVectorD goofieCO2Values(goofieCO2Sensors ? goofieCO2Sensors->NumSensors() : 0);
//      if(goofieCO2Sensors) ProcessTRDSensorArray(goofieCO2Sensors, iStamp, goofieCO2Values);
//
//      // goofie N2
//      TVectorD goofieN2Values(goofieN2Sensors ? goofieN2Sensors->NumSensors() : 0);
//      if(goofieN2Sensors) ProcessTRDSensorArray(goofieN2Sensors, iStamp, goofieN2Values);
//
//
//
//  
//

  cout << "      Start time : " << runinfo.start_time << endl;
  cout << "        End time : " << runinfo.end_time << endl;
  cout << " Cavern pressure : " << runinfo.cavern_pressure << endl;


  for (int i=0; i<10; i++) {
    cout << "chamber " << i << endl;
    cout << "      anode voltage : " << runinfo.anode_voltage[i] << endl;
    cout << "      drift voltage : " << runinfo.drift_voltage[i] << endl;
    cout << "               gain : " << runinfo.gain[i] << endl;
    cout << "     drift velocity : " << runinfo.vdrift[i] << endl;
    cout << "                ExB : " << runinfo.ExB[i] << endl;
  }
  
  
}
