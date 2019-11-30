void sim(Int_t nev=1, Int_t run=265343) {

  gSystem->Load("liblhapdf");
  gSystem->Load("libEGPythia6");
  gSystem->Load("libpythia6");
  gSystem->Load("libAliPythia6");
  gSystem->Load("libHIJING");
  gSystem->Load("libTHijing");
  gSystem->Load("libgeant321");
  
  AliSimulation simulator;
     simulator.SetRunNumber(run);

       simulator.SetMakeSDigits("TRD TOF PHOS HMPID EMCAL MUON ZDC PMD T0 VZERO FMD");
       //  simulator.SetMakeSDigits("TRD TOF PHOS HMPID EMCAL MUON ZDC T0");
     //     simulator.SetMakeSDigits("");
     simulator.SetMakeDigitsFromHits("ITS TPC");

//
// RAW OCDB
//
  simulator.SetDefaultStorage("local:///cvmfs/alice-ocdb.cern.ch/calibration/data/2016/OCDB");

//
// ITS  (1 Total)
//     Alignment from Ideal OCDB 
  // simulator.SetSpecificStorage("ITS/Align/Data",           "local:///cvmfs/alice-ocdb.cern.ch/simulation/2008/v4-15-Release/Ideal");
  simulator.SetSpecificStorage("ITS/Align/Data",           "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Ideal");
//
// MUON  (1 Total)
//     MCH                                                                                                                    
  simulator.SetSpecificStorage("MUON/Align/Data",          "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Ideal");

//                                                                                                                                    
// TPC (7 total)
//                                                                                                                      
  simulator.SetSpecificStorage("TPC/Calib/TimeGain",       "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Ideal");
  simulator.SetSpecificStorage("TPC/Calib/ClusterParam",   "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Ideal");
  simulator.SetSpecificStorage("TPC/Calib/AltroConfig",    "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Ideal");
  simulator.SetSpecificStorage("TPC/Calib/Correction",     "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Ideal");
  simulator.SetSpecificStorage("TPC/Align/Data",           "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Ideal");
  simulator.SetSpecificStorage("TPC/Calib/TimeDrift",      "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Ideal");
  simulator.SetSpecificStorage("TPC/Calib/RecoParam",      "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual/");

//
// Vertex and Mag. field from OCDB
//
  simulator.UseVertexFromCDB();
  simulator.UseMagFieldFromGRP();

/
// simulate HLT TPC clusters, trigger, and HLT ESD
//
//  simulator.SetRunHLT("chains=TPC-compression,GLOBAL-Trigger,GLOBAL-esd-converter");

//
// The rest
//
  simulator.SetRunQA(":");

  printf("Before simulator.Run(nev);\n");
  simulator.Run(nev);
  printf("After simulator.Run(nev);\n");
}
