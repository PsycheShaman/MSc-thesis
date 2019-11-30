void rec(Bool_t useHLT= kTRUE) {  
  AliReconstruction reco;

  reco.SetRunReconstruction("ITS TPC TRD TOF PHOS HMPID ZDC PMD T0 VZERO HLT");

//
// switch off cleanESD, write ESDfriends and Alignment data
//
  //reco.SetCleanESD(kFALSE);
  //reco.SetWriteESDfriend();
  //reco.SetWriteAlignmentData();
  //reco.SetFractionFriends(.1);

  reco.SetWriteESDfriend(kFALSE);
  reco.SetWriteAlignmentData();

//
// RAW OCDB
//
  reco.SetDefaultStorage("local:///cvmfs/alice-ocdb.cern.ch/calibration/data/2016/OCDB");
//  reco.SetCDBSnapshotMode("OCDB_MCrec.root");

//
// ITS (2 objects)                                                                                                                    
//
  //reco.SetSpecificStorage("ITS/Align/Data",          "local:///cvmfs/alice-ocdb.cern.ch/simulation/2008/v4-15-Release/Residual");
  //reco.SetSpecificStorage("ITS/Calib/SPDSparseDead", "local:///cvmfs/alice-ocdb.cern.ch/simulation/2008/v4-15-Release/Residual");
  reco.SetSpecificStorage("ITS/Align/Data",          "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
  reco.SetSpecificStorage("ITS/Calib/SPDSparseDead", "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
//
// MUON (1 object)                                                                                                                    
//
  //reco.SetSpecificStorage("MUON/Align/Data",         "alien://Folder=/alice/simulation/2008/v4-15-Release/Residual");
  reco.SetSpecificStorage("MUON/Align/Data",        "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
//
// TPC (7 objects)
//
    // reco.SetSpecificStorage("TPC/Align/Data",          "alien://Folder=/alice/simulation/2008/v4-15-Release/Residual");
    // reco.SetSpecificStorage("TPC/Calib/ClusterParam",  "alien://Folder=/alice/simulation/2008/v4-15-Release/Residual");
    // reco.SetSpecificStorage("TPC/Calib/RecoParam",     "alien://Folder=/alice/simulation/2008/v4-15-Release/Residual");
    // reco.SetSpecificStorage("TPC/Calib/TimeGain",      "alien://Folder=/alice/simulation/2008/v4-15-Release/Residual");
    // reco.SetSpecificStorage("TPC/Calib/AltroConfig",   "alien://Folder=/alice/simulation/2008/v4-15-Release/Residual");
    // reco.SetSpecificStorage("TPC/Calib/TimeDrift",     "alien://Folder=/alice/simulation/2008/v4-15-Release/Residual");
    // reco.SetSpecificStorage("TPC/Calib/Correction",    "alien://Folder=/alice/simulation/2008/v4-15-Release/Residual");

  reco.SetSpecificStorage("TPC/Align/Data",          "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
  reco.SetSpecificStorage("TPC/Calib/ClusterParam",  "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
  reco.SetSpecificStorage("TPC/Calib/RecoParam",     "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
  reco.SetSpecificStorage("TPC/Calib/TimeGain",      "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
  reco.SetSpecificStorage("TPC/Calib/AltroConfig",   "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
  reco.SetSpecificStorage("TPC/Calib/TimeDrift",     "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
  reco.SetSpecificStorage("TPC/Calib/Correction",    "local:///cvmfs/alice-ocdb.cern.ch/calibration/MC/Residual");
  reco.SetRecoParam("ZDC",AliZDCRecoParamPbPb::GetHighFluxParam(2760));

  // Introduce the TPC bug
  reco.SetOption("TPC","IntroduceBug");
  
  reco.SetRunQA(":");

  // -------------------------------------------------------
  reco.SetOption("TPC", "useHLT");
  // -------------------------------------------------------

  TStopwatch timer;
  timer.Start();

  reco.Run();

  timer.Stop();
  timer.Print();
}
