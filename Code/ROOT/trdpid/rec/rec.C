void rec(const char *filename="raw.root")
{
  /////////////////////////////////////////////////////////////////////////////////////////
  //
  // Reconstruction script for 2010 RAW data
  //
  /////////////////////////////////////////////////////////////////////////////////////////

  
  // Set the CDB storage location
  AliCDBManager * man = AliCDBManager::Instance();
  man->SetDefaultStorage("local:///cvmfs/alice-ocdb.cern.ch/calibration/data/2016/OCDB");

  AliReconstruction rec;

  // Set reconstruction flags (skip detectors here if neded with -<detector name>

  rec.SetRunReconstruction("ALL");

  // QA options
  //  rec.SetRunQA("Global:ESDs") ;
  //  rec.SetRunQA(":") ;
  //  rec.SetRunQA("ALL:ALL") ;
  rec.SetRunQA("Global MUON:ALL") ;

  rec.SetQARefDefaultStorage("local://$ALICE_ROOT/QAref") ;

  // AliReconstruction settings
  rec.SetWriteESDfriend(kTRUE);
  rec.SetWriteAlignmentData();
  rec.SetInput(filename);
  rec.SetUseTrackingErrorsForAlignment("ITS");

  // Upload CDB entries from the snapshot (local root file) if snapshot exist
  if (gSystem->AccessPathName("OCDB.root", kFileExists)==0) {
    rec.SetCDBSnapshotMode("OCDB.root");
  }

  // switch off cleanESD
  rec.SetCleanESD(kFALSE);

  //Ignore SetStopOnError
  rec.SetStopOnError(kFALSE);

  // Set 100% of friends
  rec.SetFractionFriends(1.0);

  // keep digits for TRD
  rec.SetOption("TRD", "cw,dc");
  
  // Delete recpoints
  rec->SetDeleteRecPoints("TPC TRD");

  AliLog::Flush();
  rec.Run();

}
