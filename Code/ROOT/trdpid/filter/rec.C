void rec(const char *filename="raw.root")
{
  /////////////////////////////////////////////////////////////////////////////////////////
  //
  // Reconstruction script for 2010 RAW data
  //
  /////////////////////////////////////////////////////////////////////////////////////////

  
  // Set the CDB storage location
  AliCDBManager * man = AliCDBManager::Instance();
  man->SetDefaultStorage("alien://folder=/alice/data/2018/OCDB");

  AliLog::SetClassDebugLevel("AliTRDReconstructor", 1);
  
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
  const double kZOutSectorCut = 3.; // cut on clusters on wrong side of CE (added to extendedRoadZ)
  AliTPCReconstructor::SetZOutSectorCut(kZOutSectorCut);

  // Upload CDB entries from the snapshot (local root file) if snapshot exist
  if (gSystem->AccessPathName("OCDB.root", kFileExists)==0) {
    rec.SetCDBSnapshotMode("OCDB.root");
  }

  // Specific AD storage, see https://alice.its.cern.ch/jira/browse/ALIROOT-6056
  //  rec.SetSpecificStorage("AD/Calib/TimeSlewing", "alien://Folder=/alice/simulation/2008/v4-15-Release/Ideal");

  // switch off cleanESD
  rec.SetCleanESD(kFALSE);

  //Ignore SetStopOnError
  rec.SetStopOnError(kFALSE);

  // keep digits for TRD
  rec.SetOption("TRD", "cw,dc");

  
  // Delete recpoints
  rec.SetDeleteRecPoints("TPC TRD");

  // Set 100% of friends
  // rec.SetFractionFriends(2.0);

  AliLog::Flush();
  rec.Run();

}
