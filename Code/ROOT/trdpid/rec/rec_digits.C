void rec_digits(const char *filename="/alice/data/2015/LHC15o/000244918/raw/15000244918019.100.root", Int_t startEvent = 0, Int_t nEvents = -1)
{

  Int_t run = ExtractRunNumber(filename);
  printf("run number: %d\n", run);

  // Set the CDB storage location
  AliCDBManager * man = AliCDBManager::Instance();
  //man->SetDefaultStorage("local://$ALICE_ROOT/OCDB");
  //man->SetDefaultStorage(Form("alien://folder=/alice/data/2010/OCDB?cacheFold=/lustre/alice/local/cache/%s", gSystem->GetUserInfo()->fUser->Data()));
  //man->SetDefaultStorage("local:///lustre/alice/alien/alice/data/2010/OCDB");
  man->SetDefaultStorage("alien://folder=/alice/data/2013/OCDB");
  man->SetCacheFlag(kTRUE);


  
  //man->SetDefaultStorage("raw://");
  man->SetRun(run);

  // Reconstruction settings
  AliReconstruction rec;
  if(nEvents > -1){
    rec.SetEventRange(startEvent, startEvent + nEvents);
  }

//   AliExternalTrackParam::SetMostProbablePt(8.);

  // QA options
  //rec.SetRunQA("ITS TPC TOF PHOS HMPID EMCAL MUON FMD ZDC PMD T0 VZERO ACORDE HLT:ALL:TRD:") ;
  //rec.SetRunQA("");
//   rec.SetRunReconstruction("ITS TPC TRD TOF");

  rec.SetRunReconstruction("");
  rec.SetRunLocalReconstruction("");
  rec.SetRunTracking("");
  rec.SetFillESD("");
  rec.SetRunQA(":");

  rec.SetLoadAlignFromCDB(kFALSE);
  rec.SetLoadAlignData("");
  
  rec.SetRunCascadeFinder(kFALSE);
  rec.SetRunV0Finder(kFALSE);
  rec.SetRunVertexFinder(kFALSE);
  rec.SetRunVertexFinderTracks(kFALSE);
  rec.SetRunMultFinder(kFALSE);
  //rec.SetRunHLTTracking(kFALSE);
  //rec.SetRunReconstruction("ALL");
  rec.SetQARefDefaultStorage("local://$ALICE_ROOT/QAref") ;

  // AliReconstruction settings
//   rec.SetWriteESDfriend(kTRUE);

//   rec.SetWriteAlignmentData();
  rec.SetInput(filename);
//   rec.SetUseTrackingErrorsForAlignment("ITS");

//   AliTPCReconstructor::SetStreamLevel(1);

  rec.SetOption("TRD", "dc");

  // switch off cleanESD
  rec.SetCleanESD(kFALSE);

  AliLog::Flush();
  rec.Run();

  // make sure job does not stay in the queue
  printf("Kill ourself\n");
  gSystem->Exec(Form("kill -9 %d",gSystem->GetPid()));
}

Int_t ExtractRunNumber(TString fString){
  TObjArray *ptoks = (TObjArray *)fString.Tokenize("?");
  TString path = ((TObjString *)ptoks->UncheckedAt(0))->String();
  TObjArray *toks = (TObjArray *)path.Tokenize("/");
  TString fname = ((TObjString *)(toks->UncheckedAt(toks->GetEntriesFast() - 1)))->String();
  TString rstr = fname(2,9);
  return rstr.Atoi();
}
