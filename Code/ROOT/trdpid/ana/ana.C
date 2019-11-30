void ana()
{
  gSystem->Load("libTree.so");
  gSystem->Load("libGeom.so");
  gSystem->Load("libVMC.so");
  gSystem->Load("libSTEERBase.so");
  gSystem->Load("libESD.so");
  gSystem->Load("libAOD.so"); 

  // load analysis framework
  gSystem->Load("libANALYSIS");
  gSystem->Load("libANALYSISalice");


  // ---------------------------------------------------------------
  // create the TChain to loop over

  TString basedir = "/alice/data/2016/LHC16q/000265377/pass1_CENT_wSDD/";
  
  TChain* chain = new TChain("esdTree");
  chain->Add(basedir+"16000265377019.100/root_archive.zip#AliESDs.root");
  chain->Add(basedir+"16000265377019.101/root_archive.zip#AliESDs.root");
  chain->Add(basedir+"16000265377019.103/root_archive.zip#AliESDs.root");
  
  // // old version with external macro, not very portable
  // gROOT->LoadMacro("CreateESDChain.C");
  // CreateESDChain("files.txt", 1, 0, kFALSE, kTRUE);




  // for includes use either global setting in $HOME/.rootrc
  // ACLiC.IncludePaths: -I$(ALICE_ROOT)/include
  // or in each macro
  gSystem->AddIncludePath("-I$ALICE_ROOT/include");

  // ---------------------------------------------------------------
  // Create the analysis manager
  AliAnalysisManager *mgr = new AliAnalysisManager("testAnalysis");


  AliESDInputHandler* esdH = new AliESDInputHandler();
  mgr->SetInputEventHandler(esdH);

  // Enable MC event handler
  //  AliMCEventHandler* handler = new AliMCEventHandler;
  //handler->SetReadTR(kFALSE);
  //mgr->SetMCtruthEventHandler(handler);


  // ---------------------------------------------------------------
  // ---------------------------------------------------------------
  // set up analysis tasks
  // ---------------------------------------------------------------
  // ---------------------------------------------------------------
  // This is where your code goes
  // ---------------------------------------------------------------
  // ---------------------------------------------------------------

  cout << "creating analysis tasks..." << endl;

  //AliAnalysisTask *task = new AliTRDdigitsFilter("DigitsFilter");
  //mgr->AddTask(task);
  
  // Create and add task
  AliTRDdigitsTask *task = new AliTRDdigitsTask("DigitsTask");
  task->SetDigitsInputFilename("TRD.FltDigits.root");
  mgr->AddTask(task);

  cout << "connecting data containers..." << endl;
  
  // Create containers for input/output
  AliAnalysisDataContainer *cinput = mgr->GetCommonInputContainer();
  AliAnalysisDataContainer *cdigitqa =
    mgr->CreateContainer("cdigitqa", TList::Class(),   
			 AliAnalysisManager::kOutputContainer,
			 "DigitsQA.1.root");

  // Connect input/output
  mgr->ConnectInput(task, 0, cinput);

  // No need to connect to a common AOD output container if the task does not
  // fill AOD info.
  //  mgr->ConnectOutput(task, 0, coutput0);
  mgr->ConnectOutput(task, 1, cdigitqa);

  
  // ---------------------------------------------------------------
  // final tweaking and run the analysis

  // Enable debug printouts
  mgr->SetDebugLevel(2);

  cout << "initialize and run analyses..." << endl;
  
  if (!mgr->InitAnalysis()) return;
  mgr->PrintStatus();
  mgr->StartAnalysis("local", chain,1000);
}
