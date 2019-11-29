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

  gROOT->LoadMacro("CreateESDChain.C");
  TChain* chain = CreateESDChain("file.txt", 1, 0, kFALSE, kTRUE);

  // for includes use either global setting in $HOME/.rootrc
  // ACLiC.IncludePaths: -I$(ALICE_ROOT)/include
  // or in each macro
  gSystem->AddIncludePath("-I$ALICE_ROOT/include");

  // Create the analysis manager
  AliAnalysisManager *mgr = new AliAnalysisManager("testAnalysis");



  AliESDInputHandler* esdH = new AliESDInputHandler();
  mgr->SetInputEventHandler(esdH);

  // Enable MC event handler
  //  AliMCEventHandler* handler = new AliMCEventHandler;
  //handler->SetReadTR(kFALSE);
  //mgr->SetMCtruthEventHandler(handler);


  cout << "creating analysis tasks..." << endl;

  //AliAnalysisTask *task = new AliTRDdigitsExtract("DigitsFilter");
  //mgr->AddTask(task);

  // Create and add task
  gROOT->LoadMacro("AliTRDdigitsExtract.cxx+g");
  AliAnalysisTask *task = new AliTRDdigitsExtract("DigitsExtract");
  mgr->AddTask(task);

  cout << "connecting data containers..." << endl;

  // Create containers for input/output
  AliAnalysisDataContainer *cinput = mgr->GetCommonInputContainer();
  AliAnalysisDataContainer *cdigitqa =
    mgr->CreateContainer("cdigitqa", TList::Class(),
			 AliAnalysisManager::kOutputContainer,
			 "DigitsQA.3.root");

  // Connect input/output
  mgr->ConnectInput(task, 0, cinput);

  // No need to connect to a common AOD output container if the task does not
  // fill AOD info.
  //  mgr->ConnectOutput(task, 0, coutput0);
  mgr->ConnectOutput(task, 1, cdigitqa);


  // Enable debug printouts
  mgr->SetDebugLevel(2);

  cout << "initialize and run analyses..." << endl;

  if (!mgr->InitAnalysis()) return;
  mgr->PrintStatus();
  mgr->StartAnalysis("local", chain,1000);
}
