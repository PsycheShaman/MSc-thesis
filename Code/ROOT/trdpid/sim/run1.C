void run1()
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
  TChain* chain = CreateESDChain("files.txt", 1, 0, kFALSE, kTRUE);

  // for includes use either global setting in $HOME/.rootrc
  // ACLiC.IncludePaths: -I$(ALICE_ROOT)/include
  // or in each macro
  gSystem->AddIncludePath("-I$ALICE_ROOT/include");

  // Create the analysis manager
  AliAnalysisManager *mgr = new AliAnalysisManager("testAnalysis");



  AliESDInputHandler* esdH = new AliESDInputHandler();
  mgr->SetInputEventHandler(esdH);

  // Enable MC event handler
  AliMCEventHandler* handler = new AliMCEventHandler;
  handler->SetReadTR(kFALSE);
  mgr->SetMCtruthEventHandler(handler);


  
  // Create task

  gROOT->LoadMacro("AliAnalysisTaskTom.cxx+g");
  AliAnalysisTask *taskpt = new AliAnalysisTaskTom("TaskTom");

  // Add task(s)
  mgr->AddTask(taskpt);


  // Create containers for input/output
  AliAnalysisDataContainer *cinput = mgr->GetCommonInputContainer();
  AliAnalysisDataContainer *coutputpt = mgr->CreateContainer("chistpt", TList::Class(),   
      AliAnalysisManager::kOutputContainer, "Pt.ESD.1.root");

  // Connect input/output
  mgr->ConnectInput(taskpt, 0, cinput);

  // No need to connect to a common AOD output container if the task does not
  // fill AOD info.
  //  mgr->ConnectOutput(task, 0, coutput0);
  mgr->ConnectOutput(taskpt, 1, coutputpt);


  // Enable debug printouts
  mgr->SetDebugLevel(2);

  if (!mgr->InitAnalysis()) return;
  mgr->PrintStatus();
  mgr->StartAnalysis("local", chain);
}
