void filter()
{
  // for includes use either global setting in $HOME/.rootrc
  // ACLiC.IncludePaths: -I$(ALICE_ROOT)/include
  // or in each macro
  gSystem->AddIncludePath("-I$ALICE_ROOT/include");

  gSystem->Load("libTree.so");
  gSystem->Load("libGeom.so");
  gSystem->Load("libVMC.so");
  gSystem->Load("libSTEERBase.so");
  gSystem->Load("libESD.so");
  gSystem->Load("libAOD.so"); 

  // load analysis framework
  gSystem->Load("libANALYSIS");
  gSystem->Load("libANALYSISalice");

  
  gSystem->Load("libPWGPP.so"); 

  
  gROOT->LoadMacro("CreateESDChain.C");
  TChain* chain = CreateESDChain("files.txt", 1, 0, kFALSE, kTRUE);


  // Create the analysis manager
  AliAnalysisManager *mgr = new AliAnalysisManager("testAnalysis");



  AliESDInputHandler* esdH = new AliESDInputHandler();
  mgr->SetInputEventHandler(esdH);

  // Enable MC event handler
  AliMCEventHandler* handler = new AliMCEventHandler;
  handler->SetReadTR(kFALSE);
  mgr->SetMCtruthEventHandler(handler);

  
  gROOT->LoadMacro(Form("%s/PWGPP/TRD/macros/AddTRDdigitsFilter.C",
			gSystem->Getenv("ALICE_PHYSICS")));

  AddTRDdigitsFilter(0);
  
  
//  // Create digits filter (df) task
//
//  gROOT->LoadMacro("AliTRDdigitsFilter.cxx+g");
//  AliAnalysisTask *taskdf = new AliTRDdigitsFilter("DigitsFilter");
//
//  // Add task(s)
//  mgr->AddTask(taskdf);
//
//  // Create containers for input/output
//  AliAnalysisDataContainer *cinput = mgr->GetCommonInputContainer();
//  AliAnalysisDataContainer *coutputdf =
//    mgr->CreateContainer("cdigflt", TList::Class(),   
//			 AliAnalysisManager::kOutputContainer,
//			 "DigitsFilter.root");
//
//  // Connect input/output
//  mgr->ConnectInput(taskdf, 0, cinput);
//
//  // No need to connect to a common AOD output container if the task does not
//  // fill AOD info.
//  //  mgr->ConnectOutput(task, 0, coutput0);
//  mgr->ConnectOutput(taskdf, 1, coutputdf);
//  cout << "done" << endl;
//  
//  // Enable debug printouts
//  mgr->SetDebugLevel(2);
//
  
  if (!mgr->InitAnalysis()) return;
  mgr->PrintStatus();
  mgr->StartAnalysis("local", chain);
}
