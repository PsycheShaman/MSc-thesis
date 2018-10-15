
//#include "AliAnalysisAlien.h"
//#include "AliAnalysisManager.h"
//#include "AliAODInputHandler.h"
//#include "AliTRDdigitsExtract.h"
//#include "ana.h
#if !defined (__CINT__) || defined (__CLING__)
#include "AliAnalysisAlien.h"
#include "AliAnalysisManager.h"
#include "AliESDInputHandler.h"
#endif

void ana()
{

  Bool_t local = kTRUE;
  Bool_t gridTest = kTRUE;

  gROOT->ProcessLine(".include $ROOTSYS/include");
  gROOT->ProcessLine(".include $ALICE_ROOT/include");

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
  //  AliMCEventHandler* handler = new AliMCEventHandler;
  //handler->SetReadTR(kFALSE);
  //mgr->SetMCtruthEventHandler(handler);


  cout << "creating analysis tasks..." << endl;

  gROOT->LoadMacro("\$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C");
  AddTaskPIDResponse(); // *

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
			 "DigitsQA.3.root"); //cdigitqa,  DigitsQA.3.root

  // Connect input/output
  mgr->ConnectInput(task, 0, cinput);

  // No need to connect to a common AOD output container if the task does not
  // fill AOD info.
  //  mgr->ConnectOutput(task, 0, coutput0);
  mgr->ConnectOutput(task, 1, cdigitqa);


  // Enable debug printouts
  mgr->SetDebugLevel(2);

  cout  << "initialize and run analyses..." << endl;

  if (!mgr->InitAnalysis()) return;
  mgr->PrintStatus();
  
   if (local){
    mgr->StartAnalysis("local", chain,100);
  
  }else{
    // 
    AliAnalysisAlien *alienHandler = new AliAnalysisAlien();
    // 
    alienHandler->AddIncludePath("-I. -I$ROOTSYS/include -I$ALICE_ROOT -I$ALICE_ROOT/include -I$ALICE_PHYSICS/include");
    //
    alienHandler->SetAdditionalLibs("AliTRDdigitsExtract.cxx AliTRDdigitsExtract.h");
    alienHandler->SetAnalysisSource("AliTRDdigitsExtract.cxx");
    //
    //
    alienHandler->SetAliPhysicsVersion("vAN-20180910-1");
    // Set the Alien API version
    alienHandler->SetAPIVersion("V1.1x");
    //
    alienHandler->SetGridDataDir("/alice/data/2017/LHC17l"); //  /2017/LHC17l
    alienHandler->SetDataPattern("pass1/*/*ESDs.root");  //  pass1/*/*ESDs.root   /pass1_CENT_wSDD/16000265377039.901
    //
    alienHandler->SetRunPrefix("000");
    // runnumber
    alienHandler->AddRunNumber(277312);
    // number of files per subjob
    alienHandler->SetSplitMaxInputFileNumber(40);
    alienHandler->SetExecutable("myTask.sh");
    //
    alienHandler->SetTTL(10000);
    alienHandler->SetJDLName("myTask.jdl");
    
    alienHandler->SetOutputToRunNo(kTRUE);
    alienHandler->SetKeepLogs(kTRUE);
    //
    //
    alienHandler->SetDefaultOutputs(kFALSE);
   
    alienHandler->SetOutputFiles("DigitsQA.3.root");
    //TString archive = "log_archive.zip:stdout,stderr,pythonDict.txt root_archive.zip:";
    alienHandler->SetOutputArchive("log_archive.zip:stdout,stderr,pythonDict.txt root_archive.zip:DigitsQA.3.root");

    //
    //
    alienHandler->SetMaxMergeStages(1);
    alienHandler->SetMergeViaJDL(kTRUE);
    
    //
    alienHandler->SetGridWorkingDir("myWorkingDir");
    alienHandler->SetGridOutputDir("myOutputDir");
    
    //
    mgr->SetGridHandler(alienHandler);
    if (gridTest){
	//
	alienHandler->SetNtestFiles(1);
	//
	alienHandler->SetRunMode("test");
	mgr->StartAnalysis("grid");
    }else{
	//
	alienHandler->SetRunMode("full");
	mgr->StartAnalysis("grid", chain);
    }
    
  }    
        
       
}      
             
         


