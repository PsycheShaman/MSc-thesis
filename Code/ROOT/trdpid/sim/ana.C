//#include "AliAnalysisAlien.h"
//#include "AliAnalysisManager.h"
//#include "AliAODInputHandler.h"
//#include "AliTRDdigitsExtract.h"
//#include "ana.h
#if !defined (__CINT__) || defined (__CLING__)
#include "AliAnalysisAlien.h"
#include "AliAnalysisManager.h"
#include "AliESDInputHandler.h"
#include "AliTRDdigitsExtract.h"
#endif

void ana()
{

  Bool_t local = kTRUE;
  Bool_t gridTest = kFALSE;

#if !defined (__CINT__) || defined (__CLING__)
  gInterpreter->ProcessLine(".include $ROOTSYS/include");
  gInterpreter->ProcessLine(".include $ALICE_ROOT/include");
  gInterpreter->ProcessLine(".include $ALICE_PHYSICS/include");
#else
  gROOT->ProcessLine(".include $ROOTSYS/include");
  gROOT->ProcessLine(".include $ALICE_ROOT/include");
  gROOT->ProcessLine(".include $ALICE_PHYSICS/include");
#endif

  gSystem->Load("libTree.so");
  gSystem->Load("libGeom.so");
  gSystem->Load("libVMC.so");
  gSystem->Load("libSTEERBase.so");
  gSystem->Load("libESD.so");
  gSystem->Load("libAOD.so");

  // load analysis framework
  gSystem->Load("libANALYSIS");
  gSystem->Load("libANALYSISalice");

#if !defined (__CINT__) || defined (__CLING__)
  gInterpreter->LoadMacro("CreateESDChain.C");
#else
  gROOT->LoadMacro("CreateESDChain.C");
#endif
//
//  TChain* chain = CreateESDChain("files.txt", 1, 0, kFALSE, kTRUE);
//
//

  // for includes use either global setting in $HOME/.rootrc
  // ACLiC.IncludePaths: -I$(ALICE_ROOT)/include
  // or in each macro
  gSystem->AddIncludePath("-I$ALICE_ROOT/include");
  gSystem->AddIncludePath("-I$ALICE_PHYSICS/include");

  // Create the analysis manager
  AliAnalysisManager *mgr = new AliAnalysisManager("testAnalysis");



  AliESDInputHandler* esdH = new AliESDInputHandler();
  //esdH->SetReadFriends(kTRUE);
  mgr->SetInputEventHandler(esdH);

  // Enable MC event handler
  //  AliMCEventHandler* handler = new AliMCEventHandler;
  //handler->SetReadTR(kFALSE);
  //mgr->SetMCtruthEventHandler(handler);


  cout << "creating analysis tasks..." << endl;

#if !defined (__CINT__) || defined (__CLING__)
  //AliAnalysisTaskMyTask *task = reinterpret_cast<AliAnalysisTaskMyTask*>(
  gInterpreter->ExecuteMacro("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C");
#else
  gROOT->LoadMacro("\$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C");
  AddTaskPIDResponse(); // *
#endif

  ///////////////////////////
  //// The TRD filter Task
  ///////////////////////////
  //AliTRDdigitsFilter *filterTask = new AliTRDdigitsFilter();
  //
  //mgr->AddTask(filterTask);
  //
  //AliAnalysisDataContainer *cinput = mgr->GetCommonInputContainer();
  //
  //if (!cinput) cinput = mgr->CreateContainer("cchain",TChain::Class(),
  //                                    AliAnalysisManager::kInputContainer);
  //
  //AliAnalysisDataContainer *coutput =mgr->CreateContainer("TRDdigitsFilter",TList::Class(), AliAnalysisManager::kOutputContainer, "DigitsFilter.root");
  //
  //
  //mgr-:wq
  //>ConnectInput(filterTask,0,cinput);
  //mgr->ConnectOutput(filterTask,1,coutput);


  // ================================================================
  // Add digits extraction task

#if !defined (__CINT__) || defined (__CLING__)
  gInterpreter->LoadMacro("AliTRDdigitsExtract.cxx+g");
  AliTRDdigitsExtract *task = reinterpret_cast<AliTRDdigitsExtract*>
    (gInterpreter->ExecuteMacro("AddExtractTask.C"));
#else
  gROOT->LoadMacro("AliTRDdigitsExtract.cxx++g");
  gROOT->LoadMacro("AddExtractTask.C");
  AddExtractTask();
#endif

  // ================================================================



  // Enable debug printouts
  mgr->SetDebugLevel(2);

  cout  << "initialize and run analyses..." << endl;

  if (!mgr->InitAnalysis()) return;
  mgr->PrintStatus();

   if (local){
     TChain* chain = new TChain("esdTree");

     //chain->Add("alien:///alice/data/2016/LHC16q/000265377/pass1_CENT_wSDD/16000265377019.102/root_archive.zip#AliESDs.root");
     chain->Add("AliESDs.root");

     // start the analysis locally, reading the events from the tchain
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
    alienHandler->SetAliPhysicsVersion("vAN-20180902-1");
    // Set the Alien API version
    alienHandler->SetAPIVersion("V1.1x");
    //
    alienHandler->SetGridDataDir("/alice/data/2016/LHC16q/");
    alienHandler->SetDataPattern("pass1_CENT_wSDD/*/*ESDs.root"); // 16000265377039.901
    //
    alienHandler->SetRunPrefix("000");

    // runnumber

    //alienHandler->AddRunNumber(265377);//stage 2 complete

   //alienHandler->AddRunNumber(265378);
    //alienHandler->AddRunNumber(265309);
    //alienHandler->AddRunNumber(265332);
//    alienHandler->AddRunNumber(265334);
    //alienHandler->AddRunNumber(265335);
    //alienHandler->AddRunNumber(265336);
    //alienHandler->AddRunNumber(265338);
    //alienHandler->AddRunNumber(265339);
    //alienHandler->AddRunNumber(265342);
    //alienHandler->AddRunNumber(265343);
   //alienHandler->AddRunNumber(265344);
    //alienHandler->AddRunNumber(265381);
   // alienHandler->AddRunNumber(265383);
    //alienHandler->AddRunNumber(265385);
    //alienHandler->AddRunNumber(265388);
    //alienHandler->AddRunNumber(265419);
    //alienHandler->AddRunNumber(265420);
    //alienHandler->AddRunNumber(265425);
    //alienHandler->AddRunNumber(265426);
    alienHandler->AddRunNumber(265499);


    // number of files per subjob
    alienHandler->SetSplitMaxInputFileNumber(20);//was 40
    TString name = "myTask";
    //TString tmp; tmp.Form("%d",n);
    //name.Append(tmp.Data());
    name.Append(".sh");
    alienHandler->SetExecutable(name);
    //
    alienHandler->SetTTL(20000);//was 1000
    alienHandler->SetJDLName("myTask.jdl");

    alienHandler->SetOutputToRunNo(kTRUE);
    alienHandler->SetKeepLogs(kTRUE);
    //
    //
    alienHandler->SetDefaultOutputs(kFALSE);
    //TString file = "DigitsExtractQA";
    //TString outputfiles = file.Append(tmp.Data());
    alienHandler->SetOutputFiles("DigitsExtractQA.root");
    //TString archive = "log_archive.zip:stdout,stderr,pythonDict.txt root_archive.zip:";
    alienHandler->SetOutputArchive("log_archive.zip:stdout,stderr,pythonDict.txt root_archive.zip:DigitsExtractQA.root");
    //
    //
    alienHandler->SetMaxMergeStages(1);
    alienHandler->SetMergeViaJDL(kTRUE);
    //alienHandler->SetMergeViaJDL(kFALSE);

//*****************************************8
//  alienHandler->SetMergeViaJDL(kFALSE):
    //
    //
    alienHandler->SetGridWorkingDir("wd");
    alienHandler->SetGridOutputDir("od");

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
  //
  alienHandler->SetRunMode("full");
	//
  //alienHandler->SetRunMode("terminate");
	mgr->StartAnalysis("grid");

    }

  }

}
