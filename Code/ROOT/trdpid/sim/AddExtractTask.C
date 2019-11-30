
#if !defined (__CINT__) || defined (__CLING__)
#include "AliAnalysisManager.h"
#include "AliESDv0KineCuts.h"
#include "AliTRDdigitsExtract.h"
#include <TString.h>
#include <TList.h>
#endif



AliTRDdigitsExtract* AddExtractTask(TString name = "name")
{
    // get the manager via the static access member. since it's static, you don't need
    // an instance of the class to call the function
    
    AliAnalysisManager *mgr = AliAnalysisManager::GetAnalysisManager();
    if (!mgr) {
        return 0x0;
    }
    // get the input event handler, again via a static method. 
    // this handler is part of the managing system and feeds events
    // to your task
    if (!mgr->GetInputEventHandler()) {
        return 0x0;
    }
    // by default, a file is open for writing. here, we get the filename
    //TString fileName = AliAnalysisManager::GetCommonFileName();
    //fileName += ":MyTask";      // create a subfolder in the file


    // now we create an instance of your task
    AliTRDdigitsExtract* task = new AliTRDdigitsExtract(name.Data());   
    if(!task) return 0x0;
    
    task->SetDigitsInputFilename("TRD.FltDigits.root");
  
    AliESDv0KineCuts* v0cuts = new AliESDv0KineCuts();
    v0cuts->SetMode(AliESDv0KineCuts::kPurity,AliESDv0KineCuts::kPP);
    task->SetV0KineCuts(v0cuts);
    
    mgr->AddTask(task);
    
    //AliAnalysisDataContainer *cinput = 
    //       mgr->CreateContainer("cchain", TChain::Class(), AliAnalysisManager::kInputContainer); 

    //// Create containers for input/output
    //AliAnalysisDataContainer *cdigitqa =
    //  mgr->CreateContainer("cdigitqa", TList::Class(),
    //			 AliAnalysisManager::kOutputContainer,
    //			 "DigitsQA.3.root"); //cdigitqa,  DigitsQA.3.root

    // Connect common input container
    //mgr->ConnectInput(task, 0, cinput);
    mgr->ConnectInput(task, 0, mgr->GetCommonInputContainer());
    
    // No need to connect to a common AOD output container if the task does not
    // fill AOD info.
    //  mgr->ConnectOutput(task, 0, coutput0);
    
    // Connect output for 
//    mgr->ConnectOutput ( task, 1,
//			 mgr->CreateContainer("cdigitqa", TList::Class(),
//					      AliAnalysisManager::kOutputContainer,
//					      "DigitsExtractQA.root"));

    // returns a pointer to your task. this will be convenient later on
    // when you will run your analysis in an analysis train on grid
    return task;
}

