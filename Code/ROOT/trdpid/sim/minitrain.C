
void minitrain()
{
  
  // D. Create the train and set-up the handlers
  //=============================================
  AliAnalysisManager *mgr  =
    new AliAnalysisManager("Analysis Train",
			   "A test setup for the analysis train");

  // ESD input handler
  AliESDInputHandler *esdHandler = new AliESDInputHandler();
  mgr->SetInputEventHandler(esdHandler);       

  
  // Debugging if requested
  mgr->SetDebugLevel(3);
        

  // F. Run the analysis
  //=====================
  if (mgr->InitAnalysis()) {
    mgr->PrintStatus();
    mgr->StartAnalysis(analysisMode, chain);
  }   
}
