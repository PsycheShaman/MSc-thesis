
void dump(TString dataset = "pPb-2016")
{
  // this should be parsed from the dataset name
  int year = 2016;

  // load the macro from AliRoot - it seems it must be compiled
  //gROOT->LoadMacro("$(ALICE_ROOT)/TRD/macros/DumpOCDBtoTree.C+");

  gROOT->ProcessLine(".L $(ALICE_ROOT)/TRD/macros/DumpOCDBtoTree.C+");
  
  // call the function to dump the tree
  DumpOCDBtoTree( dataset+".root", dataset+".txt", -1, -1,
		  Form("local:///cvmfs/alice-ocdb.cern.ch/calibration/data/%d/OCDB/",year),
		  kTRUE, // getHVInfo
		  kTRUE, // getCalibrationInfo = kFALSE,
		  kTRUE, // getGasInfo = kFALSE,
		  kTRUE, // getStatusInfo = kFALSE,
		  kTRUE, // getGoofieInfo = kFALSE,
		  kTRUE, // getDCSInfo = kFALSE,
		  kTRUE  // getGRPInfo = kTRUE));
		  );
}

