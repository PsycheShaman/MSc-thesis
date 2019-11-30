
// =======================================================================
// Test script to compare the digits on one event in two digits files.
//
// The purpose of this script is to confirm that the digits saved by
// the filtering task are identical to the original digits. This file
// only checks the digits for one event in the file - it probably
// should loop over all events in the filtered file and compare these
// events to the unfiltered ones. 
//
// (c) Dec 2016  - Tom Dietel <tom@dietel.net> 
// =======================================================================

void digdiff(TString fname1="TRD.Digits.root",
	     TString fname2="TRD.FltDigits.root",
	     Int_t evno=1699)
{

  int nchecked = 0;
  
  TFile f1(fname1);
  tr1 = (TTree*)f1.Get(Form("Event%d/TreeD",evno));

  AliTRDdigitsManager* digman1 = new AliTRDdigitsManager;
  digman1->CreateArrays();
  digman1->ReadDigits(tr1);

  TFile f2(fname2);
  tr2 = (TTree*)f2.Get(Form("Event%d/TreeD",evno));

  AliTRDdigitsManager* digman2 = new AliTRDdigitsManager;
  digman2->CreateArrays();
  digman2->ReadDigits(tr2);

  for (int det=0; det<540; det++) {

    if (!digman1->GetDigits(det))
      continue;

    digman1->GetDigits(det)->Expand();
    
    if (!digman2->GetDigits(det))
      continue;

    digman2->GetDigits(det)->Expand();
    
    //cout << digman1->GetDigits(det)->GetNrow() << endl;
    for (int r=0; r<digman1->GetDigits(det)->GetNrow(); r++) {
      for (int c=0; c<digman1->GetDigits(det)->GetNcol(); c++) {

	int tbsum = 0;
	for (int t=0; t<digman1->GetDigits(det)->GetNtime(); t++) {

	  int adc1 = digman1->GetDigitAmp(r,c,t,det); 
	  int adc2 = digman2->GetDigitAmp(r,c,t,det); 

	  if ( adc1 != adc2 ) {
	    cout << "MISMATCH: "
		 << det << "/"  << r << "/" << c  << "/" << t
		 << ": " << adc1 << " != " << adc2 << endl;
	  } else {
	    nchecked++;
	  }      
	  

	  //int adc = digman1->GetDigits(det)->GetDataBits(row,c,t); 
	  tbsum += adc1;
	  
	  
	}

	if (0 && tbsum>350 && tbsum<800) {
	  cout << det << "/" 
	       << r << "/" 
	       << c << " -> tbsum="
	       << tbsum << endl;
	}
	
      }
    }

  }

  cout << "checked " << nchecked << " ADC values" << endl;


}
