

void find_trd_signals(int ev=3)
{

  TFile fd("TRD.Digits.root");
  tr = (TTree*)fd.Get(Form("Event%d/TreeD",ev));
  tr->Print();


  AliTRDdigitsManager* digman = new AliTRDdigitsManager;
  digman->CreateArrays();
  digman->ReadDigits(tr);


  for (int det=0; det<540; det++) {
    if (!digman->GetDigits(det))
      continue;
    
    digman->GetDigits(det)->Expand();

    //cout << "Detector " << det << endl;

    // TODO: check actual number of rows, from geometry
    // here: play it same, assume 12 rows everywhere
    int nrows = 12; 

    for (int r=0; r<nrows; r++) {
      // should query nrows somewhere
      for (int c = 0; c < 144; c++) {
      
      
	int tbsum = 0;
	for (int t=0; t<digman->GetDigits(det)->GetNtime(); t++) {
	  
	  //int adc = digman->GetDigits(det)->GetDataBits(row,c,t); 
	  int adc = digman->GetDigitAmp(r,c,t,det); 
	  
	  if (adc==-7169)
	    continue;
	  
	  tbsum += adc;
	}

	if (tbsum>600) {
	  cout << det << ":" << r << ":" << c << "   "
	       << tbsum << endl;
	}
      }
    }
  }
}
