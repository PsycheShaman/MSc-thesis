

void plotrow(int det, int row)
{
  TH2F *h = new TH2F("h","Pad Row Display;pad;timebin",
		     144,-0.5,143.5, 30,-0.5,29.5);
  
  TFile fd("TRD.Digits.root");
  tr = (TTree*)fd.Get("Event1/TreeD");

  AliTRDdigitsManager* digman = new AliTRDdigitsManager;
  digman->ReadDigits(tr);

  if (!digman->GetDigits(det))
    return;

  digman->GetDigits(det)->Expand();
  
  for (int c=0; c<digman->GetDigits(det)->GetNcol(); c++) {
    for (int t=0; t<digman->GetDigits(det)->GetNtime(); t++) {
      
      if (0) {
	cout << det << "/" 
	     << row << "/" 
	     << c << "/" 
	     << t << "   "
	     << endl;
      }
      
      int adc = digman->GetDigits(det)->GetData(row,c,t); 
      h->Fill(c,t,adc);
      
    }
    
  }

  //h1->Draw();
  h->Draw("colz");
}
