

void digits(int ev, int det, int row, int col)
{

  Int_t dcol = 5;
  

  TH2F* hsig = new TH2F("hsig", ";time bin;pad number",
			30, -0.5, 29.5,
			2*dcol+1, float(col-dcol)-0.5, float(col+dcol)+0.5);
  hsig->SetTitleOffset(1.2, "X");
  hsig->SetTitleOffset(1.8, "Y");
  hsig->SetStats(0);
  
  TFile fd("TRD.Digits.root");
  tr = (TTree*)fd.Get(Form("Event%d/TreeD",ev));
  //tr->Print();

  
  AliTRDdigitsManager* digman = new AliTRDdigitsManager;
  digman->CreateArrays();
  digman->ReadDigits(tr);

  if (!digman->GetDigits(det))
    continue;

  digman->GetDigits(det)->Expand();


  Int_t ntb = digman->GetDigits(det)->GetNtime();


  // should check range for column here...
  for (int c = col-dcol; c < col+dcol; c++) {

    cout << det << ":" << row << ":" << c << "   ";
    
    int tbsum = 0;
    for (int t=0; t<ntb; t++) {
      
      //int adc = digman->GetDigits(det)->GetDataBits(row,c,t); 
      int adc = digman->GetDigitAmp(row,c,t,det); 
      
      if (adc==-7169)
	continue;
      
      tbsum += adc;
      hsig->Fill(t,c,adc);
      cout << setw(5) << adc;
    }

    cout << "  |  sum=" << tbsum << endl;
    
  }

  TCanvas *cnv = new TCanvas( Form("trdsig_ev%d_det%03d_r%02d_c%03d",
				   ev,det,row,col),
			      Form("trdsig_ev%d_det%03d_r%02d_c%03d",
				   ev,det,row,col));

  hsig->GetXaxis()->SetRangeUser(-0.5,float(ntb)-0.5);
  hsig->GetYaxis()->SetRangeUser(col-2.5, col+2.5);
  hsig->Draw("lego2");

  
}
