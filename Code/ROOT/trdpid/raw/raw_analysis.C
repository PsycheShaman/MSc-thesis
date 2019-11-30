// -*- mode: c++; c-basic-offset: 4; indent-tabs-mode: nil -*-

void raw_analysis( TString infile = "/alice/data/2018/LHC18a/000283989/raw/18000283989033.808.root",
                   TString outfile = "",
                   int nev = 1000)
{

    // ==================================================================
    // Create the raw data file reader
    //
    if (infile.Contains(".root")) {
        
        cout << "[I] Reading with ROOT" << endl;
        AliRawReaderRoot *readerDate = new AliRawReaderRoot(infile);
        readerDate->SelectEquipment(0, 1024, 1024);
        readerDate->Select("TRD");
        //readerDate->SelectEvents(7);      
        reader = (AliRawReader*)readerDate;
        
    } else if (infile.Contains(":")) {
        
        cout << "[I] Reading DATE monitoring events" << endl;
        AliRawReaderDateOnline *readerRoot = new AliRawReaderDateOnline(infile);
        readerRoot->SelectEquipment(0, 1024, 1041);
        readerRoot->Select("TRD");
        //readerRoot->SelectEvents(7);
      reader = (AliRawReader*)readerRoot;
    }
 
    // ==================================================================
    // Set up output file and histograms
    TFile of;
    if (outfile != "") {
        of.Open(outfile, "RECREATE");
    }

    TH1F *hntrkl = new TH1F("hntrkl", "number of tracklets",
                            1000, -0.5, 999.5);

    TH1F *hadc   = new TH1F("hadc", "ADC spectrum",
                            1024, -0.5, 1023.5);

    TH1F *htbsum = new TH1F("htbsum", "TBsum spectrum",
                            1000, 0.0, 30000.);
   
    TH1F *hptphase = new TH1F("hptphase", "pretrigger phase",
                              32, -0.5, 31.5.);
   
    gStyle->SetPalette(1);
    gStyle->SetOptStat(0);

    
    // ==================================================================
    // some old debug settings, maybe they will be useful...
    //AliLog::SetClassDebugLevel("AliTRDrawStreamTB", 20);
    //AliLog::SetClassDebugLevel("AliTRDrawStreamTB", 8);
    //AliLog::SetClassDebugLevel("AliTRDrawStreamTB", 5);
    
    //AliTRDrawStreamTB::SetNoDebug();
    //AliTRDrawStreamTB::SetNoErrorWarning();
    //AliTRDrawStreamTB::SetForceCleanDataOnly(); // cosmics
    //AliTRDrawStream::AllowCorruptedData(); //noise
    //AliTRDrawStreamTB::SetSkipCDH();
    //AliTRDrawStreamTB::SetExtraWordsFix();
    //AliTRDrawStream::EnableDebugStream();
    //AliTRDrawStreamTB::SetDumpHead(320);
    //AliTRDrawStreamTB::SetDumpHead(80);


    // ==================================================================
    // Set up the reader classes
    Int_t ievent = 0; //-1
    
    AliTRDdigitsManager *digMan = new AliTRDdigitsManager();
    digMan->CreateArrays();
    
    rawStream = new AliTRDrawStream(reader);

    TClonesArray trkl("AliTRDtrackletMCM");
    rawStream->SetTrackletArray(&trkl);


    // ==================================================================
    // The Event Loop

    while (reader->NextEvent()) {
        ievent++;
        
        if (ievent >= nev) break;

        //digMan->ResetArrays();
        
        if (ievent % 10 == 0) {
            cout << "Event " << ievent << endl;
        }

        // ------------------------------------------------------------
        // pretrigger phases
        //while(rawStream->NextChamber(digMan, trklContainer) >= 0) {
        while(rawStream->NextChamber(digMan) >= 0) {
            //hptphase->Fill(digMan->GetDigitsParam()->GetPretriggerPhase());
        }

        // ------------------------------------------------------------
        // number of tracklets
        hntrkl->Fill(trkl.GetEntries());

        
        // ------------------------------------------------------------
        // ADC and TBsum spectra
        for (int det=0; det<540; det++) {
            
            AliTRDSignalIndex* idx = digMan->GetIndexes(det);
            
            if (!idx) continue;
            if (!idx->HasEntry()) continue;
            
            int r,c;
            while (idx->NextRCIndex(r,c)) {

                int tbsum = 0;
                for (int t=0; t<digMan->GetDigits(det)->GetNtime(); t++) {
                    int adc = digMan->GetDigits(det)->GetData(r,c,t);
                    hadc->Fill(adc);
                    tbsum += adc;
                }
                htbsum->Fill(tbsum);
            }
        }
        
        trkl.Clear();

    } //while event

    
    // ==================================================================
    delete rawStream;
   
    if (reader)
        delete reader;
    

    // ==================================================================

    TCanvas* cadc = new TCanvas("adc", "ADC Spectrum");
    cadc->SetLogy();
    cadc->cd();
    hadc->SetXTitle("ADC");
    hadc->Draw();
        
    TCanvas* ctbsum = new TCanvas("tbsum", "TBsum Spectrum");
    ctbsum->SetLogy();
    ctbsum->cd();
    htbsum->SetXTitle("#Sigma ADC");
    htbsum->Draw();
        
    //hntrkl->Draw();
    //htbsum->Draw();
    
}
