void readesd(const char * fname ="AliESDs.root")
{
  TFile * file = TFile::Open(fname);
  TTree * tree = (TTree*)file->Get("esdTree");

  AliESDEvent * esd = new AliESDEvent();
  esd->ReadFromTree(tree);

  Int_t nev = tree->GetEntries();

  for (Int_t iev=0; iev<nev; iev++) {
    tree->GetEntry(iev); // Get ESD
    Int_t ntrk = esd->GetNumberOfTracks();

    for(Int_t irec=0; irec<ntrk; irec++) {
      // The signal ESD object is put here
      AliESDtrack * track = esd->GetTrack(irec);
      cout << "Pt: " << track->Pt() << endl;
    }
  }
  file->Close();
}
