#include "TSystem.h"
#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TBranch.h"
#include "TEventList.h"
#include "TObject.h"
#include "TNamed.h"

#include "AliESDVZERO.h"
#include "AliESD.h"
#include "AliESDfriend.h"
#include "AliESDtrack.h"
#include "AliTracker.h"

#include "AliAnalysisTask.h"
#include "AliAnalysisManager.h"

#include "AliESDEvent.h"
#include "AliESDInputHandler.h"

#include "AliMCEventHandler.h"
#include "AliMCEvent.h"

#include "AliTRDpadPlane.h"
#include "AliTRDtrackV1.h"
#include "AliTRDseedV1.h"

#include "AliTRDdigitsManager.h"
#include "AliTRDarrayADC.h"

#include "AliESDv0KineCuts.h"

#include "AliTRDdigitsExtract.h"

#include "AliPIDResponse.h"

#include "AliCDBManager.h"
#include "AliCDBEntry.h"
#include "AliRunLoader.h"
#include "AliRun.h"
#include "AliESD.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
class AliTRDdigitsExtract;
class AliESD;
using namespace std;


ClassImp(AliTRDdigitsExtract)

//________________________________________________________________________
AliTRDdigitsExtract::AliTRDdigitsExtract(const char *name)
: AliTRDdigitsTask(name), 
 fPIDResponse(0)
{
  // Constructor

  // Define input and output slots here
  // Input slot #0 works with a TChain
  DefineInput(0, TChain::Class());
  // Output slot #0 id reserved by the base class for AOD
  // Output slot #1 writes into a TH1 container
  DefineOutput(1, TList::Class());

  // V0 PID Obj arrays
  fV0electrons = new TObjArray;
  fV0pions     = new TObjArray;
  //fV0protons   = new TObjArray;
  universalTracki = 0;

}


//________________________________________________________________________
void AliTRDdigitsExtract::UserCreateOutputObjects()
{
  fEventNoInFile = -1;
    
  ofstream ofile;
  ofile.open("pythonDict.txt", ios::app);
  if (!ofile.is_open()) {
    printf("ERROR: Could not open output file (pythonDict.txt).");
  }
  ofile << "{";
  ofile.close();

  cout << "fV0cuts: " << fV0cuts << endl;

  fV0cuts = new AliESDv0KineCuts();

  fV0electrons = new TObjArray;
  fV0pions     = new TObjArray;

  // create list for output (QA) stuff
  fOutputList = new TList();
  fOutputList->SetOwner(kTRUE);
  
  // some histograms
  TH1F* fhpte = new TH1F("fhpte","fhpte",1000,0.,20.);
  TH1F* fhptp = new TH1F("fhptp","fhptp",1000,0,20);
  TH1F* fhptTRDe = new TH1F("fhptTRDe","fhptTRDe",1000,0,20);
  TH1F* fhptTRDp = new TH1F("fhptTRDp","fhptTRDp",1000,0,20);
  TH1F* fhsigmae = new TH1F("fhsigmae","fhsigmae",480,-6,6);
  TH1F* fhsigmap = new TH1F("fhsigmap","fhsigmap",480,-6,6);
  TH2* fhdEdx = new TH2F("fhdEdx","fhdEdx",20000,0,20,20000,0,400);    
    
  // for simga electron
  TH1F* fhtmpe1 = new TH1F("fhsige1","fhsige1",500,-10,10);
  TH1F* fhtmpe2 = new TH1F("fhsige2","fhsige2",500,-10,10);
  TH1F* fhtmpe3 = new TH1F("fhsige3","fhsige3",500,-10,10);
  TH1F* fhtmpe4 = new TH1F("fhsige4","fhsige4",500,-10,10);
  TH1F* fhtmpe5 = new TH1F("fhsige5","fhsige5",500,-10,10);
  
  TH1F* fhtmpp1 = new TH1F("fhsigp1","fhsigp1",500,-8,12);
  TH1F* fhtmpp2 = new TH1F("fhsigp2","fhsigp2",50,-8,12);
  TH1F* fhtmpp3 = new TH1F("fhsigp3","fhsigp3",500,-8,12);
  TH1F* fhtmpp4 = new TH1F("fhsigp4","fhsigp4",500,-8,12);
  TH1F* fhtmpp5 = new TH1F("fhsigp5","fhsigp5",500,-8,12);
  
  // only one particle at a time 
  TH1F* fhsigmae1 = new TH1F("fhsigmae1","fhsigmae1",500,-10,10);
  TH1F* fhsigmae2 = new TH1F("fhsigmae2","fhsigmae2",500,-10,10);
  TH1F* fhsigmae3 = new TH1F("fhsigmae3","fhsigmae3",500,-10,10);
  TH1F* fhsigmae4 = new TH1F("fhsigmae4","fhsigmae4",500,-10,10);
  TH1F* fhsigmae5 = new TH1F("fhsigmae5","fhsigmae5",500,-10,10);
  
  TH1F* fhsigmap1 = new TH1F("fhsigmap1","fhsigmap1",500,-8,12);
  TH1F* fhsigmap2 = new TH1F("fhsigmap2","fhsigmap2",500,-8,12);
  TH1F* fhsigmap3 = new TH1F("fhsigmap3","fhsigmap3",500,-8,12);
  TH1F* fhsigmap4 = new TH1F("fhsigmap4","fhsigmap4",500,-8,12);
  TH1F* fhsigmap5 = new TH1F("fhsigmap5","fhsigmap5",500,-8,12);
  

  fOutputList->Add(fhtmpe1);
  fOutputList->Add(fhtmpe2);
  fOutputList->Add(fhtmpe3);
  fOutputList->Add(fhtmpe4);
  fOutputList->Add(fhtmpe5);

  fOutputList->Add(fhtmpp1);
  fOutputList->Add(fhtmpp2);
  fOutputList->Add(fhtmpp3);
  fOutputList->Add(fhtmpp4);
  fOutputList->Add(fhtmpp5);

  fOutputList->Add(fhsigmae1);
  fOutputList->Add(fhsigmae2);
  fOutputList->Add(fhsigmae3);
  fOutputList->Add(fhsigmae4);
  fOutputList->Add(fhsigmae5);

  fOutputList->Add(fhsigmap1);
  fOutputList->Add(fhsigmap2);
  fOutputList->Add(fhsigmap3);
  fOutputList->Add(fhsigmap4);
  fOutputList->Add(fhsigmap5);

  fOutputList->Add(fhpte);
  fOutputList->Add(fhptp);
  fOutputList->Add(fhptTRDe);
  fOutputList->Add(fhptTRDp);
  fOutputList->Add(fhsigmae);
  fOutputList->Add(fhsigmap);
  
  fOutputList->Add(fhdEdx); 

  AliAnalysisManager *man = AliAnalysisManager::GetAnalysisManager();
  if (man){
    AliInputEventHandler* inputHandler = (AliInputEventHandler*)(man->GetInputEventHandler());
    if (inputHandler) fPIDResponse = inputHandler->GetPIDResponse();
  }
  //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ pre post data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
  PostData(1, fOutputList);
  //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ post post data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
}



//________________________________________________________________________
void AliTRDdigitsExtract::UserExec(Option_t *)
{

  fEventNoInFile++;
  // -----------------------------------------------------------------
  // -----------------------------------------------------------------
  // IMPORTANT: call NextEvent() for book-keeping
  // -----------------------------------------------------------------
  NextEvent();
  // -----------------------------------------------------------------
  // -----------------------------------------------------------------
 
  // Skip processing of event if no digits are available
  //if ( ! ReadDigits() ) {
  //  cout << "no digits found, skipping event" << endl;
  //  return;
  //}

  // Main loop
  // Called for each event
  
  // -----------------------------------------------------------------
  // prepare event data structures
  fESD = dynamic_cast<AliESDEvent*>(InputEvent());
   //esdfriend->Print();
  cout << "this is an experiment " << dynamic_cast<AliESDEvent*>(InputEvent()) << endl;
   //else { cout << "the trees are awakening" << endl; }
  if (!fESD) {
    printf("ERROR: fESD not available\n");
    return;
  }

  printf("There are %d tracks in this event\n", fESD->GetNumberOfTracks());
  
  if (fESD->GetNumberOfTracks() <= 0) {
    // skip empty event
    return;
  }

  // make digits available
  cout << "let's see if this works out" << endl;
  //ReadDigit();
  //AliTRDdigitsTask::ReadDigits();
  ReadDigits();

  // create reference data samples
  FillV0PIDlist();

  if (!fV0tags) {
    cout << "No V0 tags found?!?" << endl;
    return;
  }
  
  AnalyseEvent();
  
  PostData(1, fOutputList);
}
 



void AliTRDdigitsExtract::AnalyseEvent()
{

  // make histograms from output list available
  
  TH1* fhpte = (TH1*)fOutputList->FindObject("fhpte");
  TH1* fhptp = (TH1*)fOutputList->FindObject("fhptp");
  TH1* fhptTRDp = (TH1*)fOutputList->FindObject("fhptTRDp");
  TH1* fhptTRDe = (TH1*)fOutputList->FindObject("fhptTRDe");
  TH1* fhsigmae = (TH1*)fOutputList->FindObject("fhsigmae");

 TH1* fhsigmap = (TH1*)fOutputList->FindObject("fhsigmap");
  TH2* fhdEdx = (TH2*)fOutputList->FindObject("fhdEdx");
  //TList* fhel = (TList*)fOutputList->FindObject 
  TH1F* fhsige1 = (TH1F*)fOutputList->FindObject("fhsige1");
  TH1F* fhsige2 = (TH1F*)fOutputList->FindObject("fhsige2");
  TH1F* fhsige3 = (TH1F*)fOutputList->FindObject("fhsige3");
  TH1F* fhsige4 = (TH1F*)fOutputList->FindObject("fhsige4");
  TH1F* fhsige5 = (TH1F*)fOutputList->FindObject("fhsige5");

  TH1F* fhsigp1 = (TH1F*)fOutputList->FindObject("fhsigp1");
  TH1F* fhsigp2 = (TH1F*)fOutputList->FindObject("fhsigp2");
  TH1F* fhsigp3 = (TH1F*)fOutputList->FindObject("fhsigp3");
  TH1F* fhsigp4 = (TH1F*)fOutputList->FindObject("fhsigp4");
  TH1F* fhsigp5 = (TH1F*)fOutputList->FindObject("fhsigp5");

  TH1F* fhsigmae1 = (TH1F*)fOutputList->FindObject("fhsigmae1");
  TH1F* fhsigmae2 = (TH1F*)fOutputList->FindObject("fhsigmae2");
  TH1F* fhsigmae3 = (TH1F*)fOutputList->FindObject("fhsigmae3");
  TH1F* fhsigmae4 = (TH1F*)fOutputList->FindObject("fhsigmae4");
  TH1F* fhsigmae5 = (TH1F*)fOutputList->FindObject("fhsigmae5");

  TH1F* fhsigmap1 = (TH1F*)fOutputList->FindObject("fhsigmap1");
  TH1F* fhsigmap2 = (TH1F*)fOutputList->FindObject("fhsigmap2");
  TH1F* fhsigmap3 = (TH1F*)fOutputList->FindObject("fhsigmap3");
  TH1F* fhsigmap4 = (TH1F*)fOutputList->FindObject("fhsigmap4");
  TH1F* fhsigmap5 = (TH1F*)fOutputList->FindObject("fhsigmap5");
  // yes there are alot of them

  Int_t pt = 0;
  // loop over tracks
  Int_t nTracks = fESD->GetNumberOfTracks();
  for(Int_t i=0; i < nTracks; i++) {
  
    AliESDtrack* track = fESD->GetTrack(i);
    fhdEdx->Fill(track->P(), track->GetTPCsignal());  

    // only analyse tagged tracks
    if (fV0tags[i] == 0) continue;
    
    pt = track->P();
      
    if ( abs(fV0tags[i]) == 11) {
      fhpte->Fill(track->P());
      pt = (Int_t)track->P();
      fhsigmae->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));  // only sigma
      // sigma on different GeV
      
      // on the electron line
      if (pt >= 1 && pt < 2){ /*fhsige1->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));*/ fhsigmae1->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      if (pt >= 2 && pt < 3){ /*fhsige2->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));*/ fhsigmae2->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      if (pt >= 3 && pt < 4){ /*fhsige3->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));*/ fhsigmae3->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      if (pt >= 4 && pt < 5){ /*fhsige4->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));*/ fhsigmae4->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      if (pt >= 5 && pt < 6){ /*fhsige5->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));*/ fhsigmae5->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      
      // on the pion line
      if (pt >= 1 && pt < 2){ fhsigp1->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      if (pt >= 2 && pt < 3){ fhsigp2->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      if (pt >= 3 && pt < 4){ fhsigp3->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      if (pt >= 4 && pt < 5){ fhsigp4->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      if (pt >= 5 && pt < 6){ fhsigp5->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
 
      //fhcompe->Fill(pt);

         
    }
    if ( abs(fV0tags[i]) == 211 ) {
      fhptp->Fill(track->P()); 
      fhsigmap->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion)); // only sigma
      // sigma on different GeV

      // on the electron line
      if (pt >= 1 && pt < 2){ fhsige1->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      if (pt >= 2 && pt < 3){ fhsige2->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      if (pt >= 3 && pt < 4){ fhsige3->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      if (pt >= 4 && pt < 5){ fhsige4->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}
      if (pt >= 5 && pt < 6){ fhsige5->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));}

      // on the pion line
      if (pt >= 1 && pt < 2){ fhsigp1->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion)); fhsigmap1->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      if (pt >= 2 && pt < 3){ fhsigp2->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion)); fhsigmap2->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      if (pt >= 3 && pt < 4){ fhsigp3->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion)); fhsigmap3->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      if (pt >= 4 && pt < 5){ fhsigp4->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion)); fhsigmap4->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      if (pt >= 5 && pt < 6){ fhsigp5->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion)); fhsigmap5->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));}
      
    }   
       
     // for particles with digits
    if (ReadDigits()) {
	cout << "has digits" << endl;
      if ( abs(fV0tags[i]) == 11 ) {
	fhptTRDe->Fill(track->P());	
        
      } 
      if ( abs(fV0tags[i]) == 211 ) {
	fhptTRDp->Fill(track->P());
      }
    }else {cout << "this goes off" << endl;}
   
  }
}

//______________________________________________________________________________
void AliTRDdigitsExtract::FillV0PIDlist(){

  cout << "entering FillV0PIDlist" << endl;
    
  // no need to run if the V0 cuts are not set
  if (!fV0cuts) {
    cout << "FillV0PIDlist: skip event - no V0 cuts" << endl;
    return;
  }
  
  // basic sanity check...
  if (!fESD) {
    cout << "FillV0PIDlist: skip event - no ESD event" << endl;
    return;
  }

  const Int_t numTracks = fESD->GetNumberOfTracks();
  
  if (numTracks < 0) {
    AliFatal("negative number of tracks?!?");
  }
   
  // Ensure there is sufficient memory for the V0 tags
  if (!fV0tags) {
    cout << "FillV0PIDlist: create fV0tags" << endl;
    fV0tags = new Int_t[numTracks];
  }else if ( sizeof(fV0tags)/sizeof(Int_t) < numTracks ) {
    cout << "FillV0PIDlist: re-create fV0tags" << endl;
    delete fV0tags;
    fV0tags = new Int_t[numTracks];
  } 
  else {
    // there is more than enough space to store the tags, no need to
    // do anything.
  }
  
  // Reset the V0 tags and reference particle arrays
  for (Int_t i = 0; i < numTracks; i++) {
    fV0tags[i] = 0;
  }

  fV0electrons->Clear();
  fV0pions->Clear();

  
  // V0 selection
  // loop over V0 particles
  fV0cuts->SetEvent(fESD);


  for(Int_t iv0=0; iv0<fESD->GetNumberOfV0s();iv0++){
    
    AliESDv0 *v0 = (AliESDv0 *) fESD->GetV0(iv0);
    
    if(!v0) continue;
    if(v0->GetOnFlyStatus()) continue;

    // Get the particle selection
    Bool_t foundV0 = kFALSE;
    Int_t pdgV0, pdgP, pdgN;
    foundV0 = fV0cuts->ProcessV0(v0, pdgV0, pdgP, pdgN);
    if(!foundV0) continue;

    Int_t iTrackP = v0->GetPindex();  // positive track inded
    Int_t iTrackN = v0->GetNindex();  // negative track

    if (fV0tags[iTrackP]) {
      printf("Warning: particle %d tagged more than once\n", iTrackP);
    }

    if (fV0tags[iTrackN]) {
      printf("Warning: particle %d tagged more than once\n", iTrackN);
    }
 
   
    // fill the Object arrays
    // positive particles
    if( pdgP == -11){
      AliESDtrack* track = fESD->GetTrack(iTrackP);
      DigitsDictionary(track, iv0, iTrackP, iv0, pdgP);
      fV0electrons->Add(fESD->GetTrack(iTrackP));
      fV0tags[iTrackP] = pdgP;
    }
    else if( pdgP == 211){
      AliESDtrack* track = fESD->GetTrack(iTrackP);
      DigitsDictionary(track, iv0, iTrackP, iv0, pdgP);
      fV0pions->Add(fESD->GetTrack(iTrackP));
      fV0tags[iTrackP] = pdgP;
    }


    // negative particles
    if( pdgN == 11){
      AliESDtrack* track = fESD->GetTrack(iTrackN);
      DigitsDictionary(track, iv0, iTrackN, -iv0, pdgN);
      fV0electrons->Add(fESD->GetTrack(iTrackN));
      fV0tags[iTrackN] = pdgN;
    }
    else if( pdgN == -211){
      AliESDtrack* track = fESD->GetTrack(iTrackN);
      DigitsDictionary(track, iv0, iTrackN, -iv0, pdgN);
      fV0pions->Add(fESD->GetTrack(iTrackN));
      fV0tags[iTrackN] = pdgN;
    }

  }
}

////________________________________________________________________________
void AliTRDdigitsExtract::DigitsDictionary(AliESDtrack* track, Int_t i, Int_t iTrack,  Int_t iV0, Int_t pdgCode) {  
    
    if (!ReadDigits()){ return; }  // this is try something out

    AliTRDtrackV1* trdtrack = NULL;
    
    // skip boring tracks
    if (trdtrack && trdtrack->GetNumberOfTracklets() == 0) return;
    
    if (track->Pt() < 1.5) return;
    // are there tracks without outer params?
    if ( ! track->GetOuterParam() ) {
      AliWarning(Form("Track %d has no OuterParam", iTrack));
      return;
    }
    
    // newest addition                                                    
    // print some info about the track
    cout << " ====== TRACK " << iTrack
         << "   pT = " << track->Pt() << " GeV";
    if (trdtrack) {
       cout << ", " << trdtrack->GetNumberOfTracklets()
            << " tracklets";
    }
    cout << " ======" << endl;
   
    ofstream ofile;
    ofile.open("pythonDict.txt", ios::app);
    if (!ofile.is_open()) {
      printf("ERROR: Could not open output file (pythonDict.txt).");
    }
 
    ofile << "\n" << universalTracki << ": {'Event': " << fEventNoInFile << ",\n\t'V0TrackID': " << iV0
    << ",\n\t'track': " << iTrack << ",\n\t'pdgCode': " << pdgCode << ",";
    universalTracki++;

    // look for tracklets in all 5 layers
    for (int ly=0;ly<6;ly++) {
      Int_t det=-1;
      Int_t row=-1;
      Int_t col=-1;
        
      Int_t det2,row2,col2;
      if ( FindDigits(track->GetOuterParam(),
     	      fESD->GetMagneticField(), ly,
              &det2,&row2,&col2) ) {
    
        if (det>=0 && det!=det2) {
           AliWarning("DET mismatch between tracklet and extrapolation: " 
                 + TString(Form("%d != %d", det, det2)));
    
          if (row>=0 && row!=row2) {
            AliWarning("ROW mismatch between tracklet and extrapolation: " 
                        + TString(Form("%d != %d", row, row2)));
          }
        } 
              		       	  		
        det = det2;
        row = row2;
        col = col2;
                    			
        cout << "    outparam: "
             << det2 << ":" << row2 << ":" << col2 << "   "
             << track->GetOuterParam()->GetX() << " / "
             << track->GetOuterParam()->GetY() << " / "
             << track->GetOuterParam()->GetZ() 
             << endl;
      }
      
      if (det<0) {
        ofile << "},";
        ofile.close();
        continue;
      }    	     	     	     	                 
      
      if (det>=0) {
        cout << "Found tracklet at "
             << det << ":" << row << ":" << col << endl;
         
        ofile << "\n\t'det" << ly << "': " << det <<","
        << "\n\t'row" << ly << "': " << row << ","
        << "\n\t'col" <<ly << "': " << col << ",";
           		     	     	     	     	                       		     	
        int np = 5;
        if ( col-np < 0 || col+np >= 144 ){
          ofile << "},";
          ofile.close();
          continue; 
        }
        ofile << "\n\t'layer" << ly << "': [";                  
        for (int c = col-np; c<=col+np;c++) {
          cout << "  " << setw(3) << c << " ";
          ofile << "[";
          for (int t=0; t<fDigMan->GetDigits(det)->GetNtime(); t++) {
            cout << setw(4) << fDigMan->GetDigitAmp(row,c,t,det);
            ofile << fDigMan->GetDigitAmp(row,c,t,det) << ", ";
          }
          ofile << "],\n\t\t\t\t";
          cout << endl;
        }
        ofile << "],";
      } 
      ofile << "},";
      ofile.close();
      cout << endl;
      
    }
}



//// ___________________________________________________________________
void AliTRDdigitsExtract::Terminate(Option_t *)   // added this
{
    // Draw result to the screen
    // Called once at the end of the query

  ofstream ofile;
  ofile.open("pythonDict.txt", ios::app);
  if (!ofile.is_open()) {
      printf("ERROR: Could not open output file (pythonDict.txt).");
  }
  ofile << "}";
  ofile.close();
}

/*
Bool_t AliTRDdigitsExtract::ReadDigit()
{

  if (!fDigMan) {
    AliError("no digits manager");
    return kFALSE;
  }

  // reset digit arrays
  for (Int_t det=0; det<540; det++) {
    fDigMan->ClearArrays(det);
    fDigMan->ClearIndexes(det);
  }

  cout <<"Is this file actually available for use "<< fDigitsInputFile << endl;
  if (!fDigitsInputFile) {
    AliError("digits file not available");
    return kFALSE;
  }
  //if (fEventNoInFile==55) {return;}
  // read digits from file
  TTree* tr = (TTree*)fDigitsInputFile->Get(Form("Event%d/TreeD",
                                                 fEventNoInFile)); // added a +
 
  cout << tr << " one of those trees      " << fDigitsInputFileName  << endl; 
  if (!fDigitsInputFile) {
    AliError(Form("digits tree for event %d not found", fEventNoInFile));
    return kFALSE; 
  }
  
  //cout << "does is not know how to read here " << fDigitsInputFile << "         " << tr << "     " << fDigMan << endl;
  if (tr == 0){ cout << "Getting out of this " << endl; return kFALSE;}
  fDigMan->ReadDigits(tr);
  //cout << "inbetween the tow steps" << endl;
  delete tr;

  //cout << "adding things to fDigMan                         hey" << endl;

  // expand digits for use in this task
  for (Int_t det=0; det<540; det++) {
    if (fDigMan->GetDigits(det)) {
      fDigMan->GetDigits(det)->Expand();
      
    }
  }
  return kTRUE; 
}*/

//*/
