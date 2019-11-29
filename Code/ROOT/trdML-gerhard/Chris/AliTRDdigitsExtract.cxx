#include "TSystem.h"
#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"

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

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

// example of an analysis task writing V0 particle TRD tracks to a human and
// python readable text file.
// Authors: Dr Thomas Dietel, Chris Finlay

ClassImp(AliTRDdigitsExtract)

//________________________________________________________________________
AliTRDdigitsExtract::AliTRDdigitsExtract(const char *name)
: AliAnalysisTaskSE(name), fDigMan(0), fGeo(0), fESD(0), fV0tags(0x0), fV0cuts(0x0),
  fV0electrons(0x0), fV0pions(0x0), fOutputList(0), fCollisionSystem(3),
  fDigitsInputFileName("TRD.Digits.root"),
  fDigitsOutputFileName(""),
  fDigitsInputFile(), fDigitsOutputFile(0)
{
  // Constructor

  // Define input and output slots here
  // Input slot #0 works with a TChain
  DefineInput(0, TChain::Class());
  // Output slot #0 id reserved by the base class for AOD
  // Output slot #1 writes into a TH1 container
  DefineOutput(1, TList::Class());

  // create the digits manager
  fDigMan = new AliTRDdigitsManager;
  fDigMan->CreateArrays();

  // the geometry could be created in the constructor or similar
  fGeo = new AliTRDgeometry;
  if (! fGeo) {
    AliFatal("cannot create geometry ");
  }

  universalTracki = 0;

}

TFile* AliTRDdigitsExtract::OpenDigitsFile(TString inputfile,
					   TString digfile,
					   TString opt)
{
  // we should check if we are reading ESDs or AODs - for now, only
  // ESDs are supported

  if (digfile == "") {
    return NULL;
  }

  // construct the name of the digits file from the input file
  inputfile.ReplaceAll("AliESDs.root", digfile);

  // open the file
  AliInfo( "opening digits file " + inputfile
	   + " with option \"" + opt + "\"");
  TFile* dfile = new TFile(inputfile, opt);
  if (!dfile) {
    AliWarning("digits file '" + inputfile + "' cannot be opened");
  }

  return dfile;
}

//_______________________________________________________________________
Bool_t AliTRDdigitsExtract::UserNotify()
{
  delete fDigitsInputFile;
  delete fDigitsOutputFile;

  AliESDInputHandler *esdH = dynamic_cast<AliESDInputHandler*>
    (AliAnalysisManager::GetAnalysisManager()->GetInputEventHandler());

  if ( ! esdH ) return kFALSE;
  if ( ! esdH->GetTree() ) return kFALSE;
  if ( ! esdH->GetTree()->GetCurrentFile() ) return kFALSE;

  TString fname = esdH->GetTree()->GetCurrentFile()->GetName();

  fDigitsInputFile  = OpenDigitsFile(fname,fDigitsInputFileName,"");
  fDigitsOutputFile = OpenDigitsFile(fname,fDigitsOutputFileName,"RECREATE");

  fEventNoInFile = -1;

  return kTRUE;
}


//________________________________________________________________________
void AliTRDdigitsExtract::UserCreateOutputObjects()
{

    ofstream ofile;
    ofile.open("pythonDict.txt", ios::app);
    if (!ofile.is_open()) {
        printf("ERROR: Could not open output file (pythonDict.txt).");
    }
    ofile << "{";
    ofile.close();


    // V0 Kine cuts
    fV0cuts = new AliESDv0KineCuts();

    // V0 PID Obj arrays
    fV0electrons = new TObjArray;
    fV0pions     = new TObjArray;

  PostData(1, fOutputList);
}



//________________________________________________________________________
void AliTRDdigitsExtract::UserExec(Option_t *)
{
  // Main loop
  // Called for each event

  // update event counter for access to digits
  fEventNoInFile++;

  // -----------------------------------------------------------------
  // prepare event data structures
  fESD = dynamic_cast<AliESDEvent*>(InputEvent());
  if (!fESD) {
    printf("ERROR: fESD not available\n");
    return;
  }

  printf("There are %d tracks in this event\n", fESD->GetNumberOfTracks());

  if (fESD->GetNumberOfTracks() == 0) {
    // skip empty event
    return;
  }

  // make digits available
  ReadDigits();

  FillV0PIDlist();

  PostData(1, fOutputList);
}

//________________________________________________________________________
void AliTRDdigitsExtract::DigitsDictionary(Int_t iTrack, Int_t iV0, Int_t pdgCode, Double_t P) {

    ofstream ofile;
    ofile.open("pythonDict.txt", ios::app);
    if (!ofile.is_open()) {
        printf("ERROR: Could not open output file (pythonDict.txt).");
    }

    AliESDfriendTrack* friendtrack = fESDfriend->GetTrack(iTrack);
    if (!friendtrack) {
        printf("ERROR: Could not receive friend track %d\n", iTrack);
        return;
     }
     AliTRDtrackV1* trdtrack = FindTRDtrackV1(friendtrack);
     if (!trdtrack) {
        printf("ERROR: Could not receive TRD track %d\n", iTrack);
        return;
     }

    for (int ly=0;ly<6;ly++) {
      Int_t det,row,col;
      if (FindTrackletPos(trdtrack, ly, &det,&row,&col) < 0) {
        return;
      }
    }

    ofile << "\n" << universalTracki << ": {'Event': " << fEventNoInFile <<",\n\t'V0TrackID': " << iV0
    << ",\n\t'track': " << iTrack << ",\n\t'pdgCode': " << pdgCode << ",\n\t'Momentum': " << P << ",";
    universalTracki++;

    for (int ly=0;ly<6;ly++) {
        Int_t det,row,col;
        if (FindTrackletPos(trdtrack, ly, &det,&row,&col) < 0) {
            // no tracklet found in this layer
            ofile << "},";
            ofile.close();
            continue;
        }

        ofile << "\n\t'det" << ly << "': " << det << ","
        << "\n\t'row" << ly << "': " << row << ","
        << "\n\t'col" << ly << "': " << col << ",";

        int np = 5;
        if ( col-np < 0 || col+np >= 144 ) {
            ofile << "},";
            ofile.close();
            continue;
        }

        ofile << "\n\t'layer" << ly << "': [";

        for (int c = col-np; c<=col+np; c++) {
            ofile << "[";
            for (int t=0; t<fDigMan->GetDigits(det)->GetNtime(); t++) {
                   ofile << fDigMan->GetDigitAmp(row,c,t,det) << ", ";
            }
            ofile << "],\n\t\t\t\t";
        }
        ofile << "],";
    }
    ofile << "},";
    ofile.close();
}
//________________________________________________________________________
AliTRDtrackV1* AliTRDdigitsExtract::FindTRDtrackV1(AliESDfriendTrack* friendtrack)
{
  if (!friendtrack) {
    AliWarning("ERROR: Could not receive friend track");
    return NULL;
  }

  // find AliTRDtrackV1
  TObject* fCalibObject = 0;
  AliTRDtrackV1* trdTrack = 0;
  // find TRD track
  int icalib=0;
  while ((fCalibObject = (TObject*)(friendtrack->GetCalibObject(icalib++)))){
    if(strcmp(fCalibObject->IsA()->GetName(), "AliTRDtrackV1") != 0)
      continue;
    trdTrack = (AliTRDtrackV1 *)fCalibObject;
  }

  return trdTrack;
}

//________________________________________________________________________
Int_t AliTRDdigitsExtract::FindTrackletPos(AliTRDtrackV1* trdTrack,
					Int_t layer,
					Int_t* det, Int_t* row, Int_t* col)
{

  // loop over tracklets
  for(Int_t itr = 0; itr < 6; ++itr) {

    AliTRDseedV1* tracklet = 0;

    if(!(tracklet = trdTrack->GetTracklet(itr)))
      continue;
    if(!tracklet->IsOK())
      continue;

    if ( tracklet->GetDetector()%6 == layer ) {


      AliTRDpadPlane *padplane = fGeo->GetPadPlane(tracklet->GetDetector());

      *det = tracklet->GetDetector();
      *row = padplane->GetPadRowNumber(tracklet->GetZ());
      *col = padplane->GetPadColNumber(tracklet->GetY());

      cout << "    tracklet: " << tracklet->GetDetector()
	   << ":" << padplane->GetPadRowNumber(tracklet->GetZ())
	   << ":" << padplane->GetPadColNumber(tracklet->GetY())
	   << "   "
	   << tracklet->GetX() << " / "
	   << tracklet->GetY() << " / "
	   << tracklet->GetZ()
	   << endl;

      return 0;
    }
  }

  return -1; // no tracklet found

}

// ___________________________________________________________________
void AliTRDdigitsExtract::Terminate(Option_t *)
{
  // Draw result to the screen
  // Called once at the end of the query

  ofstream ofile;
  ofile.open("pythonDict.txt", ios::app);
  if (!ofile.is_open()) {
      printf("ERROR: Could not open output file (pythonDict.txt).");
  }
  ofile << "},";
  ofile.close();
}

//________________________________________________________________________
void AliTRDdigitsExtract::ReadDigits()
{

  if (!fDigMan) {
    AliError("no digits manager");
    return;
  }

  // reset digit arrays
  for (Int_t det=0; det<540; det++) {
    fDigMan->ClearArrays(det);
    fDigMan->ClearIndexes(det);
  }

  if (!fDigitsInputFile) {
    AliError("digits file not available");
    return;
  }

  // read digits from file
  TTree* tr = (TTree*)fDigitsInputFile->Get(Form("Event%d/TreeD",
                                                 fEventNoInFile));

  if (!fDigitsInputFile) {
    AliError(Form("digits tree for event %d not found", fEventNoInFile));
    return;
  }

  fDigMan->ReadDigits(tr);
  delete tr;

  // expand digits for use in this task
  for (Int_t det=0; det<540; det++) {
    if (fDigMan->GetDigits(det)) {
      fDigMan->GetDigits(det)->Expand();
    }
  }
}

//________________________________________________________________________
void AliTRDdigitsExtract::WriteDigits()
{
  // check for output file
  if (!fDigitsOutputFile) {
    AliError("digits output file not available");
    return;
  }

  // compress digits for storage
  for (Int_t det=0; det<540; det++) {
    fDigMan->GetDigits(det)->Expand();
  }

  // create directory to store digits tree
  TDirectory* evdir =
    fDigitsOutputFile->mkdir(Form("Event%d", fEventNoInFile),
                             Form("Event%d", fEventNoInFile));

  evdir->Write();
  evdir->cd();

  // save digits tree
  TTree* tr = new TTree("TreeD", "TreeD");
  fDigMan->MakeBranch(tr);
  fDigMan->WriteDigits();
  delete tr;
}

//______________________________________________________________________________
void AliTRDdigitsExtract::FillV0PIDlist(){

  //
  // Fill the PID object arrays holding the pointers to identified particle tracks
  //

  // Dynamic cast to ESD events (DO NOTHING for AOD events)
  AliESDEvent *event = dynamic_cast<AliESDEvent *>(InputEvent());
  if ( !event )  return;

  if(IsPbPb()) {
      fV0cuts->SetMode(AliESDv0KineCuts::kPurity,AliESDv0KineCuts::kPbPb);
  }
  else {
      fV0cuts->SetMode(AliESDv0KineCuts::kPurity,AliESDv0KineCuts::kPP);
  }
  //SetGammaCutInvMass

  // V0 selection
  // set event
  fV0cuts->SetEvent(event);

  const Int_t numTracks = event->GetNumberOfTracks();
  fV0tags = new Int_t[numTracks];
  for (Int_t i = 0; i < numTracks; i++)
    fV0tags[i] = 0;

  // loop over V0 particles
  for(Int_t iv0=0; iv0<event->GetNumberOfV0s();iv0++){


    AliESDv0 *v0 = (AliESDv0 *) event->GetV0(iv0);

    if(!v0) continue;
    if(v0->GetOnFlyStatus()) continue;

    // Get the particle selection
    Bool_t foundV0 = kFALSE;
    Int_t pdgV0, pdgP, pdgN;
    foundV0 = fV0cuts->ProcessV0(v0, pdgV0, pdgP, pdgN);
    if(!foundV0) continue;
    Int_t iTrackP = v0->GetPindex();  // positive track
    Int_t iTrackN = v0->GetNindex();  // negative track


    // fill the Object arrays
    // positive particles
    if( pdgP == -11){
        AliESDtrack* track = event->GetTrack(iTrackP);
        Double_t P = track->P();
        DigitsDictionary(iTrackP, iv0, pdgP, P);
        fV0electrons->Add(track);
        fV0tags[iTrackP] = 11;
	//	printf("effmass here %f\n",effmass);
    }
    else if( pdgP == 211){
        AliESDtrack* track = event->GetTrack(iTrackP);
        Double_t P = track->P();
        DigitsDictionary(iTrackP, iv0, pdgP, P);
        fV0pions->Add(track);
        fV0tags[iTrackP] = 211;
    }


    // negative particles
    if( pdgN == 11){
        AliESDtrack* track = event->GetTrack(iTrackN);
        Double_t P = track->P();
        DigitsDictionary(iTrackN, -iv0, pdgN, P);
        fV0electrons->Add(track);
        fV0tags[iTrackN] = -11;
    }
    else if( pdgN == -211){
        AliESDtrack* track = event->GetTrack(iTrackN);
        Double_t P = track->P();
        DigitsDictionary(iTrackN, -iv0, pdgN, P);
        fV0pions->Add(track);
        fV0tags[iTrackN] = -211;
    }

  }

}
