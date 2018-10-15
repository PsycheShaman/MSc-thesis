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

#include "AliPIDResponse.h"

#include "AliTRDdigitsExtract.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
class AliTRDdigitsExtract;

using namespace std;

// example of an analysis task writing V0 particle TRD tracks to a human and
// python readable text file.
// Authors: Dr Thomas Dietel, Chris Finlay

ClassImp(AliTRDdigitsExtract)



//________________________________________________________________________
AliTRDdigitsExtract::AliTRDdigitsExtract(const char *name)
: AliAnalysisTaskSE(name), fDigMan(0), fGeo(0), fESD(0), fV0tags(0x0), fV0cuts(0x0),
  fV0electrons(0x0), fV0pions(0x0), fV0protons(0x0), fOutputList(0), fCollisionSystem(3), 
  fHistPt(0), fHistSE(0), fHistSPr(0), fHistSPi(0), fHistdEdx(0),
  fDigitsInputFileName("TRD.Digits.root"),
  fDigitsOutputFileName("DigitsQA.3.root"),
  fDigitsInputFile(0), fDigitsOutputFile(0),
  fPIDResponse(0)
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

//AliTRDdigitsExtract::~AliTRDdigitsExtract() // TODO: DO THIS
//{
//   Destructor

//  if (fOutputList){
//    delete fOutputList;
//  }
//}

//_______________________________________________________________________
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
  cout << inputfile << endl;
  inputfile = "alien://" + inputfile;
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
 // delete fDigitsOutputFile;
 
  cout << "################### This shoud go once ########################" << endl;

  AliESDInputHandler *esdH = dynamic_cast<AliESDInputHandler*>
    (AliAnalysisManager::GetAnalysisManager()->GetInputEventHandler());

  // TODO: remember to compare this with the line above  */
  //AliAnalysisManager *man = AliAnalysisManager::GetAnalysisManager();
  //if (man){
  //  AliInputEventHandler* inputHandler = (AliInputEventHandler*)(man->GetInputEventHandler());
   // if (inputHandler) fPIDResponse = inputHandler->GetPIDResponse();
  //}


  if ( ! esdH ) return kFALSE;
  if ( ! esdH->GetTree() ) return kFALSE;
  if ( ! esdH->GetTree()->GetCurrentFile() ) return kFALSE;

  TString fname = esdH->GetTree()->GetCurrentFile()->GetName();
  
  fDigitsInputFile  = OpenDigitsFile(fname,fDigitsInputFileName,"");
  //fDigitsOutputFile = OpenDigitsFile(fname,fDigitsOutputFileName,"RECREATE");

  fEventNoInFile = -1;
  
  fEventsLost = 0;
  fNumOfProtons = 0; // 
  fNumOfPions = 0;
  fNumOfElectrons = 0;
  fNumOfPositrons = 0;
  fElectronOver3Sigmas = 0;

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

    //    
    fOutputList = new TList();
    fOutputList->SetOwner(kTRUE);
    
    // histograms
    //gStyle->SetOptStat(kFALSE);
    fHistPt = new TH1F("fHistPt", "fHistPt", 16, -2, 7);
    fHistSE = new TH1F("fHistSE","fHistSE", 24, -6, 6);
    fHistSPr = new TH1F("fHistSPr", "fHistSPr", 24, -6, 6);    
    fHistSPi = new TH1F("fHistSPi", "fHistSPi", 24, -6, 6);    
    fHistdEdx = new TH2F("fHistdEdx", "fHistdEdx", 50, 0, 1, 100, 0, 50);

    fHistdEdx->SetMarkerStyle(kFullCircle);
    fHistdEdx->SetLineColor(kRed);
    //fHistdEdx->SetHistLineStyle(0);

    fOutputList->Add(fHistPt);
    fOutputList->Add(fHistSE);
    fOutputList->Add(fHistSPi);
    fOutputList->Add(fHistSPr);
    fOutputList->Add(fHistdEdx);

    // V0 Kine cuts
    fV0cuts = new AliESDv0KineCuts();

    // V0 PID Obj arrays
    fV0electrons = new TObjArray;
    fV0pions     = new TObjArray;
    fV0protons   = new TObjArray; 	// *
   
    AliAnalysisManager *man = AliAnalysisManager::GetAnalysisManager();
    if (man){
	AliInputEventHandler* inputHandler = (AliInputEventHandler*)(man->GetInputEventHandler());
	if (inputHandler) fPIDResponse = inputHandler->GetPIDResponse();
    }    

    PostData(1, fOutputList);
}



//________________________________________________________________________
void AliTRDdigitsExtract::UserExec(Option_t *)
{
  // Main loop
  // Called for each event
  cout << fESDfriend->GetNumberOfTracks() << " newest one here" << endl;
  // update event counter for access to digits
  fEventNoInFile++;

  //AliAnalysisManager *man = AliAnalysisManager::GetAnalysisManager();
  //if (man){
//	AliInputEventHandler* inputHandler = (AliInputEventHandler*)(man->GetInputEventHandler());
// 	if (inputHandler) fPIDResponse = inputHandler->GetPIDResponse();
//  }

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
    fEventsLost++;
    return;
  }

  // make digits available
 // cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  ReadDigits();
 // cout << "???????????????????????????????????????????????????????????????????????????????????????????????????????" << endl;

  FillV0PIDlist();

  PostData(1, fOutputList);
}

//________________________________________________________________________
void AliTRDdigitsExtract::DigitsDictionary(Int_t iTrack, Int_t iV0, Int_t pdgCode) {

    ofstream ofile;
    ofile.open("pythonDict.txt", ios::app);
    if (!ofile.is_open()) {
        printf("ERROR: Could not open output file (pythonDict.txt).");
    }

    AliESDfriendTrack* friendtrack = (AliESDfriendTrack*)fESDfriend->GetTrack(iTrack);
     cout << "type " << typeid(fESDfriend).name() << fESDfriend->GetNumberOfTracks() << endl;
     if (!friendtrack) {
        printf("ERROR: Could not receive friend track %d\n", iTrack);
        return;
     }
     AliTRDtrackV1* trdtrack = FindTRDtrackV1(friendtrack);
     if (!trdtrack) {
        printf("ERROR: Could not receive TRD track %d\n", iTrack);
        return;
     }

     ofile << "\n" << universalTracki << ": {'Event': " << fEventNoInFile <<",\n\t'V0TrackID': " << iV0
     << ",\n\t'track': " << iTrack << ",\n\t'pdgCode': " << pdgCode << ",";
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
	//cout << "nwncwjnkwn    " << fDigMan << endl;
        for (int c = col-np; c<=col+np; c++) {
            ofile << "[";
	    //cout << "some thing new ######################  " << fDigMan->GetDigits(det)->GetNtime() << endl;
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
    //cout << "how many times does this run?????? " <<  (strcmp(fCalibObject->IsA()->GetName(), "AliTRDtrackV1")) << endl;
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
  ofile << "}";
  ofile.close();

  if (fOutputList){
	cout << "****************** this will cause a memory leak******************" << endl;
	//delete fOutputList;
 	fHistdEdx->SetMarkerStyle(20);
	fHistdEdx->Draw("C");
  }

  cout << " This is the number of events analyzed: " << fEventNoInFile  << endl;
  cout << " This is the number of events which had no tracks: " << fEventsLost << endl;
  cout << " Number of electrons is: " << fNumOfElectrons << endl;
  cout << " Number of pions is: " << fNumOfPions << endl;
  cout << " Number of positrons is: " << fNumOfPositrons << endl;
  cout << " NUmber of protons is: " << fNumOfProtons << endl;
  cout << " NUmber of certain electron: " << fElectronOver3Sigmas << endl;

  //if (fOutputList){
  //   delete fOutputList;
  //}
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

  cout << "run to at least over here " << tr << endl;

  fDigMan->ReadDigits(tr);
  delete tr;

  cout << "let us see what this gives us " << endl;

  // expand digits for use in this task
  for (Int_t det=0; det<540; det++) {
    if (fDigMan->GetDigits(det)) {
      fDigMan->GetDigits(det)->Expand();
     // cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! this loop runs" << endl;
    }
  }
}

//________________________________________________________________________
void AliTRDdigitsExtract::WriteDigits()
{
  cout << "~~~~~~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!~~~  This is in the write digits function, i am running  ~~~!!!!!!!!!!!!!!!!!!!!!!!!!!~~~~~~~~~~~~~~~" << endl;
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
  cout << "this is marked? " << fESDfriend->GetNumberOfTracks() << endl;


  //
  // Fill the PID object arrays holding the pointers to identified particle tracks
  //
  
 // AliAnalysisManager *man = AliAnalysisManager::GetAnalysisManager();
 // if (man){
//	AliInputEventHandler* inputHandler = (AliInputEventHandler*)(man->GetInputEventHandler());
//	if (inputHandler){
//		 fPIDResponse = inputHandler->GetPIDResponse(); 
//		cout << "############################  yes initialized #########################################" << endl;}
//	cout << "^^^^^^^^^^^^^^^^^^^^^^^ this thing is at the end of it's cycle ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl; 
// }

  // Dynamic cast to ESD events (DO NOTHING for AOD events)
  AliESDEvent *event = dynamic_cast<AliESDEvent *>(InputEvent());
  if ( !event )  return;

  //if(IsPbPb()) {
  //    fV0cuts->SetMode(AliESDv0KineCuts::kPurity,AliESDv0KineCuts::kPbPb);
  //}
  //else {
  //    fV0cuts->SetMode(AliESDv0KineCuts::kPurity,AliESDv0KineCuts::kPP);
  //}
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

    //if (std::abs(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kProton))){ fNumOfProtons++; }
    //AliAODtrack* track2 = event->GetTrack(iTrackP);
    // fill the Object arrays
    // positive particles
    Double_t p = 0;
    Double_t m = 0;
    if( pdgP == -11){
        AliESDtrack* track = event->GetTrack(iTrackP);
        DigitsDictionary(iTrackP, iv0, pdgP);
        fV0electrons->Add(track);
        fV0tags[iTrackP] = 11;
	if (std::abs(fPIDResponse->NumberOfSigmasTPC((AliVParticle*) track, AliPID::kElectron)) > 3 ) { 
           fElectronOver3Sigmas++;
	   fHistPt->Fill(track->Pt());
        }
	//fHistdEdx->Fill(track->Pt(), (track->GetTPCdEdxInfo())->GetTPCsignalLongPad());	
	m = track->GetMass();
	p = track->Pt();
	Double_t phi = track->Phi();
	Double_t theta = track->Theta();
	Double_t alpha = std::atan(std::pow( pow(std::tan(phi), 2)+ std::pow (std::tan(theta), 2), 0.5 ));
	//fHistdEdx->Fill(track->Pt(), -( (m)/(p*p) )*(std::log((p*p)/m)));
	fHistdEdx->Fill(track->Pt(), -std::cos(alpha)  );
	cout << "############## dE/dx: "<< (track->GetTPCdEdxInfo())->GetTPCsignalLongPad()  << endl;
	fNumOfElectrons++;	// 
	fNumOfPositrons++;
	fHistSE->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));
	//p rintf("effmass here %f\n",effmass);
    }
    else if( pdgP == 211){
        AliESDtrack* track = event->GetTrack(iTrackP);
        DigitsDictionary(iTrackP, iv0, pdgP);
        fV0pions->Add(track);
        fV0tags[iTrackP] = 211;
	fNumOfPions++;
	fHistSPi->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));
    }
    else if (pdgP == 2212 ){ 	// proton
	AliESDtrack* track = event->GetTrack(iTrackP);
	DigitsDictionary(iTrackP, iv0, pdgP);
	fV0protons->Add(track);
	fV0tags[iTrackP] = 2212;
	fNumOfProtons++;
    	fHistSPr->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kProton));
    }

    // negative particles
    if( pdgN == 11){
        AliESDtrack* track = event->GetTrack(iTrackN);
        DigitsDictionary(iTrackN, -iv0, pdgN);
        fV0electrons->Add(track);
        fV0tags[iTrackN] = -11;
	fNumOfElectrons++;
	//fHistdEdx->Fill(track->GetTPCdEdxInfo());
	fHistSE->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kElectron));
    }
    else if( pdgN == -211){
        AliESDtrack* track = event->GetTrack(iTrackN);
        DigitsDictionary(iTrackN, -iv0, pdgN);
        fV0pions->Add(track);
        fV0tags[iTrackN] = -211;
	fNumOfPions++;
	fHistSPi->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kPion));
    }
    else if (pdgN == -2212){ 	// antiproton
	AliESDtrack* track = event->GetTrack(iTrackN);	
	DigitsDictionary(iTrackN, -iv0, pdgN);
	fV0protons->Add(track);
	fV0tags[iTrackN] == -2212;
	fNumOfProtons++;
	fHistSPr->Fill(fPIDResponse->NumberOfSigmasTPC(track, AliPID::kProton));
    }

  }

}
