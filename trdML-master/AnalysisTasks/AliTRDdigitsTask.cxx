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

//#include "AliKalmanTrack.h"

#include "AliTRDpadPlane.h"
#include "AliTRDtrackV1.h"
#include "AliTRDseedV1.h"

#include "AliTRDdigitsManager.h"
#include "AliTRDarrayADC.h"

#include "AliTRDdigitsTask.h"

#include <iostream>
#include <iomanip>
using namespace std;

// example of an analysis task creating a p_t spectrum
// Authors: Panos Cristakoglou, Jan Fiete Grosse-Oetringhaus, Christian Klein-Boesing
// Reviewed: A.Gheata (19/02/10)

ClassImp(AliTRDdigitsTask)

//________________________________________________________________________
AliTRDdigitsTask::AliTRDdigitsTask(const char *name)
: AliAnalysisTaskSE(name), fDigMan(0), fGeo(0), fESD(0), fOutputList(0),
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

}

//_______________________________________________________________________
Bool_t AliTRDdigitsTask::UserNotify()
{
  delete fDigitsInputFile;
  delete fDigitsOutputFile;

  AliESDInputHandler *esdH = dynamic_cast<AliESDInputHandler*> (AliAnalysisManager::GetAnalysisManager()->GetInputEventHandler());

  if ( fDigitsInputFileName != "" ) {
    TString ifname = esdH->GetInputFileName();
    ifname.ReplaceAll("AliESDs.root", fDigitsInputFileName);
    AliInfo("opening digits file " + ifname + " for reading");
    fDigitsInputFile = new TFile(ifname);
    if (!fDigitsInputFile) {
      AliWarning("digits input file '" + ifname + "' cannot be opened");
    }
  } else {
    fDigitsInputFile = NULL;
  }

  if ( fDigitsOutputFileName != "" ) {
    TString ofname = esdH->GetInputFileName();
    ofname.ReplaceAll("AliESDs.root", fDigitsOutputFileName);
    AliInfo("opening digits file " + ofname + " for writing");
    fDigitsOutputFile = new TFile(ofname);
    if (!fDigitsOutputFile) {
      AliWarning("digits output file '" + ofname + "' cannot be opened");
    }
  } else {
    fDigitsOutputFile = NULL;
  }

  fEventNoInFile = -1;

  return kTRUE;
}


//________________________________________________________________________
void AliTRDdigitsTask::UserCreateOutputObjects()
{
  // Create histograms
  // Called once
  

  fOutputList = new TList();

  fhTrackCuts = new TH1F("hTrackCuts","TrackCuts QA",10,-0.5,9.5);
  fOutputList->Add(fhTrackCuts);

  fhPtAll = new TH1F("hPtAll", "pT spectrum - all tracks", 200, 0., 20.);
  fOutputList->Add(fhPtAll);

  fhPtTRD = new TH1F("hPtTRD", "pT spectrum - TRD tracks", 200, 0., 20.);
  fOutputList->Add(fhPtTRD);

  fhTrdAdc = new TH1F("hTrdAdc", "TRD ADC spectrum", 1024, -0.5, 1023.5);
  fOutputList->Add(fhTrdAdc);

  PostData(1, fOutputList);
}



//________________________________________________________________________
void AliTRDdigitsTask::UserExec(Option_t *)
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



//  AliMCEvent* mcEvent = MCEvent();
//  if (!mcEvent) {
//    Printf("ERROR: Could not retrieve MC event");
//    return;
//  }
//  Printf("MC particles: %d", mcEvent->GetNumberOfTracks());
//
//


  // -----------------------------------------------------------------
  // Track loop to fill a pT spectrum
  for (Int_t iTracks = 0; iTracks < fESD->GetNumberOfTracks(); iTracks++) {

    fhTrackCuts->Fill(0);

    // ---------------------------------------------------------------
    // gather track information
    AliESDtrack* track = fESD->GetTrack(iTracks);
    if (!track) {
      printf("ERROR: Could not receive track %d\n", iTracks);
      continue;
    }
    fhTrackCuts->Fill(1);
    fhPtAll->Fill(track->Pt());

    AliESDfriendTrack* friendtrack = fESDfriend->GetTrack(iTracks);
    if (!friendtrack) {
      printf("ERROR: Could not receive friend track %d\n", iTracks);
      continue;
    }
    fhTrackCuts->Fill(2);


    AliTRDtrackV1* trdtrack = FindTRDtrackV1(friendtrack);
    if (!trdtrack) {
      // this happens often, because not all tracks reach the TRD
      //printf("NOTICE: Could not receive TRD track %d\n", iTracks);
      continue;
    }
    fhTrackCuts->Fill(3);
    fhPtTRD->Fill(track->Pt());


    if (abs(track->GetPID())==11) {
        cout << endl;
        for (int ly=0;ly<6;ly++) {
            Int_t det,row,col;
            if (FindTrackletPos(trdtrack, ly, &det,&row,&col) < 0) {
                // no tracklet found in this layer
                continue;
            }

            cout << "Found tracklet at "
            << det << ":" << row << ":" << col
            << " with PID: " << track->GetPID()
            << " from track " << iTracks
            << endl;

            int np = 5;
            if ( col-np < 0 || col+np >= 144 ) continue;

            for (int c = col-np; c<=col+np; c++) {
                cout << "  " << setw(3) << c << " ";
                for (int t=0; t<fDigMan->GetDigits(det)->GetNtime(); t++) {
                       cout << setw(4) << fDigMan->GetDigitAmp(row,c,t,det);
    	        }
    	        cout << endl;
            }
        }
    }



    //printf("------------------------------------\n");
    //printf("track %6d\n", iTracks);
  }


  for (int det=0; det<540; det++) {

    if (!fDigMan->GetDigits(det)) {
      AliWarning(Form("No digits found for detector %d", det));
      continue;
    }

    AliTRDpadPlane* padplane = fGeo->GetPadPlane(det);
    if (!padplane) {
      AliError(Form("AliTRDpadPlane for detector %d not found", det));
      continue;
    }

    for (int row=0; row < padplane->GetNrows(); row++) {
        for (int col=0; col < padplane->GetNcols(); col++) {
	         for (int tb=0; tb < fDigMan->GetDigits(det)->GetNtime(); tb++) {
	                fhTrdAdc->Fill(fDigMan->GetDigitAmp(row,col,tb,det));
	}
      }
    }
  }


//    if (!track->GetInnerParam()) {
//      cerr << "ERROR: no inner param" << endl;
//      continue;
//    }
//
//
//    AliVParticle* mctrack = mcEvent->GetTrack(track->GetLabel());
//    if (!mctrack) {
//      Printf("ERROR: Could not receive MC track %d", track->GetLabel());
//      continue;
//    }
//
//    const AliExternalTrackParam* trdtrack = friendtrack->GetTRDIn();
//    if (!trdtrack) {
//      Printf("WARNING: Could not receive TRD track");
//      continue;
//    }
//

  PostData(1, fOutputList);
}



//________________________________________________________________________
AliTRDtrackV1* AliTRDdigitsTask::FindTRDtrackV1(AliESDfriendTrack* friendtrack)
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

//
//
//    // ---------------------------------------------------------------
//    // display general track information
//
//    cout << "  label = " << track->GetLabel() << endl;
//
//    cout << "  eta  [rec] = " << track->GetInnerParam()->Eta() << endl;
//    cout << "  phi  [rec] = " << track->GetInnerParam()->Phi() << endl;
//    cout << "  pt   [rec] = " << track->GetInnerParam()->Pt() << endl;
//
//    cout << "  eta  [MC]  = " << mctrack->Eta() << endl;
//    cout << "  phi  [MC]  = " << mctrack->Phi() << endl;
//    cout << "  pt   [MC]  = " << mctrack->Pt() << endl;
//    cout << "  pdg  [MC]  = ";
//
//    switch (abs(mctrack->PdgCode())) {
//    case   11:   cout << "electron"; break;
//    case  211:   cout << "pion"; break;
//    case 2212:   cout << "proton"; break;
//    default:     cout << "unknown"; break;
//    }
//
//    cout << " (" << mctrack->PdgCode() << ")" << endl;
//
//
//    cout << "  pt   [TRD] = " << trdtrack->Pt() << endl;
//    cout << "  x    [TRD] = " << trdtrack->GetX() << endl;
//    cout << "  y    [TRD] = " << trdtrack->GetY() << endl;
//    cout << "  z    [TRD] = " << trdtrack->GetZ() << endl;
//
//    // cout << "  trd clusters: " << friendtrack->GetMaxTRDcluster() << endl;
//    // Int_t trkidx[6];
//    // track->GetTRDtracklets(trkidx);
//    // for (int ic = 0; ic < track->GetTRDntracklets(); ic++) {
//    //   cout << "  trkl: " << ic << ": " << trkidx[ic] << endl;
//    // }
//
//    // const AliTrackPointArray *array= friendtrack->GetTrackPointArray();
//    // if (!array) {
//    //   cout << "no track points found" << endl;
//    //   continue;
//    // }
//
//    // cout << "track " << iTracks << " has " << array->GetNPoints()
//    // 	 << " points" << endl;
//
//
//    // Int_t cls[180];
//    // track->GetTRDclusters(cls);
//    // for (int i=0;i<180;i++) {
//    //   if (cls[i] != -2) {
//    // 	cout << "  cls [" << i << "]  " << cls[i] << endl;
//    //   }
//    // }
//
//    // //cout << "  trk " << int(track->GetTRDntracklets()) << endl;
//    // Int_t trk[AliESDtrack::kTRDnPlanes];
//    // track->GetTRDtracklets(trk);
//
//    // for (int it=0; it<AliESDtrack::kTRDnPlanes; it++) {
//    //   if (trk[it] != -2) {
//    // 	cout << "  trk [" << it << "]  " << trk[it] << endl;
//    //   }
//    // }
//
//
//

Int_t AliTRDdigitsTask::FindTrackletPos(AliTRDtrackV1* trdTrack,
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

//    // AliTrackPoint tp;
//    // for (int ip=0; ip<array->GetNPoints(); ip++) {
//    //   array->GetPoint(tp,ip);
//    //   cout << "  point " << ip << ": "
//    // 	   << tp.GetVolumeID() << "  -  "
//    // 	   << tp.GetX() << "/" << tp.GetY() << "/" << tp.GetZ()
//    // 	   << "   r = " << TMath::Hypot(tp.GetX(),tp.GetY())
//    // 	   << endl;
//    // }
//
//
//
//
//
//
//    fHistPt->Fill(track->Pt());
//  } //track loop
//
//  delete geo;
//


//________________________________________________________________________
void AliTRDdigitsTask::Terminate(Option_t *)
{
  // Draw result to the screen
  // Called once at the end of the query

//  fOutputList = dynamic_cast<TList*> (GetOutputData(1));
//  if (!fOutputList) {
//    printf("ERROR: Output list not available\n");
//    return;
//  }
//
//  fHistPt = dynamic_cast<TH1F*> (fOutputList->At(0));
//  if (!fHistPt) {
//    printf("ERROR: fHistPt not available\n");
//    return;
//  }
//
  //TCanvas *c1 = new TCanvas("AliTRDdigitsTask","Pt",10,10,510,510);
  //c1->cd(1)->SetLogy();
  //fHistPt->DrawCopy("E");

  cout << "#################################### I do hope this works #####################################" << endl;
  cout << fEventNoInFile  << endl;
}


//________________________________________________________________________
void AliTRDdigitsTask::ReadDigits()
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
void AliTRDdigitsTask::WriteDigits()
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
