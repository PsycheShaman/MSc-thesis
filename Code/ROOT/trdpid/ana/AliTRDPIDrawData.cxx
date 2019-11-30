/**************************************************************************
* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
*                                                                        *
* Author:             *
* Contributors are mentioned in the code where appropriate.              *
*                                                                        *
* Permission to use, copy, modify and distribute this software and its   *
* documentation strictly for non-commercial purposes is hereby granted   *
* without fee, provided that the above copyright notice appears in all   *
* copies and that both the copyright notice and this permission notice   *
* appear in the supporting documentation. The authors make no claims     *
* about the suitability of this software for any purpose. It is          *
* provided "as is" without express or implied warranty.                  *
**************************************************************************/
//
// The task:
// write out raw
//
//
// Author:
//
//


#include "AliTRDPIDrawData.h"
#include "AliAnalysisManager.h"
#include "AliESDEvent.h"
#include "AliESDtrack.h"
#include "AliInputEventHandler.h"
#include "AliESDInputHandler.h"
#include "AliESDv0KineCuts.h"
#include "AliESDv0.h"
#include "AliCentrality.h"

#include "AliTRDdigitsManager.h"
#include "AliTRDarrayADC.h"

#include "TChain.h"
#include "TFile.h"

class TCanvas;
class TAxis;
class TFile;
class TStyle;
class TString;
class TH1F;
class TH2D;
class THnSparse;
class TLegend;
class TVirtualFitter;
class AliESDtrackCuts;
class AliStack;
class AliMCParticle;


using namespace std;

ClassImp(AliTRDPIDrawData)



//________________________________________________________________________
AliTRDPIDrawData::AliTRDPIDrawData(const char *name)
    : AliAnalysisTaskSE(name), fV0tags(0x0), fV0cuts(0x0), fV0electrons(0x0), fV0pions(0x0), 
    fESDEvent(0), fOutputContainer(0), fESDtrackCuts(0),
    fESDtrackCutsV0(0), fListQATRD(0x0), fListQATRDV0(0x0),
    fNumTagsStored(0), fCollisionSystem(3),
    fDigitsInputFile(0), fDigitsOutputFile(0),
    fEventNoInFile(-1), fDigMan(0)
{
  //fhtrackCuts(0), fhNumberEle(0),fhNumberEleCut(0),fhNumberEleCutp(0), fhNumberPion(0),fhNumberPionCut(0),fhNumberPionCutp(0), fhCent(0), fhEv(0), fhArmenteros(0)
  //
  // Constructor
  //

  fDigMan = new AliTRDdigitsManager;
  fDigMan->CreateArrays();

  DefineInput(0, TChain::Class());
  DefineOutput(1, TList::Class());

}


//_________________________________________________
AliTRDPIDrawData::~AliTRDPIDrawData()
{

  //
  // Destructor
  //

    delete fDigitsInputFile;
    delete fDigitsOutputFile;

    delete fDigMan;
    
    delete fV0cuts;
    delete fV0electrons;
    delete fV0pions;
    delete fV0tags;
    fV0tags = 0;
    fNumTagsStored = 0;
}


//________________________________________________________________________
void AliTRDPIDrawData::UserCreateOutputObjects()
{
    //
    // Definition of user output ntuple and histogram file
    //


    AliInputEventHandler* inputHandler = dynamic_cast<AliInputEventHandler*>(AliAnalysisManager::GetAnalysisManager()->GetInputEventHandler());
    if (!inputHandler)
    printf("Inputhandler not available \n");
  
   // V0 Kine cuts 
  fV0cuts = new AliESDv0KineCuts();

  // V0 PID Obj arrays
  fV0electrons = new TObjArray;
  fV0pions     = new TObjArray;
  


  OpenFile(1);
  fListQATRD=new TList;
  fListQATRD->SetOwner();
  fListQATRDV0=new TList;
  fListQATRDV0->SetOwner();
  fListQATRDV0->SetName("V0decay");
  fListQATRD->Add(fListQATRDV0);

  SetupV0qa();

  fhtrackCuts  = new TH1F("fhtrackCuts","TrackEventCuts QA",10,-0.5,9.5);
  fListQATRD->Add(fhtrackCuts);

  fhCent = new TH1F("cent","cent",10,0,100);
  fListQATRD->Add(fhCent);
  fhEv = new TH1F("Ev","Ev",10,0,10);
  fListQATRD->Add(fhEv);
  fhTPCsignalvsp = new TH2F("TPC","TPC",100,0,10,200,0,200);
  fListQATRD->Add(fhTPCsignalvsp);
  
  fhNumberEle = new TH2F("numberele","numberele",10,0,100,200,0,200);
  fListQATRD->Add(fhNumberEle);
  fhNumberEleCutp = new TH2F("numberelecutp","numberelecutp",10,0,100,200,0,200);
  fListQATRD->Add(fhNumberEleCutp);

  fhNumberEleEvent = new TH1F("numbereleevent","numbereleevent",10,0,100);
  fListQATRD->Add(fhNumberEleEvent);
  fhNumberEleEventCutp = new TH1F("numbereleeventcutp","numbereleeventcutp",10,0,100);
  fListQATRD->Add(fhNumberEleEventCutp);

  for(Int_t i=0;i<6;i++){
    fhNumberEleCut[i] = new TH2F(Form("numberelecut%i",i),Form("numberelecut%i",i),10,0,100,200,0,200);
    fListQATRD->Add(fhNumberEleCut[i]);
    fhNumberEleEventCut[i] = new TH1F(Form("numbereleeventcut%i",i),Form("numbereleeventcut%i",i),10,0,100);
    fListQATRD->Add(fhNumberEleEventCut[i]);
    
    fhNumberPionCut[i] = new TH2F(Form("numberpioncut%i",i),Form("numberpioncut%i",i),10,0,100,200,0,200);
    fListQATRD->Add(fhNumberPionCut[i]);
    fhNumberPionEventCut[i] = new TH1F(Form("numberpioneventcut%i",i),Form("numberpioneventcut%i",i),10,0,100);
    fListQATRD->Add(fhNumberPionEventCut[i]);    
  }

  fhNumberPion = new TH2F("numberpion","numberpion",10,0,100,200,0,200);
  fListQATRD->Add(fhNumberPion);
  fhNumberPionCutp = new TH2F("numberpioncutp","numberpioncutp",10,0,100,200,0,200);
  fListQATRD->Add(fhNumberPionCutp);

  fhNumberPionEvent = new TH1F("numberpionevent","numberpionevent",10,0,100);
  fListQATRD->Add(fhNumberPionEvent);
  fhNumberPionEventCutp = new TH1F("numberpioneventcutp","numberpioneventcutp",10,0,100);
  fListQATRD->Add(fhNumberPionEventCutp);


  
  PostData(1,fListQATRD);


}

//_____________________________________________________________________________
Bool_t AliTRDPIDrawData::UserNotify()
{
  delete fDigitsInputFile;
  delete fDigitsOutputFile;
  
  AliESDInputHandler *esdH = dynamic_cast<AliESDInputHandler*> (AliAnalysisManager::GetAnalysisManager()->GetInputEventHandler());
  
  TString ofname = esdH->GetInputFileName();
  TString ifname = ofname;

  ifname.ReplaceAll("AliESDs.root", "TRD.Digits.root");
  ofname.ReplaceAll("AliESDs.root", "TRD.FltDigits.root");

  fDigitsInputFile  = new TFile(ifname);
  fDigitsOutputFile = new TFile(ofname,"RECREATE");
  
  fEventNoInFile = 0;
  
}

//_____________________________________________________________________________
void AliTRDPIDrawData::UserExec(Option_t *)
{
    //
    //calls the Process function
    //

    AliESDInputHandler *esdH = dynamic_cast<AliESDInputHandler*> (AliAnalysisManager::GetAnalysisManager()->GetInputEventHandler());

    if (!esdH) {
      printf("ERROR: Could not get ESDInputHandler \n");
    }
    else fESDEvent = (AliESDEvent *) esdH->GetEvent();
    
    if(fESDEvent) fhEv->Fill(1,1);

    FillV0PIDlist();
    //
    Process(fESDEvent);

    PostData(1,fListQATRD);

    // Clear the V0 PID arrays
    ClearV0PIDlist();

    // increment the event counter for this file
    fEventNoInFile++;
    
}



//________________________________________________________________________
void AliTRDPIDrawData::Process(AliESDEvent *const esdEvent)
{
    //
    //called for each event
    //

  if (!esdEvent) {
    Printf("ERROR: esdEvent not available"); 
    return;
  }
  fhEv->Fill(3,1);
  
  Bool_t keepEvent = kFALSE;
  Bool_t keepDet[540];

  for (Int_t i=0; i<540; i++) {
    keepDet[i] = kFALSE;
  }
  
  Float_t centralityFper=99;

  AliCentrality *esdCentrality = esdEvent->GetCentrality();
  centralityFper = esdCentrality->GetCentralityPercentile("V0M");
  if(centralityFper==0.0) return;

  const AliESDVertex* fESDEventvertex = esdEvent->GetPrimaryVertexTracks(); 
  if (!fESDEventvertex)
    return;

 
  Int_t ncontr = fESDEventvertex->GetNContributors();

  if (ncontr <= 0) return;
 

  // printf("numbers %f %f \n",centralityFper,fV0electrons->GetEntries());
  if(fV0electrons->GetEntries()>0){
    fhNumberEle->Fill(centralityFper,fV0electrons->GetEntries());
    fhNumberEleEvent->Fill(centralityFper,1);
  }
  if((fhNumberPion)&&(fV0pions->GetEntries()>0)) fhNumberPion->Fill(centralityFper,fV0pions->GetEntries());
  if(fhCent) fhCent->Fill(centralityFper);

  
				    
  Int_t counterele=0;
  Int_t counterelep=0;
  Int_t counterelet[6];
  for(Int_t i=0;i<6;i++) counterelet[i]=0;
  
  // - Begin: track loop for electrons from V0 -
    for(Int_t itrack = 0; itrack < fV0electrons->GetEntries(); itrack++){
    //  AliVTrack *track=(AliVTrack*)fV0electrons->At(itrack);
      AliESDtrack *track=(AliESDtrack*)fV0electrons->At(itrack);

      for(Int_t i=0;i<6;i++){
        if(PassTrackCuts(track,i)&&PassTrackPIDCuts(track)) counterelet[i]++;
      }
	fhTPCsignalvsp->Fill(track->GetP(),track->GetTPCsignal());
     
      if(PassTrackCuts(track,3)&&PassTrackPIDCuts(track)){
	if((track->GetP()>1.2)&&(track->GetP()<4)) counterelep++;
	//	fhTPCsignalvsp->Fill(track->GetP(),track->GetTPCsignal());
      }

    } // - End: track loop for electrons from V0 -
    
    for(Int_t i=0;i<6;i++){
      if(counterelet[i]>0) fhNumberEleCut[i]->Fill(centralityFper,counterelet[i]);
      if(counterelet[i]>0) fhNumberEleEventCut[i]->Fill(centralityFper,1);
    }
    if(counterelep>0) {
      fhNumberEleCutp->Fill(centralityFper,counterelep);
      fhNumberEleEventCutp->Fill(centralityFper,1);
    }

    // Int_t counterpion=0;
  Int_t counterpionp=0;
  Int_t counterpiont[6];
  for(Int_t i=0;i<6;i++) counterpiont[i]=0;
  
  // - Begin: track loop for pions from V0 -
    for(Int_t itrack = 0; itrack < fV0pions->GetEntries(); itrack++){
	//       AliVTrack *track=(AliVTrack*)fV0pions->At(itrack);
	AliESDtrack *track=(AliESDtrack*)fV0pions->At(itrack);
	
	//	 if(!PassTrackCuts(track)) continue;
      for(Int_t i=0;i<6;i++){
        if(PassTrackCuts(track,i)) counterpiont[i]++;
      }

      //WriteRaw();
      keepEvent = kTRUE;

      // counterpion++;
      if((track->GetP()>1.2)&&(track->GetP()<4)&&(PassTrackCuts(track,3))) counterpionp++;
    }
    //  if(fhNumberPionCut) fhNumberPionCut->Fill(centralityFper,counterpion);
    //if(fhNumberPionCutp) fhNumberPionCutp->Fill(centralityFper,counterpionp);
    for(Int_t i=0;i<6;i++){
      if(counterpiont[i]>0){
	fhNumberPionCut[i]->Fill(centralityFper,counterpiont[i]);
        fhNumberPionEventCut[i]->Fill(centralityFper,1);
      }
    }
    if(counterpionp>0) {
      fhNumberPionCutp->Fill(centralityFper,counterpionp);
      fhNumberPionEventCutp->Fill(centralityFper,1);
    }


    if (keepEvent) {

      // dummy decision which detectors to keep, just to check
      // rejection; adjust the increment to emulate different
      // cuts
      for (Int_t i=0; i<540; i+=1) {
	keepDet[i] = kTRUE;
      }

      // load the digits from TRD.Digits.root
      ReadDigits();

      // get rid of data from chambers that are not used
      for (Int_t det=0; det<540; det++) {
	if ( ! keepDet[det] ) {
	  fDigMan->ClearArrays(det);
	  fDigMan->ClearIndexes(det);
	  fDigMan->GetDigits(det)->Expand();
	  fDigMan->GetDigits(det)->Reset();
	  fDigMan->GetDigits(det)->Compress();
    
	}
      }      
      
      // store the digits in TRD.FltDigits.root
      WriteDigits();
    }
    
    PostData(1,fListQATRD);
}


//________________________________________________________________________
void AliTRDPIDrawData::ReadDigits()
{
  TTree* tr = (TTree*)fDigitsInputFile->Get(Form("Event%d/TreeD",
						 fEventNoInFile));
  for (Int_t det=0; det<540; det++) {
    fDigMan->ClearArrays(det);
    fDigMan->ClearIndexes(det);
  }

  fDigMan->ReadDigits(tr);
  delete tr;
}

//________________________________________________________________________
void AliTRDPIDrawData::WriteDigits()
{
  TDirectory* evdir =
    fDigitsOutputFile->mkdir(Form("Event%d", fEventNoInFile),
			     Form("Event%d", fEventNoInFile));

  evdir->Write();
  evdir->cd();

  TTree* tr = new TTree("TreeD", "TreeD");
  fDigMan->MakeBranch(tr);
  fDigMan->WriteDigits();
  delete tr;
}

//________________________________________________________________________
void AliTRDPIDrawData::WriteRaw()
{

  // do something

}

			      



//________________________________________________________________________
void AliTRDPIDrawData::Terminate(const Option_t *)
{
    //
    // Terminate function
    //


}


//______________________________________________________________________________
void AliTRDPIDrawData::FillV0PIDlist(){

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

  fNumTagsStored = numTracks;

  // loop over V0 particles
  for(Int_t iv0=0; iv0<event->GetNumberOfV0s();iv0++){


    AliESDv0 *v0 = (AliESDv0 *) event->GetV0(iv0);
 
    if(!v0) continue;
    if(v0->GetOnFlyStatus()) continue;

//    Double_t effmass=v0->GetEffMass(0,0);
    // printf("effmass %f\n",effmass);
    // if(effmass>0.15) continue;
   	 
    // Get the particle selection 
    Bool_t foundV0 = kFALSE;
    Int_t pdgV0, pdgP, pdgN;
    foundV0 = fV0cuts->ProcessV0(v0, pdgV0, pdgP, pdgN);
    if(!foundV0) continue;
    Int_t iTrackP = v0->GetPindex();  // positive track
    Int_t iTrackN = v0->GetNindex();  // negative track
    //printf("effmass %f\n",effmass);
    // v0 Armenteros plot (QA)
    Float_t armVar[2] = {0.0,0.0};
    fV0cuts->Armenteros(v0, armVar);
//    if ( !(TMath::Power(armVar[0]/0.95,2)+TMath::Power(armVar[1]/0.05,2) < 1) ) continue;


    if(fListQATRDV0&&fhArmenteros)fhArmenteros->Fill(armVar[0],armVar[1]);

    // fill the Object arrays
    // positive particles
    if( pdgP == -11){
	fV0electrons->Add((AliVTrack*)event->GetTrack(iTrackP));
        fV0tags[iTrackP] = 11;
	//	printf("effmass here %f\n",effmass);
    }
    else if( pdgP == 211){
	fV0pions->Add((AliVTrack*)event->GetTrack(iTrackP));
        fV0tags[iTrackP] = 211;
    }
    

    // negative particles
    if( pdgN == 11){
	fV0electrons->Add((AliVTrack*)event->GetTrack(iTrackN));
        fV0tags[iTrackN] = -11;
    }
    else if( pdgN == -211){
	fV0pions->Add((AliVTrack*)event->GetTrack(iTrackN));
        fV0tags[iTrackN] = -211;
    }
   


  }
}

//______________________________________________________________________________
Int_t AliTRDPIDrawData::GetV0tag(Int_t trackIndex) const
{
  //
  // Get the tag for the corresponding trackIndex. Returns -99 in case of invalid index/tag list.
  //



  if (trackIndex < 0 || trackIndex >= fNumTagsStored || !fV0tags) return -99;
  else
  {
      return fV0tags[trackIndex];
  }
}

//______________________________________________________________________________
void AliTRDPIDrawData::ClearV0PIDlist(){

  //
  // Clear the PID object arrays
  //

  fV0electrons->Clear();
  fV0pions->Clear();
 

  delete fV0tags;
  fV0tags = 0;

  fNumTagsStored = 0;
}


//______________________________________________________________________________
void AliTRDPIDrawData::SetupV0qa()
{
  //
  // Create the qa objects for V0 Kine cuts
  //
  
  fhArmenteros  = new TH2F("fhArmenteros","Armenteros plot",200,-1.,1.,200,0.,0.4);
  fListQATRDV0->Add(fhArmenteros);
 
}


//________________________________________________________________________
Bool_t AliTRDPIDrawData::PassTrackPIDCuts(AliESDtrack *fESDTrack)
{
    //
    // check if tracks pass minimum quality critieria
    //
    if(!fESDTrack) return kFALSE;
    if(fESDTrack->GetTPCsignal()<85) kFALSE;
    if(fESDTrack->GetTPCsignal()>115) kFALSE;
    return kTRUE;
}

//________________________________________________________________________
Bool_t AliTRDPIDrawData::PassTrackCuts(AliESDtrack *fESDTrack, Int_t threshold)
{
    //
    // check if tracks pass minimum quality critieria
    //
  
    if(!fESDTrack) return kFALSE;

    // DCA to PV
    Float_t dca[2];
    fESDTrack->GetImpactParameters(dca[0],dca[1]);
    if(dca[0]>5||dca[1]>10) return kFALSE;
    
    // eta cut
    if((TMath::Abs(fESDTrack->Eta()))>0.9) return kFALSE;

    //TRD out
    if((fESDTrack->GetStatus()&AliVTrack::kTRDout)==0)return kFALSE;

    // TPC refit
    if((fESDTrack->GetStatus()&AliVTrack::kTPCrefit)==0)return kFALSE;
    // remove kinks
    if(fESDTrack->GetKinkIndex(0)>0) return kFALSE;

    Float_t tpcchi2=99;
    Int_t tpcnclusF=fESDTrack->GetTPCNclsF();
    if(tpcnclusF!=0) tpcchi2=(Float_t)fESDTrack->GetTPCchi2()/tpcnclusF;
    else tpcchi2=1000;
    if(tpcchi2 > 4) return kFALSE;

    
    Int_t ntrackletstracking=fESDTrack->GetTRDntracklets();
    if(ntrackletstracking<threshold) return kFALSE;
 
     // QA #TRD PID tracklets 
    if(fESDTrack->GetTRDntrackletsPID()<threshold) return kFALSE;
    
    Int_t ntrl=0;
    for(Int_t jPl=0;jPl<6;jPl++){
	Double_t signal=0;
	for(int isl= 0; isl<= 8;isl++){
	    Double_t sigsl=fESDTrack->GetTRDslice(jPl,isl);
	    if(sigsl>0)signal+=sigsl;
	}
        // if signal is missing, stop counting
	if(signal<=0||fESDTrack->GetTRDmomentum(jPl)<=0)break;
	ntrl++;
    }
    if(ntrl<threshold) return kFALSE;

   

    return kTRUE;
}

