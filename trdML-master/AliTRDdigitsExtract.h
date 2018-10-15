#ifndef ALITRDdigitsExtract_H
#define ALITRDdigitsExtract_H

// example of an analysis task to analyse TRD digits
// Authors: Tom Dietel
// based on the AliAnalysisTaskPt

class AliESDEvent;
class AliTRDdigitsManager;
class AliPIDResponse;

#include "AliAnalysisTaskSE.h"

//class AliPIDResponse; 

class AliTRDdigitsExtract : public AliAnalysisTaskSE {
public:
    typedef enum{
        kpp = 0,
        kpPb = 1,
        kPbPb = 2
      } ECollisionSystem_t;

  AliTRDdigitsExtract()
    : AliAnalysisTaskSE(),
      fDigMan(0), fGeo(0),
      fESD(0), fOutputList(0),
      fDigitsInputFileName("TRD.FltDigits.root"),
      fDigitsOutputFileName("DigitsQA.3.root"),
      fDigitsInputFile(0), fDigitsOutputFile(0),
      fPIDResponse(0)
  {}
  AliTRDdigitsExtract(const char *name);
  virtual ~AliTRDdigitsExtract() {}

  virtual void   UserCreateOutputObjects();
  virtual Bool_t UserNotify();
  virtual void   UserExec(Option_t *option);
  virtual void   Terminate(Option_t *);

  Int_t          *fV0tags;  //! Pointer to array with tags for identified particles from V0 decays

  Bool_t Ispp() const { return fCollisionSystem.TestBitNumber(kpp); }
  Bool_t IspPb() const { return fCollisionSystem.TestBitNumber(kpPb); }
  Bool_t IsPbPb() const { return fCollisionSystem.TestBitNumber(kPbPb); }

  void SetCollisionSystem(ECollisionSystem_t system){
      fCollisionSystem.Clear();
      fCollisionSystem.SetBitNumber(system, kTRUE);
  }
  void SetppAnalysis(){
      fCollisionSystem.SetBitNumber(kpPb, kFALSE);
      fCollisionSystem.SetBitNumber(kPbPb, kFALSE);
      fCollisionSystem.SetBitNumber(kpp, kTRUE);
  }
  void SetpPbAnalysis() {
      fCollisionSystem.SetBitNumber(kpp, kFALSE);
      fCollisionSystem.SetBitNumber(kPbPb, kFALSE);
      fCollisionSystem.SetBitNumber(kpPb, kTRUE);
  }
  void SetPbPbAnalysis() {
      fCollisionSystem.SetBitNumber(kpp, kFALSE);
      fCollisionSystem.SetBitNumber(kpPb, kFALSE);
      fCollisionSystem.SetBitNumber(kPbPb, kTRUE);
  };

  void SetDigitsInputFilename(TString x) {fDigitsInputFileName=x;}
  void SetDigitsOutputFilename(TString x) {fDigitsOutputFileName=x;}

protected:

    AliESDv0KineCuts *fV0cuts;           //! ESD V0 cuts
    TObjArray *fV0electrons;             //! array with pointer to identified particles from V0 decays (electrons)
    TObjArray *fV0pions;                 //! array with pointer to identified particles from V0 decays (pions)
    TObjArray *fV0protons;

  void ReadDigits();
  void WriteDigits();

  void FillV0PIDlist();

  void DigitsDictionary(Int_t iTrack, Int_t iTracklet, Int_t pdgCode);

  AliTRDtrackV1* FindTRDtrackV1(AliESDfriendTrack* friendtrack);

  AliTRDdigitsManager* fDigMan; //! digits manager
  AliTRDgeometry* fGeo; //! TRD geometry


  TFile* OpenDigitsFile(TString inputfile, TString digfile, TString opt);

  Int_t FindTrackletPos(AliTRDtrackV1* trdTrack, Int_t layer,
			Int_t* det, Int_t* row, Int_t* col);

private:
  AliESDEvent* fESD;    //! ESD object  
  TList*       fOutputList; //! Output list

  TH1F* fHistPt;
  TH1F* fHistSE;     //! graph the sigma quantity electron
  TH1F* fHistSPi;    //!
  TH1F* fHistSPr;    //!
  TH2* fHistdEdx;   //!

  TString fDigitsInputFileName;         //! Name of digits file for reading
  TString fDigitsOutputFileName;        //! Name of digits file for writing

  TFile* fDigitsInputFile;             //! Digits file for reading
  TFile* fDigitsOutputFile;            //! Digits file for writing

  AliPIDResponse* fPIDResponse;		//!

  Int_t fEventNoInFile;
  
  Int_t fEventsLost;		// * events with no tracks in them
  Int_t fNumOfPions;		// * # of pions
  Int_t fNumOfProtons;		// * # of protons
  Int_t fNumOfElectrons;	// * # of electrons
  Int_t fNumOfPositrons;	// * 
  Int_t fElectronOver3Sigmas;   

  Int_t universalTracki;

  TBits fCollisionSystem;              //! Collision System;

  AliTRDdigitsExtract(const AliTRDdigitsExtract&); // not implemented
  AliTRDdigitsExtract& operator=(const AliTRDdigitsExtract&); // not implemented

  ClassDef(AliTRDdigitsExtract, 1); // example of analysis
};

#endif
