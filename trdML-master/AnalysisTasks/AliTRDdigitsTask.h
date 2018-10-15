#ifndef ALITRDDIGITSTASK_H
#define ALITRDDIGITSTASK_H

// example of an analysis task to analyse TRD digits 
// Authors: Tom Dietel
// based on the AliAnalysisTaskPt

class AliESDEvent;
class AliTRDdigitsManager;

#include "AliAnalysisTaskSE.h"

class AliTRDdigitsTask : public AliAnalysisTaskSE {
public:
  AliTRDdigitsTask()
    : AliAnalysisTaskSE(),
      fDigMan(0), fGeo(0),
      fESD(0), fOutputList(0),
      fDigitsInputFileName("TRD.FltDigits.root"),
      fDigitsInputFile(0), fDigitsOutputFile(0)
  {}
  AliTRDdigitsTask(const char *name);
  virtual ~AliTRDdigitsTask() {}
  
  virtual void   UserCreateOutputObjects();
  virtual Bool_t UserNotify();
  virtual void   UserExec(Option_t *option);
  virtual void   Terminate(Option_t *);

  void SetDigitsInputFilename(TString x) {fDigitsInputFileName=x;}
  void SetDigitsOutputFilename(TString x) {fDigitsOutputFileName=x;}
  
protected:

  void ReadDigits();
  void WriteDigits();

  AliTRDtrackV1* FindTRDtrackV1(AliESDfriendTrack* friendtrack);

  AliTRDdigitsManager* fDigMan; //! digits manager
  AliTRDgeometry* fGeo; //! TRD geometry


  Int_t FindTrackletPos(AliTRDtrackV1* trdTrack, Int_t layer,
			Int_t* det, Int_t* row, Int_t* col);
  
private:
  AliESDEvent *fESD;    //! ESD object
  TList       *fOutputList; //! Output list
  TH1*        fhTrackCuts;  //! track cut statistics
  TH1*        fhPtAll; //!
  TH1*        fhPtTRD; //!
  TH1*        fhTrdAdc; //!

  //TH1F        *fHistAdcSpectrum; //! TRD ADC spectrum
  //TH2F        *fHistPadResponse; //! (pseudo) pad response function

  TString fDigitsInputFileName;         //! Name of digits file for reading
  TString fDigitsOutputFileName;        //! Name of digits file for writing
  
  TFile* fDigitsInputFile;             //! Digits file for reading
  TFile* fDigitsOutputFile;            //! Digits file for writing

  Int_t fEventNoInFile;
  Int_t fDataloss;			// number of events ignored (not added to the file)

  Int_t fNumOfPions;			//
  Int_t fNumOfProtons;			//
  Int_t fNumOfElectrons;		// 
 
  AliTRDdigitsTask(const AliTRDdigitsTask&); // not implemented
  AliTRDdigitsTask& operator=(const AliTRDdigitsTask&); // not implemented
  
  ClassDef(AliTRDdigitsTask, 1); // example of analysis
};

#endif
