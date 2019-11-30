#ifndef ALITRDdigitsExtract_H
#define ALITRDdigitsExtract_H

// example of an analysis task to analyse TRD digits
// Authors: Tom Dietel
// based on the AliAnalysisTaskPt

class AliPIDResponse;
class AliESDEvent;
class AliTRDdigitsManager;

#include "AliAnalysisTaskSE.h"
#include "AliTRDdigitsTask.h"

#include <iostream>

class AliTRDdigitsExtract : public AliTRDdigitsTask {
public:

  //AliTRDdigitsExtract();
  AliTRDdigitsExtract(const char *name="AliTRDdigitsExtract");
  virtual ~AliTRDdigitsExtract() {}

  virtual void   UserCreateOutputObjects();
  //virtual Bool_t UserNotify();
  virtual void   UserExec(Option_t *option);
  virtual void   Terminate(Option_t *);
  void SetDigitsInputFilename(TString x) {fDigitsInputFileName=x;}
  void DigitsDictionary(AliESDtrack* track, Int_t i, Int_t iTrack, Int_t iV0, Int_t pdgCode);

public:
  void SetV0KineCuts(AliESDv0KineCuts *c){
    fV0cuts = c;
    std::cout << "set V0 cuts to " << fV0cuts << std::endl;
  }
  AliESDv0KineCuts* GetV0KineCuts() {return fV0cuts;}

protected:

  void FillV0PIDlist();

  virtual void AnalyseEvent();

  AliPIDResponse* fPIDResponse;

  Int_t fEventNoInFile;
  Int_t universalTracki;
  //Int_t runNumber;

private:

  AliTRDdigitsExtract(const AliTRDdigitsExtract&); // not implemented
  AliTRDdigitsExtract& operator=(const AliTRDdigitsExtract&); // not implemented
  TString fDigitsInputFileName;
  ClassDef(AliTRDdigitsExtract, 2); // example of analysis
};

#endif
