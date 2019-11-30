#ifndef AliAnalysisTaskTom_cxx
#define AliAnalysisTaskTom_cxx

// example of an analysis task creating a p_t spectrum
// Authors: Panos Cristakoglou, Jan Fiete Grosse-Oetringhaus, Christian Klein-Boesing

class TH1F;
class AliESDEvent;

#include "AliAnalysisTaskSE.h"

class AliAnalysisTaskTom : public AliAnalysisTaskSE {
 public:
  AliAnalysisTaskTom() : AliAnalysisTaskSE(), fESD(0), fOutputList(0), fHistPt(0) {}
  AliAnalysisTaskTom(const char *name);
  virtual ~AliAnalysisTaskTom() {}
  
  virtual void   UserCreateOutputObjects();
  virtual void   UserExec(Option_t *option);
  virtual void   Terminate(Option_t *);
  
 private:
  AliESDEvent *fESD;    //! ESD object
  TList       *fOutputList; //! Output list
  TH1F        *fHistPt; //! Pt spectrum
   
  AliAnalysisTaskTom(const AliAnalysisTaskTom&); // not implemented
  AliAnalysisTaskTom& operator=(const AliAnalysisTaskTom&); // not implemented
  
  ClassDef(AliAnalysisTaskTom, 1); // example of analysis
};

#endif
