
#include "TAxis.h"
#include "TF1.h"
#include "TGraph.h"
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TH2F.h"
//#include "TNuple.h"
#include "TFolder.h"
#include "TList.h"
#include "TLegend.h"
#include "TText.h"

#include <iostream>


void histMod(){		

	TCanvas *C = new TCanvas("C","C",800,600);
	TFile* file = new TFile("DigitsExtractQA.root", "READ");
	TList *lst = (TList*)(file->FindObjectAny("cdigitqa;1"));
	///TH2F* fhdEdxe = new TH2F("fhdEdxe","fhdEdxe",100, 0, 10, 100, 0, 10);

	//fhdEdxe->Write();	
	file->GetListOfKeys()->Print();

	// get your histograms here
	gStyle->SetOptTitle(kFALSE);
	gStyle->SetOptStat(kFALSE);

	TH1F* tmp1 = (TH1*)lst->FindObject("fhsigmae2");
        TH1F* tmp2 = (TH1*)lst->FindObject("fhsigmae2");
	
	// Now design your histogram
	
	//tmp1->Rebin();
	//
	//tmp1->SetBit(TH1::kCanRebin);
	//TAxis *Xaxis2 = tmp1->GetXaxis();
	//tmp1->RebinAxis(1000, Xaxis2);
        //tmp1->GetXaxis()->SetRange(-10,10);	
	gPad->SetLogy();
	/*
	for (Int_t i = 0; i < 1000; i++){
	   y = tmp1->GetBinContent(i);
	   fit1->SetBinContent(i,max-10*(i-500)*(i-500));
	}
	*/
	//tmp1->FitSliceX([-2.2]);

	TF1* g1 = new TF1("g1","gaus",0.5,10);
  	TF1* g2 = new TF1("g2","gaus",-8,0.5);	

	g1->SetLineColor(kMagenta);
	g2->SetLineColor(kRed);
	tmp1->SetLineColor(kBlack);
	//tmp2->SetLineColor(kRed);
	
	//tmp1->Fit(g1, "R");
	//tmp1->Fit(g2, "R+");
	//tmp1.Fit("gaus","1","1",-10,-2);
 	//tmp1.Fit("gaus","2","2",-3, 7);
	//g1->SetLineColor(kGreen);
	//tmp2->SetLineColor(kBlue);	
	//tmp1->SetMarkerStyle(20); // add points on the line
	
	tmp1->GetXaxis()->SetTitle("#frac{dE/dx - <dE/dx>_{e}}{#sigma_{e}}");
	tmp1->GetYaxis()->SetTitle("Counts");

	tmp1->Sumw2();
	tmp2->Sumw2();	
	//h1 = tmp2->Clone("h1");	
	TText *t = new TText(4,66000,"Pion - Electron Sample");	
	TText *pi = new TText(-7.1, 40000,"pion peak");
	TText *pe = new TText(-2.2, 8000, "electron peak");
	t->SetTextFont(32);

	//pi->SetTextSize(28);
	//pe->SetTextSize(28);

	pi->SetTextColor(kBlue);
	pe->SetTextColor(kRed);
	//t->SetTextSize(40);	
	//tmp2->Draw("same,e");
	//tmp1->Add(tmp2, -1);
	//tmp1->Divide(tmp1, tmp2);
	//tmp1->Scale(0.001);
	
	tmp2->Fit(g1,"R");
	tmp2->Fit(g2, "R+");
	
	tmp2->Add(g1,-1);
	tmp2->Add(g2,-1);

	//tmp1->Draw("e");
	tmp2->Draw("same,e");
	t->Draw();
	pi->Draw();
	pe->Draw();
	
	Double_t m1 = 0;
	Double_t m2 = 0;
	Double_t utmp1 = 0;
	Double_t utmp2 = 0;
	for (Int_t i = 0; i < 500; i++){
	    utmp1 += tmp1->GetBinError(i)*tmp1->GetBinError(i);
	    utmp2 += tmp2->GetBinError(i)*tmp2->GetBinError(i);
	    if (tmp1->GetBinContent(i)>0)
	       m1 += tmp1->GetBinContent(i);
	    cout << tmp2->GetBinContent(i) << endl;
	    m2 += tmp2->GetBinContent(i);
	}		

	//cout << (tmp1->GetBinContent(200)/tmp2->GetBinContent(200)); 
	cout << m1 << " += " << utmp1 << endl;
	cout << m2 << " += " << utmp2 << endl;
	//resize->Draw("same");
	//fit1->Draw("same");
	//gPad->BuildLegend();
	//TLegend legend = new TLegend(0.1,0.7,0.48,0.9);
	//legend->SetLegendBorderSize(0.01);
	//legend->SetLegendFont(32);
	//SetLegendBorderSize(0.01);
	//gPad->BuildLegend();
	/*TAxis*axis = new TGaxis(gPad->GetUxmax(),gPad->GetUymin(),
                            gPad->GetUxmax(),gPad->GetUymax(),
                            0,rightmax,510,"+L");
  	axis->SetLineColor(kRed);
  	axis->SetLabelColor(kRed);
 	axis->Draw();*/
 	//tmp1->Write();
	file->Close();
	//
}
