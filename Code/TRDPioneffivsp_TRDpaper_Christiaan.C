void TRDPioneffivsp_TRDpaper_Christiaan()
{
//=========Macro generated from canvas: TRDPIDPerformance/TRDPIDPerformance
//=========  (Sat Nov 16 10:01:54 2019) by ROOT version 6.18/00
   TCanvas *TRDPIDPerformance = new TCanvas("TRDPIDPerformance", "TRDPIDPerformance",10,53,800,800);
   gStyle->SetOptStat(0);
   TRDPIDPerformance->Range(-0.9361547,-3.353874,6.265036,-0.4049206);
   TRDPIDPerformance->SetFillColor(10);
   TRDPIDPerformance->SetBorderMode(0);
   TRDPIDPerformance->SetBorderSize(1);
   TRDPIDPerformance->SetLogy();
   TRDPIDPerformance->SetTickx(1);
   TRDPIDPerformance->SetTicky(1);
   TRDPIDPerformance->SetLeftMargin(0.13);
   TRDPIDPerformance->SetRightMargin(0.03);
   TRDPIDPerformance->SetTopMargin(0.04);
   TRDPIDPerformance->SetBottomMargin(0.12);
   TRDPIDPerformance->SetFrameFillColor(0);
   TRDPIDPerformance->SetFrameLineWidth(2);
   TRDPIDPerformance->SetFrameBorderMode(0);
   TRDPIDPerformance->SetFrameLineWidth(2);
   TRDPIDPerformance->SetFrameBorderMode(0);
   
   TH1F *hist__1 = new TH1F("hist__1","",100,0,6.575);
   hist__1->SetMinimum(0.001);
   hist__1->SetMaximum(0.3);
   hist__1->SetDirectory(0);
   hist__1->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   hist__1->SetLineColor(ci);
   hist__1->SetLineWidth(2);
   hist__1->GetXaxis()->SetTitle("#it{p} (GeV/#it{c})");
   hist__1->GetXaxis()->SetRange(1,92);
   hist__1->GetXaxis()->SetLabelFont(42);
   hist__1->GetXaxis()->SetLabelOffset(0.01);
   hist__1->GetXaxis()->SetLabelSize(0.045);
   hist__1->GetXaxis()->SetTitleSize(0.05);
   hist__1->GetXaxis()->SetTickLength(0.02);
   hist__1->GetXaxis()->SetTitleOffset(1.1);
   hist__1->GetXaxis()->SetTitleFont(42);
   hist__1->GetYaxis()->SetTitle("Pion efficiency");
   hist__1->GetYaxis()->SetNdivisions(506);
   hist__1->GetYaxis()->SetLabelFont(42);
   hist__1->GetYaxis()->SetLabelOffset(0.01);
   hist__1->GetYaxis()->SetLabelSize(0.045);
   hist__1->GetYaxis()->SetTitleSize(0.05);
   hist__1->GetYaxis()->SetTickLength(0.02);
   hist__1->GetYaxis()->SetTitleOffset(1.3);
   hist__1->GetYaxis()->SetTitleFont(42);
   hist__1->GetZaxis()->SetNdivisions(506);
   hist__1->GetZaxis()->SetLabelFont(42);
   hist__1->GetZaxis()->SetLabelSize(0.03);
   hist__1->GetZaxis()->SetTitleSize(0.035);
   hist__1->GetZaxis()->SetTickLength(0.02);
   hist__1->GetZaxis()->SetTitleOffset(1.05);
   hist__1->GetZaxis()->SetTitleFont(42);
   hist__1->Draw("");
   
   Double_t LQ1D_5trl_PionEffvsP_fx3001[13] = {
   0.4246237,
   0.5985138,
   0.7952877,
   0.9949122,
   1.195459,
   1.439668,
   1.781109,
   2.220042,
   2.719777,
   3.386895,
   4.633428,
   6.708676,
   8.769783};
   Double_t LQ1D_5trl_PionEffvsP_fy3001[13] = {
   0.05995413,
   0.02618553,
   0.01798861,
   0.01464972,
   0.0130867,
   0.01180527,
   0.01225982,
   0.01265445,
   0.01365352,
   0.015793,
   0.02082409,
   0.02549183,
   0.0432357};
   Double_t LQ1D_5trl_PionEffvsP_felx3001[13] = {
   0.1746237,
   0.09851377,
   0.0952877,
   0.09491219,
   0.09545939,
   0.1396684,
   0.1811093,
   0.2200421,
   0.2197765,
   0.3868948,
   0.6334283,
   0.7086763,
   0.7697825};
   Double_t LQ1D_5trl_PionEffvsP_fely3001[13] = {
   0.00041929,
   0.0001626565,
   0.0001379372,
   0.0001425759,
   0.0001578967,
   0.0001499509,
   0.0001780204,
   0.0002343499,
   0.0003606809,
   0.000456453,
   0.0008567724,
   0.002538123,
   0.007183467};
   Double_t LQ1D_5trl_PionEffvsP_fehx3001[13] = {
   0.07537628,
   0.1014862,
   0.1047123,
   0.1050878,
   0.1045406,
   0.1603316,
   0.2188907,
   0.2799579,
   0.2802235,
   0.6131052,
   1.366572,
   1.291324,
   1.230217};
   Double_t LQ1D_5trl_PionEffvsP_fehy3001[13] = {
   0.000421131,
   0.0001633114,
   0.0001386347,
   0.0001434929,
   0.0001591597,
   0.0001512152,
   0.0001797422,
   0.0002372427,
   0.00036706,
   0.0004652737,
   0.0008804119,
   0.002713623,
   0.008034492};
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(13,LQ1D_5trl_PionEffvsP_fx3001,LQ1D_5trl_PionEffvsP_fy3001,LQ1D_5trl_PionEffvsP_felx3001,LQ1D_5trl_PionEffvsP_fehx3001,LQ1D_5trl_PionEffvsP_fely3001,LQ1D_5trl_PionEffvsP_fehy3001);
   grae->SetName("LQ1D_5trl_PionEffvsP");
   grae->SetTitle("");
   grae->SetFillColor(1);
   grae->SetFillStyle(0);
   grae->SetLineWidth(2);
   grae->SetMarkerStyle(20);
   grae->SetMarkerSize(1.6);
   
   TH1F *Graph_Graph3001 = new TH1F("Graph_Graph3001","",100,0,10.975);
   Graph_Graph3001->SetMinimum(0.001);
   Graph_Graph3001->SetMaximum(0.3);
   Graph_Graph3001->SetDirectory(0);
   Graph_Graph3001->SetStats(0);
   Graph_Graph3001->SetLineWidth(2);
   Graph_Graph3001->GetXaxis()->SetTitle("#it{p} (GeV/#it{c})");
   Graph_Graph3001->GetXaxis()->SetRange(1,37);
   Graph_Graph3001->GetXaxis()->SetLabelFont(42);
   Graph_Graph3001->GetXaxis()->SetLabelOffset(0.01);
   Graph_Graph3001->GetXaxis()->SetLabelSize(0.045);
   Graph_Graph3001->GetXaxis()->SetTitleSize(0.045);
   Graph_Graph3001->GetXaxis()->SetTickLength(0.02);
   Graph_Graph3001->GetXaxis()->SetTitleOffset(1.3);
   Graph_Graph3001->GetXaxis()->SetTitleFont(42);
   Graph_Graph3001->GetYaxis()->SetTitle("pion efficiency");
   Graph_Graph3001->GetYaxis()->SetNdivisions(506);
   Graph_Graph3001->GetYaxis()->SetLabelFont(42);
   Graph_Graph3001->GetYaxis()->SetLabelOffset(0.01);
   Graph_Graph3001->GetYaxis()->SetLabelSize(0.045);
   Graph_Graph3001->GetYaxis()->SetTitleSize(0.045);
   Graph_Graph3001->GetYaxis()->SetTickLength(0.02);
   Graph_Graph3001->GetYaxis()->SetTitleOffset(1.5);
   Graph_Graph3001->GetYaxis()->SetTitleFont(42);
   Graph_Graph3001->GetZaxis()->SetNdivisions(506);
   Graph_Graph3001->GetZaxis()->SetLabelFont(42);
   Graph_Graph3001->GetZaxis()->SetLabelSize(0.03);
   Graph_Graph3001->GetZaxis()->SetTickLength(0.02);
   Graph_Graph3001->GetZaxis()->SetTitleOffset(1.05);
   Graph_Graph3001->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3001);
   
   grae->Draw("p");
   
   Double_t LQ2D_2trl_PionEffvsP_fx3002[13] = {
   0.4246237,
   0.5985138,
   0.7952877,
   0.9949122,
   1.195459,
   1.439668,
   1.781109,
   2.220042,
   2.719777,
   3.386895,
   0,
   6.708676,
   8.769783};
   Double_t LQ2D_2trl_PionEffvsP_fy3002[13] = {
   0.0182008,
   0.008690082,
   0.007186723,
   0.006815969,
   0.006465314,
   0.006566484,
   0.007098259,
   0.008929275,
   0.008960263,
   0.01155957,
   0,
   0.02786934,
   0.01497006};
   Double_t LQ2D_2trl_PionEffvsP_felx3002[13] = {
   0.1746237,
   0.09851377,
   0.0952877,
   0.09491219,
   0.09545939,
   0.1396684,
   0.1811093,
   0.2200421,
   0.2197765,
   0.3868948,
   0,
   0.7086763,
   0.7697825};
   Double_t LQ2D_2trl_PionEffvsP_fely3002[13] = {
   0.0002555716,
   0.0001013664,
   9.360688e-05,
   0.0001038548,
   0.0001180569,
   0.0001185432,
   0.0001431494,
   0.0002072308,
   0.0003064928,
   0.0004085486,
   0,
   0.002757274,
   0.004251596};
   Double_t LQ2D_2trl_PionEffvsP_fehx3002[13] = {
   0.07537628,
   0.1014862,
   0.1047123,
   0.1050878,
   0.1045406,
   0.1603316,
   0.2188907,
   0.2799579,
   0.2802235,
   0.6131052,
   0,
   1.291324,
   1.230217};
   Double_t LQ2D_2trl_PionEffvsP_fehy3002[13] = {
   0.0002579389,
   0.0001021526,
   9.442127e-05,
   0.000104914,
   0.0001194995,
   0.0001199791,
   0.000145084,
   0.0002104578,
   0.0003135798,
   0.0004182893,
   0,
   0.002946113,
   0.005231558};
   grae = new TGraphAsymmErrors(13,LQ2D_2trl_PionEffvsP_fx3002,LQ2D_2trl_PionEffvsP_fy3002,LQ2D_2trl_PionEffvsP_felx3002,LQ2D_2trl_PionEffvsP_fehx3002,LQ2D_2trl_PionEffvsP_fely3002,LQ2D_2trl_PionEffvsP_fehy3002);
   grae->SetName("LQ2D_2trl_PionEffvsP");
   grae->SetTitle("");
   grae->SetFillColor(2);
   grae->SetFillStyle(0);
   grae->SetLineColor(2);
   grae->SetLineWidth(2);
   grae->SetMarkerColor(2);
   grae->SetMarkerStyle(21);
   grae->SetMarkerSize(1.6);
   
   TH1F *Graph_LQ2D_2trl_PionEffvsP3002 = new TH1F("Graph_LQ2D_2trl_PionEffvsP3002","",100,0,11);
   Graph_LQ2D_2trl_PionEffvsP3002->SetMinimum(3.3897e-05);
   Graph_LQ2D_2trl_PionEffvsP3002->SetMaximum(0.033897);
   Graph_LQ2D_2trl_PionEffvsP3002->SetDirectory(0);
   Graph_LQ2D_2trl_PionEffvsP3002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_LQ2D_2trl_PionEffvsP3002->SetLineColor(ci);
   Graph_LQ2D_2trl_PionEffvsP3002->GetXaxis()->SetTitle("#it{p} (GeV/#it{c})");
   Graph_LQ2D_2trl_PionEffvsP3002->GetXaxis()->SetLabelFont(42);
   Graph_LQ2D_2trl_PionEffvsP3002->GetXaxis()->SetLabelOffset(0.01);
   Graph_LQ2D_2trl_PionEffvsP3002->GetXaxis()->SetLabelSize(0.045);
   Graph_LQ2D_2trl_PionEffvsP3002->GetXaxis()->SetTitleSize(0.05);
   Graph_LQ2D_2trl_PionEffvsP3002->GetXaxis()->SetTickLength(0.02);
   Graph_LQ2D_2trl_PionEffvsP3002->GetXaxis()->SetTitleOffset(1.1);
   Graph_LQ2D_2trl_PionEffvsP3002->GetXaxis()->SetTitleFont(42);
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetTitle("pion efficiency");
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetNdivisions(506);
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetLabelFont(42);
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetLabelOffset(0.01);
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetLabelSize(0.045);
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetTitleSize(0.05);
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetTickLength(0.02);
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetTitleOffset(1.3);
   Graph_LQ2D_2trl_PionEffvsP3002->GetYaxis()->SetTitleFont(42);
   Graph_LQ2D_2trl_PionEffvsP3002->GetZaxis()->SetLabelFont(42);
   Graph_LQ2D_2trl_PionEffvsP3002->GetZaxis()->SetLabelSize(0.035);
   Graph_LQ2D_2trl_PionEffvsP3002->GetZaxis()->SetTitleSize(0.035);
   Graph_LQ2D_2trl_PionEffvsP3002->GetZaxis()->SetTitleOffset(1);
   Graph_LQ2D_2trl_PionEffvsP3002->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_LQ2D_2trl_PionEffvsP3002);
   
   grae->Draw("p");
   
   Double_t truncmean_2trl_PionEffvsP_fx3003[11] = {
   0.4282423,
   0.6000849,
   0.7962816,
   0.9943921,
   1.194152,
   1.435419,
   1.775172,
   2.208079,
   2.715589,
   3.39307,
   4.66792};
   Double_t truncmean_2trl_PionEffvsP_fy3003[11] = {
   0.01087039,
   0.008806097,
   0.009541358,
   0.01061348,
   0.01210252,
   0.01536979,
   0.02205133,
   0.0257691,
   0.02921964,
   0.03458587,
   0.04394453};
   Double_t truncmean_2trl_PionEffvsP_felx3003[11] = {
   0.1782423,
   0.1000849,
   0.09628164,
   0.09439208,
   0.09415187,
   0.1354187,
   0.1751718,
   0.2080794,
   0.2155892,
   0.3930699,
   0.66792};
   Double_t truncmean_2trl_PionEffvsP_fely3003[11] = {
   0.0002332891,
   0.0001339876,
   0.0001392597,
   0.000162656,
   0.0002041303,
   0.0002315973,
   0.0003275415,
   0.0004846174,
   0.0008150425,
   0.001053386,
   0.001840933};
   Double_t truncmean_2trl_PionEffvsP_fehx3003[11] = {
   0.07175772,
   0.09991508,
   0.1037184,
   0.1056079,
   0.1058481,
   0.1645813,
   0.2248282,
   0.2919206,
   0.2844108,
   0.6069301,
   1.33208};
   Double_t truncmean_2trl_PionEffvsP_fehy3003[11] = {
   0.0002332891,
   0.0001339876,
   0.0001392597,
   0.000162656,
   0.0002041303,
   0.0002315973,
   0.0003275415,
   0.0004846174,
   0.0008150425,
   0.001053386,
   0.001840933};
   grae = new TGraphAsymmErrors(11,truncmean_2trl_PionEffvsP_fx3003,truncmean_2trl_PionEffvsP_fy3003,truncmean_2trl_PionEffvsP_felx3003,truncmean_2trl_PionEffvsP_fehx3003,truncmean_2trl_PionEffvsP_fely3003,truncmean_2trl_PionEffvsP_fehy3003);
   grae->SetName("truncmean_2trl_PionEffvsP");
   grae->SetTitle("");

   ci = TColor::GetColor("#009900");
   grae->SetFillColor(ci);
   grae->SetFillStyle(0);

   ci = TColor::GetColor("#009900");
   grae->SetLineColor(ci);
   grae->SetLineWidth(2);

   ci = TColor::GetColor("#009900");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(23);
   grae->SetMarkerSize(1.92);
   
   TH1F *Graph_Graph_Graph73003 = new TH1F("Graph_Graph_Graph73003","",100,0,6.575);
   Graph_Graph_Graph73003->SetMinimum(0.004960775);
   Graph_Graph_Graph73003->SetMaximum(0.04949679);
   Graph_Graph_Graph73003->SetDirectory(0);
   Graph_Graph_Graph73003->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph_Graph73003->SetLineColor(ci);
   Graph_Graph_Graph73003->SetLineWidth(2);
   Graph_Graph_Graph73003->GetXaxis()->SetTitle("#it{p} (GeV/#it{c})");
   Graph_Graph_Graph73003->GetXaxis()->SetLabelFont(42);
   Graph_Graph_Graph73003->GetXaxis()->SetLabelOffset(0.01);
   Graph_Graph_Graph73003->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph73003->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph73003->GetXaxis()->SetTickLength(0.02);
   Graph_Graph_Graph73003->GetXaxis()->SetTitleOffset(1.05);
   Graph_Graph_Graph73003->GetXaxis()->SetTitleFont(42);
   Graph_Graph_Graph73003->GetYaxis()->SetTitle("pion efficiency");
   Graph_Graph_Graph73003->GetYaxis()->SetNdivisions(506);
   Graph_Graph_Graph73003->GetYaxis()->SetLabelFont(42);
   Graph_Graph_Graph73003->GetYaxis()->SetLabelOffset(0.01);
   Graph_Graph_Graph73003->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph73003->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph73003->GetYaxis()->SetTickLength(0.02);
   Graph_Graph_Graph73003->GetYaxis()->SetTitleOffset(1.5);
   Graph_Graph_Graph73003->GetYaxis()->SetTitleFont(42);
   Graph_Graph_Graph73003->GetZaxis()->SetNdivisions(506);
   Graph_Graph_Graph73003->GetZaxis()->SetLabelFont(42);
   Graph_Graph_Graph73003->GetZaxis()->SetLabelSize(0.03);
   Graph_Graph_Graph73003->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph73003->GetZaxis()->SetTickLength(0.02);
   Graph_Graph_Graph73003->GetZaxis()->SetTitleOffset(1.05);
   Graph_Graph_Graph73003->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph_Graph73003);
   
   grae->Draw("p");
   
   Double_t LQ3D_5trl_PionEffvsP_fx3004[13] = {
   0.4246237,
   0.5985138,
   0.7952877,
   0.9949122,
   1.195459,
   1.439668,
   1.781109,
   2.220042,
   2.719777,
   3.386895,
   0,
   6.708676,
   8.769783};
   Double_t LQ3D_5trl_PionEffvsP_fy3004[13] = {
   0.008587062,
   0.004324015,
   0.003892957,
   0.004236479,
   0.00420567,
   0.004426742,
   0.005222723,
   0.006813136,
   0.007418575,
   0.01003108,
   0,
   0.03515625,
   0.01610306};
   Double_t LQ3D_5trl_PionEffvsP_felx3004[13] = {
   0.1746237,
   0.09851377,
   0.0952877,
   0.09491219,
   0.09545939,
   0.1396684,
   0.1811093,
   0.2200421,
   0.2197765,
   0.3868948,
   0,
   0.7086763,
   0.7697825};
   Double_t LQ3D_5trl_PionEffvsP_fely3004[13] = {
   0.0001918521,
   7.709787e-05,
   7.38549e-05,
   8.741507e-05,
   0.0001012156,
   0.0001031333,
   0.0001297245,
   0.0001905773,
   0.0002929545,
   0.0003988052,
   0,
   0.003224016,
   0.004571037};
   Double_t LQ3D_5trl_PionEffvsP_fehx3004[13] = {
   0.07537628,
   0.1014862,
   0.1047123,
   0.1050878,
   0.1045406,
   0.1603316,
   0.2188907,
   0.2799579,
   0.2802235,
   0.6131052,
   0,
   1.291324,
   1.230217};
   Double_t LQ3D_5trl_PionEffvsP_fehy3004[13] = {
   0.0001947267,
   7.802294e-05,
   7.479652e-05,
   8.862788e-05,
   0.0001028577,
   0.0001047534,
   0.0001318987,
   0.0001941726,
   0.0003008153,
   0.0004095506,
   0,
   0.003425921,
   0.005622436};
   grae = new TGraphAsymmErrors(13,LQ3D_5trl_PionEffvsP_fx3004,LQ3D_5trl_PionEffvsP_fy3004,LQ3D_5trl_PionEffvsP_felx3004,LQ3D_5trl_PionEffvsP_fehx3004,LQ3D_5trl_PionEffvsP_fely3004,LQ3D_5trl_PionEffvsP_fehy3004);
   grae->SetName("LQ3D_5trl_PionEffvsP");
   grae->SetTitle("");
   grae->SetFillColor(1);
   grae->SetFillStyle(0);

   ci = TColor::GetColor("#ff00ff");
   grae->SetLineColor(ci);
   grae->SetLineWidth(2);

   ci = TColor::GetColor("#ff00ff");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(25);
   grae->SetMarkerSize(1.6);
   
   TH1F *Graph_LQ3D_5trl_PionEffvsP3004 = new TH1F("Graph_LQ3D_5trl_PionEffvsP3004","",100,0,11);
   Graph_LQ3D_5trl_PionEffvsP3004->SetMinimum(4.244039e-05);
   Graph_LQ3D_5trl_PionEffvsP3004->SetMaximum(0.04244039);
   Graph_LQ3D_5trl_PionEffvsP3004->SetDirectory(0);
   Graph_LQ3D_5trl_PionEffvsP3004->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_LQ3D_5trl_PionEffvsP3004->SetLineColor(ci);
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetTitle("#it{p} (GeV/#it{c})");
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetRange(1,37);
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetLabelFont(42);
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetLabelOffset(0.01);
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetLabelSize(0.045);
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetTitleSize(0.045);
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetTickLength(0.02);
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetTitleOffset(1.3);
   Graph_LQ3D_5trl_PionEffvsP3004->GetXaxis()->SetTitleFont(42);
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetTitle("pion efficiency");
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetNdivisions(506);
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetLabelFont(42);
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetLabelOffset(0.01);
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetLabelSize(0.045);
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetTitleSize(0.045);
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetTickLength(0.02);
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetTitleOffset(1.5);
   Graph_LQ3D_5trl_PionEffvsP3004->GetYaxis()->SetTitleFont(42);
   Graph_LQ3D_5trl_PionEffvsP3004->GetZaxis()->SetLabelFont(42);
   Graph_LQ3D_5trl_PionEffvsP3004->GetZaxis()->SetLabelSize(0.035);
   Graph_LQ3D_5trl_PionEffvsP3004->GetZaxis()->SetTitleSize(0.035);
   Graph_LQ3D_5trl_PionEffvsP3004->GetZaxis()->SetTitleOffset(1);
   Graph_LQ3D_5trl_PionEffvsP3004->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_LQ3D_5trl_PionEffvsP3004);
   
   grae->Draw("p");
   
   Double_t LQ7D_5trl_PionEffvsP_fx3005[13] = {
   0.4246237,
   0.5985138,
   0.7952877,
   0.9949122,
   1.195459,
   1.439668,
   1.781109,
   2.220042,
   2.719777,
   0,
   0,
   6.708676,
   8.769783};
   Double_t LQ7D_5trl_PionEffvsP_fy3005[13] = {
   0.002198844,
   0.00169033,
   0.001693018,
   0.002433176,
   0.0027795,
   0.003243855,
   0.004485803,
   0.00581661,
   0.007621758,
   0,
   0,
   0.02690326,
   0.09782609};
   Double_t LQ7D_5trl_PionEffvsP_felx3005[13] = {
   0.1746237,
   0.09851377,
   0.0952877,
   0.09491219,
   0.09545939,
   0.1396684,
   0.1811093,
   0.2200421,
   0.2197765,
   0,
   0,
   0.7086763,
   0.7697825};
   Double_t LQ7D_5trl_PionEffvsP_fely3005[13] = {
   0.0001454883,
   7.771092e-05,
   8.019785e-05,
   0.0001092691,
   0.0001337509,
   0.000140521,
   0.0001858729,
   0.000263229,
   0.0004301765,
   0,
   0,
   0.003696484,
   0.01475164};
   Double_t LQ7D_5trl_PionEffvsP_fehx3005[13] = {
   0.07537628,
   0.1014862,
   0.1047123,
   0.1050878,
   0.1045406,
   0.1603316,
   0.2188907,
   0.2799579,
   0.2802235,
   0,
   0,
   1.291324,
   1.230217};
   Double_t LQ7D_5trl_PionEffvsP_fehy3005[13] = {
   0.0001521873,
   8.016433e-05,
   8.280832e-05,
   0.0001126328,
   0.0001381718,
   0.0001446861,
   0.00019113,
   0.0002713692,
   0.0004468621,
   0,
   0,
   0.004058301,
   0.01620625};
   grae = new TGraphAsymmErrors(13,LQ7D_5trl_PionEffvsP_fx3005,LQ7D_5trl_PionEffvsP_fy3005,LQ7D_5trl_PionEffvsP_felx3005,LQ7D_5trl_PionEffvsP_fehx3005,LQ7D_5trl_PionEffvsP_fely3005,LQ7D_5trl_PionEffvsP_fehy3005);
   grae->SetName("LQ7D_5trl_PionEffvsP");
   grae->SetTitle("");
   grae->SetFillColor(1);
   grae->SetFillStyle(0);

   ci = TColor::GetColor("#0000ff");
   grae->SetLineColor(ci);
   grae->SetLineWidth(2);

   ci = TColor::GetColor("#0000ff");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(26);
   grae->SetMarkerSize(1.7);
   
   TH1F *Graph_LQ7D_5trl_PionEffvsP3005 = new TH1F("Graph_LQ7D_5trl_PionEffvsP3005","",100,0,11);
   Graph_LQ7D_5trl_PionEffvsP3005->SetMinimum(0.0001254356);
   Graph_LQ7D_5trl_PionEffvsP3005->SetMaximum(0.1254356);
   Graph_LQ7D_5trl_PionEffvsP3005->SetDirectory(0);
   Graph_LQ7D_5trl_PionEffvsP3005->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_LQ7D_5trl_PionEffvsP3005->SetLineColor(ci);
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetTitle("#it{p} (GeV/#it{c})");
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetRange(1,37);
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetLabelFont(42);
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetLabelOffset(0.01);
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetLabelSize(0.045);
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetTitleSize(0.045);
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetTickLength(0.02);
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetTitleOffset(1.3);
   Graph_LQ7D_5trl_PionEffvsP3005->GetXaxis()->SetTitleFont(42);
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetTitle("pion efficiency");
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetNdivisions(506);
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetLabelFont(42);
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetLabelOffset(0.01);
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetLabelSize(0.045);
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetTitleSize(0.045);
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetTickLength(0.02);
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetTitleOffset(1.5);
   Graph_LQ7D_5trl_PionEffvsP3005->GetYaxis()->SetTitleFont(42);
   Graph_LQ7D_5trl_PionEffvsP3005->GetZaxis()->SetLabelFont(42);
   Graph_LQ7D_5trl_PionEffvsP3005->GetZaxis()->SetLabelSize(0.035);
   Graph_LQ7D_5trl_PionEffvsP3005->GetZaxis()->SetTitleSize(0.035);
   Graph_LQ7D_5trl_PionEffvsP3005->GetZaxis()->SetTitleOffset(1);
   Graph_LQ7D_5trl_PionEffvsP3005->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_LQ7D_5trl_PionEffvsP3005);
   
   grae->Draw("p");
   
   Double_t p_effvsp_fx3006[11] = {
   0.4350755,
   0.6029244,
   0.7949052,
   0.9947967,
   1.194642,
   1.437239,
   1.776804,
   2.216097,
   2.717597,
   0,
   0};
   Double_t p_effvsp_fy3006[11] = {
   0.001584595,
   0.002067708,
   0.001976303,
   0.002800063,
   0.002948037,
   0.00314751,
   0.003752736,
   0.005697906,
   0.007884761,
   0,
   0};
   Double_t p_effvsp_felx3006[11] = {
   0.1850755,
   0.1029244,
   0.09490524,
   0.09479674,
   0.09464166,
   0.1372393,
   0.1768043,
   0.2160971,
   0.217597,
   0,
   0};
   Double_t p_effvsp_fely3006[11] = {
   0.0001819213,
   0.0001447847,
   0.0001499419,
   0.0001954909,
   0.0002381787,
   0.0002787655,
   0.0003655312,
   0.000538673,
   0.0010679,
   0,
   0};
   Double_t p_effvsp_fehx3006[11] = {
   0.06492454,
   0.09707562,
   0.1050947,
   0.1052033,
   0.1053583,
   0.1627608,
   0.2231956,
   0.2839029,
   0.282403,
   0,
   0};
   Double_t p_effvsp_fehy3006[11] = {
   0.0001819213,
   0.0001447847,
   0.0001499419,
   0.0001954909,
   0.0002381787,
   0.0002787655,
   0.0003655312,
   0.000538673,
   0.0010679,
   0,
   0};
   grae = new TGraphAsymmErrors(11,p_effvsp_fx3006,p_effvsp_fy3006,p_effvsp_felx3006,p_effvsp_fehx3006,p_effvsp_fely3006,p_effvsp_fehy3006);
   grae->SetName("p_effvsp");
   grae->SetTitle("");
   grae->SetFillColor(1);
   grae->SetFillStyle(0);

   ci = TColor::GetColor("#0000ff");
   grae->SetLineColor(ci);
   grae->SetLineWidth(2);

   ci = TColor::GetColor("#0000ff");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(22);
   grae->SetMarkerSize(1.7);
   
   TH1F *Graph_p_effvsp3006 = new TH1F("Graph_p_effvsp3006","",100,0,3.3);
   Graph_p_effvsp3006->SetMinimum(9.847927e-06);
   Graph_p_effvsp3006->SetMaximum(0.009847927);
   Graph_p_effvsp3006->SetDirectory(0);
   Graph_p_effvsp3006->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_p_effvsp3006->SetLineColor(ci);
   Graph_p_effvsp3006->GetXaxis()->SetTitle("#it{p} (GeV/#it{c})");
   Graph_p_effvsp3006->GetXaxis()->SetRange(1,101);
   Graph_p_effvsp3006->GetXaxis()->SetLabelFont(42);
   Graph_p_effvsp3006->GetXaxis()->SetLabelOffset(0.01);
   Graph_p_effvsp3006->GetXaxis()->SetLabelSize(0.045);
   Graph_p_effvsp3006->GetXaxis()->SetTitleSize(0.045);
   Graph_p_effvsp3006->GetXaxis()->SetTickLength(0.02);
   Graph_p_effvsp3006->GetXaxis()->SetTitleOffset(1.3);
   Graph_p_effvsp3006->GetXaxis()->SetTitleFont(42);
   Graph_p_effvsp3006->GetYaxis()->SetTitle("pion efficiency");
   Graph_p_effvsp3006->GetYaxis()->SetNdivisions(506);
   Graph_p_effvsp3006->GetYaxis()->SetLabelFont(42);
   Graph_p_effvsp3006->GetYaxis()->SetLabelOffset(0.01);
   Graph_p_effvsp3006->GetYaxis()->SetLabelSize(0.045);
   Graph_p_effvsp3006->GetYaxis()->SetTitleSize(0.045);
   Graph_p_effvsp3006->GetYaxis()->SetTickLength(0.02);
   Graph_p_effvsp3006->GetYaxis()->SetTitleOffset(1.5);
   Graph_p_effvsp3006->GetYaxis()->SetTitleFont(42);
   Graph_p_effvsp3006->GetZaxis()->SetLabelFont(42);
   Graph_p_effvsp3006->GetZaxis()->SetLabelSize(0.035);
   Graph_p_effvsp3006->GetZaxis()->SetTitleSize(0.035);
   Graph_p_effvsp3006->GetZaxis()->SetTitleOffset(1);
   Graph_p_effvsp3006->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_p_effvsp3006);
   
   grae->Draw("p");
   
   TLegend *leg = new TLegend(0.698492,0.718912,0.959799,0.920984,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextFont(43);
   leg->SetTextSize(29);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(10);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("truncmean_2trl_PionEffvsP","Trunc. mean","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#009900");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(23);
   entry->SetMarkerSize(1.92);
   entry->SetTextFont(43);
   entry=leg->AddEntry("LQ1D_5trl_PionEffvsP","LQ1D","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.6);
   entry->SetTextFont(43);
   entry=leg->AddEntry("LQ2D_2trl_PionEffvsP","LQ2D","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(2);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1.6);
   entry->SetTextFont(43);
   entry=leg->AddEntry("LQ3D_5trl_PionEffvsP","LQ3D","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#ff00ff");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(25);
   entry->SetMarkerSize(1.6);
   entry->SetTextFont(43);
   entry=leg->AddEntry("LQ7D_5trl_PionEffvsP","LQ7D","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#0000ff");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(26);
   entry->SetMarkerSize(1.7);
   entry->SetTextFont(43);
   entry=leg->AddEntry("p_effvsp","NN","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#0000ff");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(22);
   entry->SetMarkerSize(1.7);
   entry->SetTextFont(43);
   leg->Draw();
   TLatex *   tex = new TLatex(0.16075,0.8147753,"6 layers");
tex->SetNDC();
   tex->SetTextFont(43);
   tex->SetTextSize(28.54616);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.16075,0.8551835,"90% electron efficiency");
tex->SetNDC();
   tex->SetTextFont(43);
   tex->SetTextSize(28.54616);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.16075,0.8955918,"ALICE p#font[122]{-}Pb #sqrt{#it{s}_{NN}} = 5.02 TeV");
tex->SetNDC();
   tex->SetTextFont(43);
   tex->SetTextSize(28.54616);
   tex->SetLineWidth(2);
   tex->Draw();
   TLine *line = new TLine(0,0.001,6,0.001);
   line->SetLineWidth(2);
   line->Draw();
   TRDPIDPerformance->Modified();
   TRDPIDPerformance->cd();
   TRDPIDPerformance->SetSelected(TRDPIDPerformance);
}
