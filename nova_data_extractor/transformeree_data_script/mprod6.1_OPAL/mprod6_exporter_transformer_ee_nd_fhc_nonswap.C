#include <iostream>
#include <string>

#include "3FlavorAna/Cuts/NumuCuts2024.h"
#include "3FlavorAna/Vars/TransformerEHelperVars.h"
#include "3FlavorAna/Vars/NueEnergy2024.h"
#include "3FlavorAna/Vars/NumuEFxs.h"
#include "3FlavorAna/Vars/NumuVars.h"
#include "CAFAna/Analysis/CSVMaker.h"
#include "CAFAna/Cuts/SpillCuts.h"
#include "CAFAna/Cuts/TruthCuts.h"
#include "CAFAna/Weights/GenieWeights.h"
#include "CAFAna/Weights/PPFXWeights.h"
#include "CAFAna/Weights/XsecTunes.h"
#include "StandardRecord/Proxy/SRProxy.h"

using namespace ana;

const std::string DATA =
    "prod_caf_R24-11-18-miniprod6.1reco.c_nd_genie_AR23_20i_00_000_nonswap_fhc_nova_v08_full_v1_miniprod6-1_miniprod61respinOPAL";

const Cut kTrueEbelow7GeV = kTrueE < 7.0;

const Cut SanityCut(
    [](const caf::SRProxy *sr) {
        return (sr->mc.nnu > 0) && (!sr->mc.nu[0].prim.empty());
    });

const Cut kNumuLoosePID(
    [](const caf::SRProxy *sr) {
        return (
            (sr->sel.remid.pid > 0.5) && (sr->sel.cvnloosepreselptp.numuid > 0.5));
    });

// Original by Alejandro Yankelevich, from NumuCuts2024.cxx. Fix provided by Shaowei Wu.
const Cut kFixedNumuContainND2024(
    [](const caf::SRProxy* sr)
    { if( !sr->vtx.mlvertex.IsValid ) return false;
      // reconstructed showers all contained
      for( unsigned int i = 0; i < sr->vtx.mlvertex.fuzzyk.nshwlid; ++i ) {
	const caf::SRVector3DProxy& start = sr->vtx.mlvertex.fuzzyk.png[i].shwlid.start;
	const caf::SRVector3DProxy& stop  = sr->vtx.mlvertex.fuzzyk.png[i].shwlid.stop;
	if( std::min( start.X(), stop.X() ) < -180.0 ) return false;
	if( std::max( start.X(), stop.X() ) >  180.0 ) return false;
	if( std::min( start.Y(), stop.Y() ) < -180.0 ) return false;
	if( std::max( start.Y(), stop.Y() ) >  180.0 ) return false;
	if( std::min( start.Z(), stop.Z() ) <   40.0 ) return false;
	if( std::max( start.Z(), stop.Z() ) > 1525.0 ) return false;
      }
      
      // only primary muon track present in muon catcher
      if( sr->trk.kalman.ntracks < 1 ) return false;
      for( unsigned int i = 0; i < sr->trk.kalman.ntracks; ++i ) {
	if( i == sr->trk.kalman.idxremid ) continue;
	else if( sr->trk.kalman.tracks[i].start.Z() > 1275 ||
		 sr->trk.kalman.tracks[i].stop.Z()  > 1275 )
	  return false;
      }
      
      return ( sr->trk.kalman.ntracks > sr->trk.kalman.idxremid
	       && sr->slc.firstplane > 1   // skip 0 and 1
	       && sr->slc.lastplane  < 212 // skip 212 and 213
	       && sr->trk.kalman.tracks[0].start.Z() < 1100
	       // vertex definitely outside mC
	       && ( sr->trk.kalman.tracks[0].stop.Z() < 1275
		    || sr->sel.contain.kalyposattrans < 55 ) // air gap
	       && sr->sel.contain.kalfwdcellnd > 5
	       && sr->sel.contain.kalbakcellnd > 10 );
    }
);

const Cut cut =
    kIsNumuCC && (kNumuBasicQuality && kFixedNumuContainND2024 && kNumuLoosePID) && kTrueEbelow7GeV && SanityCut;

const std::vector<std::pair<std::string, Var>> TRUTH_VAR_DEFS({{"mode", SIMPLEVAR(mc.nu[0].mode)},
                                                               {"trueE", SIMPLEVAR(mc.nu[0].E)},
                                                               {"trueLepE", SIMPLEVAR(mc.nu[0].prim[0].p.E)},
                                                               {"trueHadE",
                                                                Var(
                                                                    [](const caf::SRProxy *sr) -> double { return sr->mc.nu[0].E - sr->mc.nu[0].prim[0].p.E; })}});

const std::vector<std::pair<std::string, Var>> RECO_VAR_DEFS({
    {"numuRecoMuonE", kNumuMuE2024},
    {"numuRecoHadE", kNumuHadE2024},
    {"numuRecoE", kNumuE2024},
    {"nueRecoLepE", kEME_2024},
    {"nueRecoHadE", kHADE_2024},
    {"nueRecoE", kNueEnergy2024},
});

const std::vector<std::pair<std::string, Var>> EXTRA_VAR_DEFS({
    {"trkLen", kTrkLength},
    {"remID", kRemID},
    {"cvn.numuid", kCVNm},
    {"cvn.nueid", kCVNe},
    {"cvn.nutauid", kCVNt},
    {"cvn.ncid", kCVNnc},
    {"trkNHit", kTrkNhits},
    {"run", SIMPLEVAR(hdr.run)},
    {"hadCalE", SIMPLEVAR(energy.numu.hadcalE)},
    {"hadTrkE", SIMPLEVAR(energy.numu.hadtrkE)},
});

const Weight weight = kPPFXFluxCVWgt * kXSecCVWgt2020;

void mprod6_exporter_transformer_ee_nd_fhc_nonswap() {
    CSVMaker maker(DATA, "dataset_transformer_ee_nd_fhc_nonswap.csv");
    maker.setPrecision(6);

    maker.addVars(transformer_cafana::kSliceVarDefs);

    maker.addVars(TRUTH_VAR_DEFS);
    maker.addVars(RECO_VAR_DEFS);
    maker.addVars(EXTRA_VAR_DEFS);

    maker.addMultiVars(transformer_cafana::kPng3dVarDefs);

    maker.addVar("weight", VarFromWeight(kPPFXFluxCVWgt));

    maker.SetSpillCut(kStandardSpillCuts);
    maker.setCut(cut);

    maker.Go();
}
