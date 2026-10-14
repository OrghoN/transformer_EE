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
    "prod_caf_R24-11-18-miniprod6.1reco.b_fd_genie_AR23_20i_00_000_nonswap_fhc_nova_v08_full_v1_miniprod6-1_miniprod61respinOPAL";

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

const Cut cut =
    kIsNumuCC && (kNumuBasicQuality && kNumuContainFD2024 && kNumuLoosePID) && kTrueEbelow7GeV && SanityCut;

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

void mprod6_exporter_transformer_ee_fd_fhc_nonswap() {
    CSVMaker maker(DATA, "dataset_transformer_ee_fd_fhc_nonswap.csv");
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
