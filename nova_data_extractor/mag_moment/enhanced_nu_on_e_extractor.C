// enhanced_nu_on_e_extractor.C
//
// Writes a CSV training sample for transformer_ee from the ND enhanced nu-on-e
// overlay MC. The model regresses TWO targets:
//   trueE     -- true incoming neutrino energy           (mc.nu[0].E)
//   trueLepE  -- true recoil-electron energy             (nuone::kTrueElectronE)
// The column names match the "target" list in
// transformer_ee/config/input_nova_magMomentAna_nd_fhc.json.
//
// The selection is the *training* cutflow of TRAINING_CUTS.md §4. It is a loosened
// version of the sarahTechNote.pdf Table 12 analysis cutflow. Cut objects come from
// CAFAna/Cuts/NuoneCuts.h (the ND nu-on-e group's definitions) where they exist.
// Cuts that the training selection drops or loosens are written out as "flag"
// columns, so the analysis cutflow can still be applied at evaluation time
// (TRAINING_CUTS.md §5).
//
// Note: CAFAna/Cuts/NuoneCuts.h and NuMagMomentAna/Cuts/*.h both declare names in
// namespace nuone, so they cannot be included together. This macro uses the CAFAna one.

#include <iostream>
#include <string>

#include "3FlavorAna/Vars/TransformerEHelperVars.h"
#include "3FlavorAna/Vars/NueEnergy2024.h"
#include "3FlavorAna/Vars/NumuVars.h"
#include "CAFAna/Analysis/CSVMaker.h"
#include "CAFAna/Cuts/NuoneCuts.h"
#include "CAFAna/Cuts/SpillCuts.h"
#include "CAFAna/Cuts/TruthCuts.h"
#include "CAFAna/Vars/NuoneVars.h"
#include "CAFAna/Weights/PPFXWeights.h"
#include "StandardRecord/Proxy/SRProxy.h"

using namespace ana;

const std::string DATA =
    "prod_caf_R20-11-25-prod5.1reco.g_nd_genie_N1810j0211a_nonswap_fhc_nova_v08_full_v1_nuone_overlay";

// ---------------------------------------------------------------------------
// Training-cut thresholds (TRAINING_CUTS.md §4). Analysis (Table 12) values are in
// the comments. All are proposed starting points, not validated values.
// ---------------------------------------------------------------------------
const double kTrainLongestProngMax = 800.;  // ana: 600 cm   (= ND-group value)
const int    kTrainNPlaneMax       = 120;   // ana: 95       (= ND-group value)
const int    kTrainNCellMax        = 600;   // ana: 200      (= ND-group value)
const double kTrainShwEFracMin     = 0.5;   // ana: 0.8      (lower edge of the §4 validation region)
const double kTrainVtxEMax         = 0.05;  // ana: 0.016 GeV
const double kTrainGapMax          = 20.;   // ana: 10 cm    (= ND-group value)
const double kTrainShowerEMin      = 0.1;   // ana: 0.3 GeV
const double kTrainShowerEMax      = 10.;   // ana: 5 GeV
// nuoneid, epi0id, eth2: no training cut (stored as flag columns).

// ---------------------------------------------------------------------------
// Truth: nu-e elastic scattering (TRAINING_CUTS.md §3), using the ND nu-on-e signal
// definitions: mode 5 with a primary electron (nominal component), or mode 10005
// (overlaid nu-on-e component), with the true vertex in the fiducial volume.
// ---------------------------------------------------------------------------
const Cut kIsNuElasticTruth = nuone::kNuoneSigFid || nuone::kOverlaidNuoneSigFid;

// The electron-energy target must be defined: a true neutrino with a primary e-/e+.
const Cut kHasTrueElectron = nuone::kTrueElectronE > 0;

// ---------------------------------------------------------------------------
// Reco training selection
// ---------------------------------------------------------------------------
const Cut kVtxIsValid([](const caf::SRProxy *sr) { return bool(sr->vtx.elastic.IsValid); });
const Cut kHasPng = nuone::knpng > 0;

const Cut kTrainLongestProng = nuone::kfLongestProng >= 0 && nuone::kfLongestProng < kTrainLongestProngMax;
const Cut kTrainNPlane       = nuone::kNPlane < kTrainNPlaneMax;
const Cut kTrainNCell        = nuone::kNCell < kTrainNCellMax;
const Cut kTrainShwEFrac     = nuone::kShwEFrac > kTrainShwEFracMin;
const Cut kTrainVtxE         = nuone::kEvtx < kTrainVtxEMax;
const Cut kTrainGap          = nuone::kGap < kTrainGapMax;
const Cut kTrainShowerE      = nuone::kShowerE > kTrainShowerEMin && nuone::kShowerE < kTrainShowerEMax;

const Cut kTrainingSelection =
    kVtxIsValid && kHasPng &&
    nuone::ktimeCut &&                                   // closestSlc: keep (as analysis)
    kTrainLongestProng && kTrainNPlane && kTrainNCell &&
    nuone::knuoneFiducial &&                             // fiducial: keep, Table 7 box
    nuone::kcontainment &&                               // containment: keep, Table 7 box
    kTrainShwEFrac && kTrainVtxE && kTrainGap &&
    kTrainShowerE;

const Cut cut = kIsNuElasticTruth && kHasTrueElectron && kTrainingSelection;

// ---------------------------------------------------------------------------
// Analysis cutflow (sarahTechNote.pdf Table 12), written as pass/fail bits for
// evaluation. Built from the same Vars as above with the tech-note values. The
// library's nuone::kPreselection uses the older ND-group values, so it is not used.
// ---------------------------------------------------------------------------
const Cut kAnaPreselection =
    kVtxIsValid && kHasPng && nuone::ktimeCut &&
    nuone::kfLongestProng >= 0 && nuone::kfLongestProng < 600 &&
    nuone::kNPlane < 95 && nuone::kNCell < 200 &&
    nuone::knuoneFiducial && nuone::kcontainment &&
    nuone::kShwEFrac > 0.8 && nuone::kEvtx < 0.016 && nuone::kGap < 10 &&
    nuone::kShowerE > 0.3 && nuone::kShowerE < 5.0;
const Cut kAnaCNN       = nuone::knuoneid && nuone::kepi0id;  // > 0.73, > 0.92
const Cut kAnaFull      = kAnaPreselection && kAnaCNN && nuone::ketheta2;  // eth2 < 0.005

// ---------------------------------------------------------------------------
// Output columns
// ---------------------------------------------------------------------------
const Var kTrueElectronETheta2 = nuone::kTrueElectronE * nuone::kTrueElectronAngle * nuone::kTrueElectronAngle;

const std::vector<std::pair<std::string, Var>> TRUTH_VAR_DEFS({
    {"mode",             SIMPLEVAR(mc.nu[0].mode)},
    {"nuPdg",            SIMPLEVAR(mc.nu[0].pdg)},
    {"trueE",            SIMPLEVAR(mc.nu[0].E)},     // target 1: true neutrino energy
    {"trueLepE",         nuone::kTrueElectronE},     // target 2: true recoil-electron energy
    {"trueElectronTheta", nuone::kTrueElectronAngle},
    {"trueEtheta2",      kTrueElectronETheta2},
    {"trueVtxX",         nuone::kNuVtxX},
    {"trueVtxY",         nuone::kNuVtxY},
    {"trueVtxZ",         nuone::kNuVtxZ},
});

// Baseline reco energies for comparison with the transformer.
const std::vector<std::pair<std::string, Var>> RECO_VAR_DEFS({
    {"shwCalE",     nuone::kShowerE},  // tech-note electron-energy estimator (leading-shower calE)
    {"nueRecoLepE", kEME_2024},
    {"nueRecoHadE", kHADE_2024},
    {"nueRecoE",    kNueEnergy2024},
});

// Flag columns: every cut variable of the analysis cutflow, plus pass bits.
const std::vector<std::pair<std::string, Var>> FLAG_VAR_DEFS({
    {"closestSlcTime", nuone::kslcClosestSliceTime},
    {"nPng",           nuone::knpng},
    {"longestProng",   nuone::kfLongestProng},
    {"nPlane",         nuone::kNPlane},
    {"nCell",          nuone::kNCell},
    {"vtxX",           nuone::kVtxX},
    {"vtxY",           nuone::kVtxY},
    {"vtxZ",           nuone::kVtxZ},
    {"pngMinX",        nuone::kMinX},
    {"pngMaxX",        nuone::kMaxX},
    {"pngMinY",        nuone::kMinY},
    {"pngMaxY",        nuone::kMaxY},
    {"pngMinZ",        nuone::kMinZ},
    {"pngMaxZ",        nuone::kMaxZ},
    {"shwEFrac",       nuone::kShwEFrac},
    {"vtxE",           nuone::kEvtx},
    {"gap",            nuone::kGap},
    {"nuoneid",        nuone::kNuoneID},
    {"epi0id",         nuone::kEpi0ID},
    {"etheta2",        nuone::kEtheta2},
    {"theta",          nuone::kMyAngle},
    {"passAnaPresel",  VarFromCut(kAnaPreselection)},
    {"passAnaFull",    VarFromCut(kAnaFull)},
});

const std::vector<std::pair<std::string, Var>> EXTRA_VAR_DEFS({
    {"cvn.numuid",  kCVNm},
    {"cvn.nueid",   kCVNe},
    {"cvn.nutauid", kCVNt},
    {"cvn.ncid",    kCVNnc},
    {"run",         SIMPLEVAR(hdr.run)},
    {"subrun",      SIMPLEVAR(hdr.subrun)},
    {"evt",         SIMPLEVAR(hdr.evt)},
    {"subevt",      SIMPLEVAR(hdr.subevt)},
});

void enhanced_nu_on_e_extractor() {
    CSVMaker maker(DATA, "dataset_transformer_ee_enhanced_nu_on_e.csv");
    maker.setPrecision(6);

    maker.addVars(transformer_cafana::kSliceVarDefs);

    maker.addVars(TRUTH_VAR_DEFS);
    maker.addVars(RECO_VAR_DEFS);
    maker.addVars(FLAG_VAR_DEFS);
    maker.addVars(EXTRA_VAR_DEFS);

    maker.addMultiVars(transformer_cafana::kPng3dVarDefs);

    // Physics weights, kept separate from any training weight (TRAINING_CUTS.md §3).
    // "weight" is unchanged from the previous macro (flux only).
    maker.addVar("weight",  VarFromWeight(kPPFXFluxCVWgt));
    // No "xsecWgt" column: kXSecCVWgt2024 (NOvARwgt) throws std::out_of_range on overlaid
    // nu-on-e events (mode 10005, non-GENIE generator), which are all of this sample.
    // See EMPTY_CSV_DIAGNOSIS.md.

    maker.SetSpillCut(kStandardSpillCuts);
    maker.setCut(cut);

    maker.Go();
}
