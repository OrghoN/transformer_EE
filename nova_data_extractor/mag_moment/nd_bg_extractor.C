// nd_bg_extractor.C
//
// Writes a CSV *background* sample for transformer_ee from the nominal ND FHC GENIE MC
// (the full .a CAF definition, not the nuone_overlay signal sample and not the
// nuone_bkg2024 concat, which applies tighter ND-group preselection cuts; see
// docs/ND_BG_EMPTY_DIAGNOSIS.md §5).
//
// Truth selection: a true neutrino interaction that is NOT fiducial nu-e elastic
// scattering (nuone::kNuoneBkg, the ND nu-on-e group's background definition), i.e.
// mainly NC pi0, numu CC with a pi0, and nue CC. Slices with no true neutrino are
// dropped because they have no label.
//
// Labels: for background, trueLepE (nuone::kTrueElectronE) is the -0.5 sentinel unless
// there is a primary e+-, so the truth match of the leading prong (lead*), i.e. the
// particle that produced the leading shower (TRAINING_CUTS.md §3), and category bits
// (isNuoneSig/isNueCC/isNumuCC/isNC/isCC, as in nuone::nominal_chns) are also written.
//
// Columns: enhanced_nu_on_e_extractor.C, nd_bg_extractor.C and mec_extractor.C write ONE
// shared schema (same names, same order), so their CSVs can be merged with merge_csv.sh,
// which concatenates by position and keeps the first file's header. CSVMaker writes all
// Vars in add order, then all MultiVars. Any column change must be made in all three
// macros at once. See docs/CSV_SCHEMA_ALIGNMENT_LOG.md.
//
// The reco selection is the *training* cutflow of TRAINING_CUTS.md §4, identical to the
// signal macro. Cuts it loosens or drops are written as flag columns (TRAINING_CUTS.md §5).
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
    "prod_caf_R20-11-25-prod5.1reco.a_nd_genie_N1810j0211a_nonswap_fhc_nova_v08_full_v1";

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
// Truth: background = a true neutrino interaction that is not fiducial nu-e elastic
// scattering. nuone::kNuoneBkg (= !kNuoneSigFid) is the ND nu-on-e group's definition,
// used for the nuone_bkg2024 concats. kOverlaidNuoneSigFid (mode 10005) is not used: it
// only exists in the overlay sample. No primary-electron requirement: that would keep
// only nue CC and drop the NC / numu CC pi0 backgrounds.
// ana::kHasNu (CAFAna/Cuts/TruthCuts.h, mc.nnu > 0) removes slices with no true
// neutrino (noise, pile-up fragments), which pass kNuoneBkg but have no label.
// ---------------------------------------------------------------------------
const Cut kIsBkgTruth = kHasNu && nuone::kNuoneBkg;

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

const Cut cut = kIsBkgTruth && kTrainingSelection;

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

// ---------------------------------------------------------------------------
// Background labels: truth match of the leading prong, the same prong nuone::kShowerE
// uses (vtx.elastic.fuzzyk.png[0]). -5 = no valid vertex / no prong.
// ---------------------------------------------------------------------------
#define LEAD_PNG_TRUTH_VAR(NAME, EXPR)                                          \
  const Var NAME([](const caf::SRProxy *sr) -> float {                         \
    if (!sr->vtx.elastic.IsValid || sr->vtx.elastic.fuzzyk.npng < 1) return -5.f; \
    return float(sr->vtx.elastic.fuzzyk.png[0].truth.EXPR);                     \
  });
LEAD_PNG_TRUTH_VAR(kLeadPngTrueE,     p.E)
LEAD_PNG_TRUTH_VAR(kLeadPngTruePdg,   pdg)
LEAD_PNG_TRUTH_VAR(kLeadPngMotherE,   motherp.E)   // eve mother, e.g. the pi0 of a photon shower
LEAD_PNG_TRUTH_VAR(kLeadPngMotherPdg, motherpdg)
LEAD_PNG_TRUTH_VAR(kLeadPngEff,       eff)
LEAD_PNG_TRUTH_VAR(kLeadPngPur,       pur)
#undef LEAD_PNG_TRUTH_VAR

const std::vector<std::pair<std::string, Var>> BKG_LABEL_VAR_DEFS({
    {"leadPngTrueE",     kLeadPngTrueE},
    {"leadPngTruePdg",   kLeadPngTruePdg},
    {"leadPngMotherE",   kLeadPngMotherE},
    {"leadPngMotherPdg", kLeadPngMotherPdg},
    {"leadPngEff",       kLeadPngEff},
    {"leadPngPur",       kLeadPngPur},
});

// Category bits, as in nuone::nominal_chns (NuoneCuts.h).
const std::vector<std::pair<std::string, Var>> CATEGORY_VAR_DEFS({
    {"isNuoneSig", VarFromCut(nuone::kNuoneSigFid || nuone::kOverlaidNuoneSigFid)},  // mode 5 or overlaid 10005
    {"isNueCC",    VarFromCut(nuone::kNueCC || nuone::kNuebarCC)},
    {"isNumuCC",   VarFromCut(nuone::kNumuCC || nuone::kNumubarCC)},
    {"isNC",       VarFromCut(nuone::kNC)},
    {"isCC",       SIMPLEVAR(mc.nu[0].iscc)},
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

void nd_bg_extractor() {
    CSVMaker maker(DATA, "dataset_transformer_ee_nd_bg.csv");
    maker.setPrecision(6);

    maker.addVars(transformer_cafana::kSliceVarDefs);

    maker.addVars(TRUTH_VAR_DEFS);
    maker.addVars(RECO_VAR_DEFS);
    maker.addVars(FLAG_VAR_DEFS);
    maker.addVars(EXTRA_VAR_DEFS);
    maker.addVars(BKG_LABEL_VAR_DEFS);   // shared-schema columns (see header)
    maker.addVars(CATEGORY_VAR_DEFS);    // shared-schema columns (see header)

    maker.addMultiVars(transformer_cafana::kPng3dVarDefs);

    // Physics weights, kept separate from any training weight (TRAINING_CUTS.md §3).
    // "weight" is unchanged from the previous macro (flux only).
    maker.addVar("weight",  VarFromWeight(kPPFXFluxCVWgt));
    // No "xsecWgt" column, to keep the schema of the signal macro, where
    // kXSecCVWgt2024 (NOvARwgt) throws on overlaid mode-10005 events
    // (old/EMPTY_CSV_DIAGNOSIS.md). This sample is plain GENIE, so it could be added
    // here (ND_BG_EMPTY_DIAGNOSIS.md, suggested fix 4); not done yet.

    maker.SetSpillCut(kStandardSpillCuts);
    maker.setCut(cut);

    maker.Go();
}
