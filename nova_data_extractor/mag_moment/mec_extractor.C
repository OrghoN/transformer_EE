// mec_extractor.C
//
// Writes a CSV sample for transformer_ee from a dedicated ND FHC GENIE nu_e CC MEC
// single-interaction ("singles") sample: prod5.1 reco, release S23-06-22, nonswap,
// SAM definition YiwenXiao_NueCCMEC_Single_NJobs7500_CAF_NonSwap_filematch (owner yxiao).
// It is an optional background sample in the sense of TRAINING_CUTS.md §3. It is enriched
// in nu_e CC MEC, a channel the nominal ND MC (nd_bg_extractor.C) barely populates.
//
// Targets: the model regresses the same two targets as for the signal sample
// (transformer_ee/config/input_nova_magMomentAna.json):
//   trueE     -- true incoming neutrino energy                   (mc.nu[0].E)
//   trueLepE  -- true energy of the primary electron (e-/e+)     (nuone::kTrueElectronE)
// For nu_e CC the primary electron is the outgoing CC lepton, so trueLepE is well defined,
// and the truth cut requires trueLepE > 0 (no -0.5 sentinel), as the signal macro does.
// leadPngTrueE / leadPngTruePdg are kept only as diagnostics: in MEC the leading prong can
// be a knocked-out proton instead of the electron.
//
// Truth: nu_e or nu_e-bar CC, GENIE mode 10 (MEC), not fiducial nu-e elastic scattering,
// with a primary electron. In this sample every true interaction should be of this kind; the
// cut makes it explicit so that an unexpected event is removed instead of mislabelled.
//
// Reco selection: the training cutflow of TRAINING_CUTS.md §4. It is identical to
// enhanced_nu_on_e_extractor.C and nd_bg_extractor.C. Do not change it here alone.
//
// TRIMMED SCHEMA (branch trim-extractor): 53 columns, only those transformer_ee uses;
// see the block above TRUTH_VAR_DEFS. The lead* and weight columns mentioned in this
// header are no longer written.
//
// Columns: enhanced_nu_on_e_extractor.C, nd_bg_extractor.C and mec_extractor.C write ONE
// shared schema (same names, same order), so their CSVs can be merged with merge_csv.sh
// (which concatenates by position and keeps the first file's header). Any column change
// must be made in all three macros at once (docs/CSV_SCHEMA_ALIGNMENT_LOG.md). For this
// reason no xsecWgt or sampleId column is added; MEC rows are mode == 10 && isNueCC == 1.
//
// Normalisation: a singles sample has an arbitrary rate relative to the nominal MC, and
// "weight" (PPFX flux) does not correct for it. Mixed naively with nd_bg, nu_e CC MEC is
// over-represented and double-counted (nd_bg contains natural nu_e CC MEC rows). See
// docs/MEC_FIX_LOG.md before mixing.
//
// Note: CAFAna/Cuts/NuoneCuts.h and NuMagMomentAna/Cuts/*.h both declare names in
// namespace nuone, so they cannot be included together. This macro uses the CAFAna one.

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "3FlavorAna/Vars/TransformerEHelperVars.h"
#include "3FlavorAna/Vars/NueEnergy2024.h"
#include "3FlavorAna/Vars/NumuVars.h"
#include "CAFAna/Analysis/CSVMaker.h"
#include "CAFAna/Cuts/NuoneCuts.h"
#include "CAFAna/Cuts/SpillCuts.h"
#include "CAFAna/Cuts/TruthCuts.h"
#include "CAFAna/Vars/NuoneVars.h"
#include "CAFAna/Vars/Vars.h"
#include "StandardRecord/Proxy/SRProxy.h"

using namespace ana;

const std::string DATA =
    "YiwenXiao_NueCCMEC_Single_NJobs7500_CAF_NonSwap_filematch";

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
// Truth: nu_e / nu_e-bar CC MEC (GENIE mode 10, caf::kMEC in StandardRecord/SREnums.h).
// ana::kHasNu (CAFAna/Cuts/TruthCuts.h, mc.nnu > 0) removes slices with no true neutrino.
// nuone::kNuoneBkg (= !kNuoneSigFid) is the ND nu-on-e group's background definition,
// kept for consistency with nd_bg_extractor.C. kHasTrueElectron makes sure the trueLepE
// target is defined (same cut as enhanced_nu_on_e_extractor.C); kTrueEPositive does the
// same for trueE.
// ---------------------------------------------------------------------------
const Cut kIsMEC([](const caf::SRProxy *sr) {
  return sr->mc.nnu > 0 && sr->mc.nu[0].mode == caf::kMEC;
});
const Cut kHasTrueElectron = nuone::kTrueElectronE > 0;
const Cut kTrueEPositive([](const caf::SRProxy *sr) {
  return sr->mc.nnu > 0 && sr->mc.nu[0].E > 0;
});
const Cut kIsMECTruth =
    kHasNu && nuone::kNuoneBkg && (nuone::kNueCC || nuone::kNuebarCC) && kIsMEC &&
    kHasTrueElectron && kTrueEPositive;

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

const Cut cut = kIsMECTruth && kTrainingSelection;

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
// ---------------------------------------------------------------------------
// TRIMMED SCHEMA (branch trim-extractor): only the columns transformer_ee uses.
// 53 columns = the 40 training columns of config/input_nova_magMomentAna.json (27 prong
// "vector" inputs, 11 "scalar" inputs, 2 targets) + its 14 "eval_columns", with shwCalE
// counted once. Dropped (see docs/EXTRACTOR_TRIM_PLAN.md in the analysis area):
//   truth:  nuPdg, trueElectronTheta, trueEtheta2, trueVtxX/Y/Z
//   reco:   nueRecoHadE
//   flags:  closestSlcTime, longestProng, nPlane, nCell, vtxX/Y/Z, pngMin/MaxX/Y/Z,
//           nuoneid, epi0id, etheta2 (the cuts themselves are unchanged; the pass bits
//           passAnaPresel/passAnaFull are still written)
//   labels: leadPngTrueE/Pdg, leadPngMotherE/Pdg, leadPngEff/Pur
//   other:  cvn.* (never filled), coarseTiming (always 0 in the ND), isCC (= 1 - isNC),
//           png.cvnpart.* (never filled), png.bpf[k].nhit/nhitx/nhity/nplane (copies of
//           png.nhit/...), weight (PPFX flux weight)
// To keep a column, add it back here IN ALL THREE MACROS and to the config if it is an input.
// ---------------------------------------------------------------------------
const std::vector<std::pair<std::string, Var>> TRUTH_VAR_DEFS({
    {"mode",             SIMPLEVAR(mc.nu[0].mode)},  // eval column
    {"trueE",            SIMPLEVAR(mc.nu[0].E)},     // target: true neutrino energy
    {"trueLepE",         nuone::kTrueElectronE},     // target: true electron energy (recoil e for nu-e, CC lepton otherwise; -0.5 if none)
});

// Baseline reco energies for comparison with the transformer (shwCalE is also an input).
const std::vector<std::pair<std::string, Var>> RECO_VAR_DEFS({
    {"shwCalE",     nuone::kShowerE},  // tech-note electron-energy estimator (leading-shower calE)
    {"nueRecoLepE", kEME_2024},
    {"nueRecoE",    kNueEnergy2024},
});

// Inputs taken from the analysis cut variables, plus the cutflow pass bits (eval columns).
const std::vector<std::pair<std::string, Var>> FLAG_VAR_DEFS({
    {"nPng",           nuone::knpng},
    {"shwEFrac",       nuone::kShwEFrac},
    {"vtxE",           nuone::kEvtx},
    {"gap",            nuone::kGap},
    {"theta",          nuone::kMyAngle},
    {"passAnaPresel",  VarFromCut(kAnaPreselection)},
    {"passAnaFull",    VarFromCut(kAnaFull)},
});

// Category bits, as in nuone::nominal_chns (NuoneCuts.h).
const std::vector<std::pair<std::string, Var>> CATEGORY_VAR_DEFS({
    {"isNuoneSig", VarFromCut(nuone::kNuoneSigFid || nuone::kOverlaidNuoneSigFid)},  // mode 5 or overlaid 10005
    {"isNueCC",    VarFromCut(nuone::kNueCC || nuone::kNuebarCC)},
    {"isNumuCC",   VarFromCut(nuone::kNumuCC || nuone::kNumubarCC)},
    {"isNC",       VarFromCut(nuone::kNC)},
});

// Event identifiers (eval columns), to match predictions back to events.
const std::vector<std::pair<std::string, Var>> EXTRA_VAR_DEFS({
    {"run",         SIMPLEVAR(hdr.run)},
    {"subrun",      SIMPLEVAR(hdr.subrun)},
    {"evt",         SIMPLEVAR(hdr.evt)},
    {"subevt",      SIMPLEVAR(hdr.subevt)},
});

// ---------------------------------------------------------------------------
// transformer_ee slice and prong inputs read from vtx.elastic.fuzzyk.
//
// The release helper (3FlavorAna/Vars/TransformerEHelperVars.cxx) reads
// vtx.mlvertex.fuzzyk, which is empty in these prod5.1 CAFs. These copies use the same
// column names and per-field rules, but read vtx.elastic.fuzzyk instead.
// Trimmed: coarseTiming, png.cvnpart.* and png.bpf[k].nhit* are not written.
// KEEP THIS BLOCK IDENTICAL in enhanced_nu_on_e_extractor.C, nd_bg_extractor.C and
// mec_extractor.C.
// ---------------------------------------------------------------------------
namespace elastic_tfee
{

const Var kRemPngCalEElastic(
    [](const caf::SRProxy *sr) -> double
    {
      double prongCalE = 0;
      if (sr->vtx.elastic.IsValid)
      {
        auto &png = sr->vtx.elastic.fuzzyk.png;
        for (size_t i = 0; i < png.size(); i++) prongCalE += png[i].calE;
      }
      return sr->slc.calE - prongCalE;
    });

const Var kOrphCalEElastic(
    [](const caf::SRProxy *sr) -> double
    {
      if (!sr->vtx.elastic.IsValid) return -5.;
      return sr->vtx.elastic.fuzzyk.orphCalE;
    });

MultiVar constructPng3dVar(const std::function<double(const caf::SRFuzzyKProngProxy *png)> &func)
{
  return MultiVar(
      [func](const caf::SRProxy *sr)
      {
        std::vector<double> result;
        if (!sr->vtx.elastic.IsValid) return result;

        auto &png = sr->vtx.elastic.fuzzyk.png;
        result.reserve(png.size());
        for (size_t i = 0; i < png.size(); i++) result.push_back(func(&png[i]));
        return result;
      });
}

// Same rule as the helper: 0 unless all three BPF fits are valid; index 0 = proton,
// 1 = pion, 2 = muon.
MultiVar constructBPFPng3dVar(const std::function<double(const caf::SRBpfTrackProxy *bpf)> &func,
                              size_t bpfIdx)
{
  return constructPng3dVar(
      [func, bpfIdx](const caf::SRFuzzyKProngProxy *png) -> double
      {
        if (!png->bpf.muon.IsValid || !png->bpf.pion.IsValid || !png->bpf.proton.IsValid) return 0;
        switch (bpfIdx)
        {
        case 2: return func(&png->bpf.muon);
        case 1: return func(&png->bpf.pion);
        case 0: return func(&png->bpf.proton);
        default: return 0;
        }
      });
}

} // namespace elastic_tfee

#define ELASTIC_SIMPLE_PNG3D_VAR(EXPR)                                                     \
  {"png." #EXPR, elastic_tfee::constructPng3dVar(                                          \
                     [](const caf::SRFuzzyKProngProxy *png) -> double { return png->EXPR; })}

#define ELASTIC_BPF_PNG3D_VAR(EXPR, IDX)                                                   \
  {"png.bpf[" #IDX "]." #EXPR, elastic_tfee::constructBPFPng3dVar(                         \
                                   [](const caf::SRBpfTrackProxy *bpf) -> double           \
                                   { return bpf->EXPR; },                                  \
                                   IDX)}

// Slice inputs (coarseTiming dropped: always 0 outside the FD).
const std::vector<std::pair<std::string, Var>> kSliceVarDefsElastic({
    {"lowGain",      transformer_cafana::kLowGain},
    {"remPngCalE",   elastic_tfee::kRemPngCalEElastic},
    {"nHit",         kNHit},
    {"calE",         SIMPLEVAR(slc.calE)},
    {"orphCalE",     elastic_tfee::kOrphCalEElastic},
});

// The 27 prong inputs of the config.
const std::vector<std::pair<std::string, MultiVar>> kPng3dVarDefsElastic({
    ELASTIC_SIMPLE_PNG3D_VAR(dir.x),
    ELASTIC_SIMPLE_PNG3D_VAR(dir.y),
    ELASTIC_SIMPLE_PNG3D_VAR(dir.z),
    ELASTIC_SIMPLE_PNG3D_VAR(start.x),
    ELASTIC_SIMPLE_PNG3D_VAR(start.y),
    ELASTIC_SIMPLE_PNG3D_VAR(start.z),
    ELASTIC_SIMPLE_PNG3D_VAR(len),
    ELASTIC_SIMPLE_PNG3D_VAR(calE),
    ELASTIC_SIMPLE_PNG3D_VAR(nhit),
    ELASTIC_SIMPLE_PNG3D_VAR(nhitx),
    ELASTIC_SIMPLE_PNG3D_VAR(nhity),
    ELASTIC_SIMPLE_PNG3D_VAR(nplane),

    ELASTIC_BPF_PNG3D_VAR(energy, 0),
    ELASTIC_BPF_PNG3D_VAR(overlapE, 0),
    ELASTIC_BPF_PNG3D_VAR(momentum.x, 0),
    ELASTIC_BPF_PNG3D_VAR(momentum.y, 0),
    ELASTIC_BPF_PNG3D_VAR(momentum.z, 0),

    ELASTIC_BPF_PNG3D_VAR(energy, 1),
    ELASTIC_BPF_PNG3D_VAR(overlapE, 1),
    ELASTIC_BPF_PNG3D_VAR(momentum.x, 1),
    ELASTIC_BPF_PNG3D_VAR(momentum.y, 1),
    ELASTIC_BPF_PNG3D_VAR(momentum.z, 1),

    ELASTIC_BPF_PNG3D_VAR(energy, 2),
    ELASTIC_BPF_PNG3D_VAR(overlapE, 2),
    ELASTIC_BPF_PNG3D_VAR(momentum.x, 2),
    ELASTIC_BPF_PNG3D_VAR(momentum.y, 2),
    ELASTIC_BPF_PNG3D_VAR(momentum.z, 2),
});

#undef ELASTIC_SIMPLE_PNG3D_VAR
#undef ELASTIC_BPF_PNG3D_VAR

void mec_extractor() {
    CSVMaker maker(DATA, "dataset_transformer_ee_mec.csv");
    // 6 significant digits are exact for every integer column of the trimmed schema (run,
    // subrun, evt, subevt, mode and flags are all < 1e6) and ample for the float inputs.
    // Raise to 10 before adding a column with larger integers (e.g. nuclear PDG codes,
    // which 6 digits turned into 1.00001e+09).
    maker.setPrecision(6);

    maker.addVars(kSliceVarDefsElastic);   // was transformer_cafana::kSliceVarDefs (vtx.mlvertex)

    maker.addVars(TRUTH_VAR_DEFS);
    maker.addVars(RECO_VAR_DEFS);
    maker.addVars(FLAG_VAR_DEFS);
    maker.addVars(EXTRA_VAR_DEFS);
    maker.addVars(CATEGORY_VAR_DEFS);    // shared-schema columns (see header)

    maker.addMultiVars(kPng3dVarDefsElastic);   // was transformer_cafana::kPng3dVarDefs (vtx.mlvertex)

    // No "weight" column in the trimmed schema (PPFX flux weight; not used in training).

    maker.SetSpillCut(kStandardSpillCuts);
    maker.setCut(cut);

    maker.Go();
}
