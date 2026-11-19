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

// ---------------------------------------------------------------------------
// transformer_ee slice and prong inputs read from vtx.elastic.fuzzyk.
//
// The release helper (3FlavorAna/Vars/TransformerEHelperVars.cxx) reads
// vtx.mlvertex.fuzzyk, which is empty in these prod5.1 CAFs (all png.* columns empty,
// orphCalE nan, remPngCalE == calE). These copies have the same column names, in the
// same order, and the same per-field rules, but read vtx.elastic.fuzzyk instead.
// See docs/CONFIG_PLAN_magMomentAna.md §1.1 and §3.1.
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

// Same rule as the helper: if transformercvnpart is the -5 placeholder, return 0,
// except muonid, which is 1 for prongs longer than 500 cm.
MultiVar constructTransformerCVNPng3dVar(
    const std::function<double(const caf::SRFuzzyKProngProxy *png)> &func, bool isMuonID)
{
  return constructPng3dVar(
      [func, isMuonID](const caf::SRFuzzyKProngProxy *png) -> double
      {
        if (png->transformercvnpart.muonid != -5) return func(png);
        if (isMuonID && (png->len > 500)) return 1.0;
        return 0.0;
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

#define ELASTIC_TFCVN_PNG3D_VAR(EXPR, ACCESSOR, ISMUON)                                    \
  {"png." #EXPR, elastic_tfee::constructTransformerCVNPng3dVar(                            \
                     [](const caf::SRFuzzyKProngProxy *png) -> double                      \
                     { return png->transformercvnpart.ACCESSOR; },                         \
                     ISMUON)}

#define ELASTIC_BPF_PNG3D_VAR(EXPR, IDX)                                                   \
  {"png.bpf[" #IDX "]." #EXPR, elastic_tfee::constructBPFPng3dVar(                         \
                                   [](const caf::SRBpfTrackProxy *bpf) -> double           \
                                   { return bpf->EXPR; },                                  \
                                   IDX)}

// Same six columns, same order, as transformer_cafana::kSliceVarDefs.
const std::vector<std::pair<std::string, Var>> kSliceVarDefsElastic({
    {"coarseTiming", transformer_cafana::kCoarseTiming},
    {"lowGain",      transformer_cafana::kLowGain},
    {"remPngCalE",   elastic_tfee::kRemPngCalEElastic},
    {"nHit",         kNHit},
    {"calE",         SIMPLEVAR(slc.calE)},
    {"orphCalE",     elastic_tfee::kOrphCalEElastic},
});

// Same 44 columns, same order, as transformer_cafana::kPng3dVarDefs.
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

    ELASTIC_TFCVN_PNG3D_VAR(cvnpart.muonid, muonid, true),
    ELASTIC_TFCVN_PNG3D_VAR(cvnpart.electronid, electronid, false),
    ELASTIC_TFCVN_PNG3D_VAR(cvnpart.pionid, pionid, false),
    ELASTIC_TFCVN_PNG3D_VAR(cvnpart.protonid, protonid, false),
    ELASTIC_TFCVN_PNG3D_VAR(cvnpart.photonid, photonid, false),

    ELASTIC_BPF_PNG3D_VAR(energy, 0),
    ELASTIC_BPF_PNG3D_VAR(overlapE, 0),
    ELASTIC_BPF_PNG3D_VAR(momentum.x, 0),
    ELASTIC_BPF_PNG3D_VAR(momentum.y, 0),
    ELASTIC_BPF_PNG3D_VAR(momentum.z, 0),
    ELASTIC_BPF_PNG3D_VAR(nhit, 0),
    ELASTIC_BPF_PNG3D_VAR(nhitx, 0),
    ELASTIC_BPF_PNG3D_VAR(nhity, 0),
    ELASTIC_BPF_PNG3D_VAR(nplane, 0),

    ELASTIC_BPF_PNG3D_VAR(energy, 1),
    ELASTIC_BPF_PNG3D_VAR(overlapE, 1),
    ELASTIC_BPF_PNG3D_VAR(momentum.x, 1),
    ELASTIC_BPF_PNG3D_VAR(momentum.y, 1),
    ELASTIC_BPF_PNG3D_VAR(momentum.z, 1),
    ELASTIC_BPF_PNG3D_VAR(nhit, 1),
    ELASTIC_BPF_PNG3D_VAR(nhitx, 1),
    ELASTIC_BPF_PNG3D_VAR(nhity, 1),
    ELASTIC_BPF_PNG3D_VAR(nplane, 1),

    ELASTIC_BPF_PNG3D_VAR(energy, 2),
    ELASTIC_BPF_PNG3D_VAR(overlapE, 2),
    ELASTIC_BPF_PNG3D_VAR(momentum.x, 2),
    ELASTIC_BPF_PNG3D_VAR(momentum.y, 2),
    ELASTIC_BPF_PNG3D_VAR(momentum.z, 2),
    ELASTIC_BPF_PNG3D_VAR(nhit, 2),
    ELASTIC_BPF_PNG3D_VAR(nhitx, 2),
    ELASTIC_BPF_PNG3D_VAR(nhity, 2),
    ELASTIC_BPF_PNG3D_VAR(nplane, 2),
});

#undef ELASTIC_SIMPLE_PNG3D_VAR
#undef ELASTIC_TFCVN_PNG3D_VAR
#undef ELASTIC_BPF_PNG3D_VAR

void nd_bg_extractor() {
    CSVMaker maker(DATA, "dataset_transformer_ee_nd_bg.csv");
    maker.setPrecision(6);

    maker.addVars(kSliceVarDefsElastic);   // was transformer_cafana::kSliceVarDefs (vtx.mlvertex)

    maker.addVars(TRUTH_VAR_DEFS);
    maker.addVars(RECO_VAR_DEFS);
    maker.addVars(FLAG_VAR_DEFS);
    maker.addVars(EXTRA_VAR_DEFS);
    maker.addVars(BKG_LABEL_VAR_DEFS);   // shared-schema columns (see header)
    maker.addVars(CATEGORY_VAR_DEFS);    // shared-schema columns (see header)

    maker.addMultiVars(kPng3dVarDefsElastic);   // was transformer_cafana::kPng3dVarDefs (vtx.mlvertex)

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
