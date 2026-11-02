"""This machine's registered datasets (paths, display ranges, scan
shapes).  Kept in the private repository only: it is NOT shipped in
the pip / PyPI package, where data.SAMPLES starts empty and datasets
are loaded from the GUI."""
import os

LOCAL_DATA_ROOT = r"D:\DINOSR\data"


SAMPLES = {
    "Na007a": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na007a.prz"),
        "vmax": 2, "scan_shape": (126, 100), "center_mask_radius": 15,
        "approved_label": None,
    },
    "Na007b": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na007b.prz"),
        "vmax": 2, "scan_shape": (126, 100), "center_mask_radius": 15,
        "approved_label": "heat_040_050",
    },
    "Na006a": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na006a.prz"),
        "vmax": 4, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": "heat_040_070",
    },
    "EuInAs_B100": {
        "path": os.path.join(LOCAL_DATA_ROOT, "EuInAs_B100.prz"),
        "vmax": 30, "scan_shape": (66, 396), "center_mask_radius": 10,
        "approved_label": "heat_040_070",
    },
    "IMC_50nm_SI2": {
        "path": os.path.join(LOCAL_DATA_ROOT, "IMC_50nm_SI2.prz"),
        "vmax": 3, "scan_shape": (128, 128), "center_mask_radius": 15,
        "approved_label": "const_065",
    },
    "IMC_150nm_SI5": {
        "path": os.path.join(LOCAL_DATA_ROOT, "IMC_150nm_SI5.prz"),
        "vmax": 5, "scan_shape": (128, 128), "center_mask_radius": 15,
        "approved_label": "const_040",
    },
    "PLA": {
        # Polylactic acid thin film (NBED-005 dataset from Nadav).
        # Copied from X:\Nadav\251029-NYahalom-PLA-no-ann\...\4D-STEM_Single_Image.prz.
        # Cube shape (128, 128, 256, 256) float32. Intensity distribution per
        # pattern: mean ~0.67, p95 ~1.0, p99 ~2.2, per-pattern max 9k-19k
        # (hot center pixel). vmax=3 chosen to match IMC_50nm scale (same
        # detector family, same organic / amorphous-carbon support).
        "path": os.path.join(LOCAL_DATA_ROOT, "PLA.prz"),
        "vmax": 3, "scan_shape": (128, 128), "center_mask_radius": 15,
        "approved_label": None,
    },
}

# ── 240110-MgNaphi (MgNaPHI at 115mm CL, locally copied) ──────────────────────
# All entries: 115mm CL, 100x100 scan, focused probe, 3.55nm step size (115k mag).
# Path: D:\DINOSR\data\240110-MgNaphi\SI-xxx\DataCube_0.prz (PRZ size 4.9 GB each).
_MG_240110_ROOT = r"D:\DINOSR\data\240110-MgNaphi"
# Focused-probe, 115mm CL, 115k mag SIs (the "good" measurements for training).
# SI-002 (115k mag, 10eV),
# SI-005 (115k mag, 7eV slit, first good after grid change),
# SI-006 (115k mag, 10eV),
# SI-008 (115k mag, 10eV, same position as SI-007 different mag),
# SI-009 (115k mag, 10eV, same position as SI-008 different ROI),
# SI-020 (115k mag, 10eV, zoomed into thin flake),
# SI-022 (115k mag, 10eV, very thin flake, same flake as SI-021).
for _si in (2, 5, 6, 8, 9, 20, 22):
    SAMPLES[f"MgNaPHI240110_SI{_si:03d}"] = {
        "path": os.path.join(_MG_240110_ROOT, f"SI-{_si:03d}", "DataCube_0.prz"),
        "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": None,
    }
# Defocused-probe 115mm CL SIs (skipped for now; different physics).
# SI-010/011/012/013 = 4 quarters of same ROI at 20k mag with -12um defocused probe.


# ── ROI-split experiment samples ──────────────────────────────────────────────
# 240312-NaPHI-Nadja-remeasure  (58mm CL, 115k mag, focused probe, 3.55nm step)
# README: "IN SI-005 TO SI-008 SCANNED THE WHOLE ROI WITH 4 QUARTERS"
# → SI-005/006/007/008 are 4 quarters of the SAME physical ROI.
_NA_REMEAS_ROOT = r"D:\DINOSR\data\240312-NaPHI-Nadja-remeasure\EF-4DSTEM"
for _si in range(1, 11):  # SI-001 through SI-010
    SAMPLES[f"NaPHI_Nadja_SI{_si:03d}"] = {
        "path": os.path.join(_NA_REMEAS_ROOT, f"SI-{_si:03d}", "DataCube_0.prz"),
        "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": None,
    }
# Combined training sets (paths stored as list; LoadPRZMulti handles)
SAMPLES["NaPHI_Nadja_4Q_SI5"] = {
    "paths": [os.path.join(_NA_REMEAS_ROOT, "SI-005", "DataCube_0.prz")],
    "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
    "approved_label": None, "is_multi": True,
}
SAMPLES["NaPHI_Nadja_4Q_all4"] = {
    "paths": [os.path.join(_NA_REMEAS_ROOT, f"SI-{i:03d}", "DataCube_0.prz")
              for i in (5, 6, 7, 8)],
    "vmax": 2, "scan_shape": (200, 200),  # nominal; not used for multi
    "center_mask_radius": 15, "approved_label": None, "is_multi": True,
}

# 240312-MgNaPHI-remeasure  (58mm CL, 115k mag, focused probe, 3.55nm step)
# README: "SI-003 to SI-006 I SCANNED ACROSS THE FLAKE img 1217"
# The per-SI file name is irregular in this session:
#   SI-001: DataCube_0.prz
#   SI-003..SI-007, SI-010, SI-011: Survey_CH2_1.prz
#   SI-008: Survey_CH2_0_1.prz
#   SI-009: Survey_CH2_0_0_1.prz
#   SI-012: Survey_CH2_0_0_0_1.prz
# Each is a (100, 100, 512, 512) float32 cube.
_MG_REMEAS_ROOT = r"D:\DINOSR\data\240312-MgNaPHI-remeasure\EF-4DSTEM"
_MG_FILE_PER_SI = {
    1: "DataCube_0.prz",
    2: "DataCube_2_1.prz",
    3: "Survey_CH2_1.prz", 4: "Survey_CH2_1.prz",
    5: "Survey_CH2_1.prz", 6: "Survey_CH2_1.prz",
    7: "Survey_CH2_1.prz",
    8: "Survey_CH2_0_1.prz",
    9: "Survey_CH2_0_0_1.prz",
    10: "Survey_CH2_1.prz", 11: "Survey_CH2_1.prz",
    12: "Survey_CH2_0_0_0_1.prz",
}
for _si, _fn in _MG_FILE_PER_SI.items():
    SAMPLES[f"MgNaPHI_remeas_SI{_si:03d}"] = {
        "path": os.path.join(_MG_REMEAS_ROOT, f"SI-{_si:03d}", _fn),
        "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": None,
    }
SAMPLES["MgNaPHI_remeas_4Q_SI3"] = {
    "paths": [os.path.join(_MG_REMEAS_ROOT, "SI-003", _MG_FILE_PER_SI[3])],
    "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
    "approved_label": None, "is_multi": True,
}
SAMPLES["MgNaPHI_remeas_4Q_all4"] = {
    "paths": [os.path.join(_MG_REMEAS_ROOT, f"SI-{i:03d}", _MG_FILE_PER_SI[i])
              for i in (3, 4, 5, 6)],
    "vmax": 2, "scan_shape": (200, 200),
    "center_mask_radius": 15, "approved_label": None, "is_multi": True,
}

# 240521-MgPhi1-remeasure-tilt-holeyC -- 58mm CL, 10k mag, 100x100 scan,
# tilt series. Each NBED-XXX{a,b,c} is the same particle at three different
# tilt angles. README key:
#   001a tilt=-35deg (out of plane)   001b tilt=+55deg (in-plane-ish)
#   001c tilt=  0deg (in-plane-ish)   002a tilt=-45deg
#   002b tilt=  0deg (lines)          002c tilt=+35deg
# Cube file in each subfolder is DataCube_0.prz (10 GB).
_TILT_HOLEYC_ROOT = r"D:\DINOSR\data\240521-MgPhi1-remeasure-tilt-holeyC"
for _grp in (1, 2):
    for _let in ("a", "b", "c"):
        SAMPLES[f"MgPhi_tilt_NBED{_grp:03d}{_let}"] = {
            "path": os.path.join(_TILT_HOLEYC_ROOT,
                                  f"NBED-{_grp:03d}{_let}",
                                  "DataCube_0.prz"),
            "vmax": 2, "scan_shape": (100, 100),
            "center_mask_radius": 15, "approved_label": None,
        }
