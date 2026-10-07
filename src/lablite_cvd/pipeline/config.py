"""Static configuration: NHANES cycles, component files, variables and URLs.

Ported from EquiCVD Bench v1.0.0 (https://doi.org/10.5281/zenodo.23083677), with urine albumin/creatinine added.
"""
from pathlib import Path

# (label, start year, file suffix)
CYCLES = [
    ("1999-2000", 1999, ""),
    ("2001-2002", 2001, "_B"),
    ("2003-2004", 2003, "_C"),
    ("2005-2006", 2005, "_D"),
    ("2007-2008", 2007, "_E"),
    ("2009-2010", 2009, "_F"),
    ("2011-2012", 2011, "_G"),
    ("2013-2014", 2013, "_H"),
    ("2015-2016", 2015, "_I"),
    ("2017-2018", 2017, "_J"),
]

# Laboratory components had different file stems before 2005-2006.
LAB_LEGACY = {
    "TCHOL": {1999: "LAB13", 2001: "L13_B", 2003: "L13_C"},
    "HDL": {1999: "LAB13", 2001: "L13_B", 2003: "L13_C"},
    "GHB": {1999: "LAB10", 2001: "L10_B", 2003: "L10_C"},
    "BIOPRO": {1999: "LAB18", 2001: "L40_B", 2003: "L40_C"},
    "ALB_CR": {1999: "LAB16", 2001: "L16_B", 2003: "L16_C"},
}

# component -> variables to keep (only those present in a given file are read)
COMPONENTS = {
    "DEMO": ["RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR",
             "RIDEXPRG", "WTMEC2YR", "WTMEC4YR", "SDMVPSU", "SDMVSTRA"],
    "BPX": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"],
    "BPQ": ["BPQ020", "BPQ050A", "BPQ100D"],
    "BMX": ["BMXBMI"],
    "DIQ": ["DIQ010", "DIQ050", "DIQ070"],
    "SMQ": ["SMQ020", "SMQ040"],
    "MCQ": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    "TCHOL": ["LBXTC"],
    "HDL": ["LBDHDL", "LBXHDD", "LBDHDD"],
    "GHB": ["LBXGH"],
    "BIOPRO": ["LBXSCR", "LBDSCR"],
    "ALB_CR": ["URXUMA", "URXUCR"],
}

NHANES_URLS = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}.xpt",
    "https://wwwn.cdc.gov/Nchs/Nhanes/{label}/{stem}.XPT",
]
MORT_URL = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
            "NHANES_{y0}_{y1}_MORT_2019_PUBLIC.dat")

# NCHS public-use LMF 2019 fixed-width layout (1-based inclusive columns -> 0-based half-open)
MORT_COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19), (42, 45), (45, 48)]
MORT_NAMES = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_INT", "PERMTH_EXM"]
CVD_UCOD = (1, 5)  # 001 diseases of heart, 005 cerebrovascular diseases

RACE_LABELS = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
               4: "NH Black", 5: "Other/Multiracial"}

DEFAULT_DATA_DIR = Path("data")


def file_stem(component: str, year: int, suffix: str) -> str:
    return LAB_LEGACY.get(component, {}).get(year, component + suffix)


def cycle_files(year: int, suffix: str) -> dict:
    """Map file stem -> list of variables needed from it, for one cycle."""
    out: dict = {}
    for comp, cols in COMPONENTS.items():
        out.setdefault(file_stem(comp, year, suffix), []).extend(cols)
    return out
