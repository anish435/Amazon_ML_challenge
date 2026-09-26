"""
Business Entity Resolution — Preprocessing & Normalization Module
Amazon ML Challenge 2026

This module provides robust, memory-efficient text normalization functions for
business names, business addresses, and country labels.

Key Design Principles:
1. Multilingual Support: Uses Unicode NFKC normalization and safely strips Latin-only
   diacritics while preserving non-Latin scripts (Devanagari, Tamil, Kannada, etc.)
   and their critical combining vowels/matras intact.
2. Information Preservation: Retains all numeric tokens (building numbers, PIN codes,
   suite/apartment numbers) so that e.g. "12 MG Road" and "120 MG Road" remain distinct.
3. Multiple Representations: Generates normalized name, core name (legal suffix stripped),
   normalized address, token lists, and extracted address numbers.
4. Open-Set Country: Handles arbitrary country strings (US, India, France, etc.) without
   hard-coding or filtering.
5. Null/Missing Safety: Never crashes on null, empty, or unexpected values. Missing addresses
   evaluate to None, avoiding accidental false-positive matches.
"""

import re
import unicodedata
from typing import Optional, List, Dict, Any
import polars as pl

# ---------------------------------------------------------------------------
# Precomputed Lookup Tables & Regex Patterns
# ---------------------------------------------------------------------------

# Domain/URL cleanup patterns
URL_PREFIX_PATTERN = re.compile(r'^(?:https?:\/\/)?(?:www\.)?', re.IGNORECASE)
DOMAIN_SUFFIX_PATTERN = re.compile(
    r'\.(com|org|net|co|in|co\.in|fr|io|gov|edu|biz|info|de|uk|eu)\b',
    re.IGNORECASE
)

# Ampersands and plus signs
AMP_PATTERN = re.compile(r'(?:&amp;|&|\+)', re.IGNORECASE)

# Fast Unicode Punctuation & Symbol translation table (excluding combining marks)
# Precomputed once across the Basic Multilingual Plane (BMP)
_PUNCT_TABLE: Dict[int, str] = {}
for _code in range(0x10000):
    _char = chr(_code)
    _cat = unicodedata.category(_char)
    # P = Punctuation, S = Symbol
    if _cat.startswith(('P', 'S')):
        _PUNCT_TABLE[_code] = ' '

# Controlled list of legal entity suffixes and abbreviations
# Ordered by length descending so longer compound phrases match first
LEGAL_SUFFIXES: List[str] = [
    # Indian compound
    "private limited company",
    "private limited",
    "pvt limited",
    "private ltd",
    "pvt ltd",
    "p limited",
    "p ltd",
    "pvtltd",
    "limited liability partnership",
    "one person company",
    # Indic script legal forms
    "प्राइवेट लिमिटेड",
    "लिमिटेड",
    "एलएलपी",
    "பிரைவேட் லிமிடெட்",
    "லிமிடெட்",
    "எல்எல்பி",
    # French legal forms (critical for France in test set)
    "societe a responsabilite limitee",
    "societe par actions simplifiee unipersonnelle",
    "societe par actions simplifiee",
    "societe anonyme",
    "entreprise unipersonnelle a responsabilite limitee",
    "societe civile immobiliere",
    "societe en nom collectif",
    "groupement d interet economique",
    # US / UK / International legal forms
    "incorporated",
    "corporation",
    "limited",
    "company",
    "corp",
    "inc",
    "ltd",
    "llc",
    "llp",
    "pllc",
    "plc",
    "l.l.c.",
    "p.c.",
    "pc",
    "lp",
    "co",
    "gmbh",
    "sa",
    "sarl",
    "sas",
    "sasu",
    "eurl",
    "sci",
    "snc",
    "gie",
    "opc",
]
LEGAL_SUFFIXES.sort(key=len, reverse=True)

# Address abbreviations dictionary (standardized to canonical short forms)
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "drive": "dr",
    "lane": "ln",
    "boulevard": "blvd",
    "court": "ct",
    "place": "pl",
    "terrace": "ter",
    "circle": "cir",
    "square": "sq",
    "parkway": "pkwy",
    "highway": "hwy",
    "building": "bldg",
    "apartment": "apt",
    "suite": "ste",
    "floor": "fl",
    "opposite": "opp",
    "near": "nr",
    "post office": "po",
    "extension": "ext",
    # French road types
    "allee": "all",
    "impasse": "imp",
    "route": "rte",
    "chemin": "che",
}

ADDR_ABBR_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(ADDRESS_ABBREVIATIONS.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Core Normalization Functions
# ---------------------------------------------------------------------------

def remove_latin_accents(text: str) -> str:
    """
    Strips accent marks specifically from Latin characters (e.g., é -> e, ô -> o)
    while preserving non-Latin scripts (Devanagari, Tamil, Kannada, etc.) and
    their combining vowel marks (matras) completely intact.
    """
    if not text:
        return ""
    result = []
    # Decompose into base + combining marks
    for char in unicodedata.normalize('NFKD', text):
        if unicodedata.combining(char):
            # Only drop combining mark if the preceding base character is Latin [A-Za-z]
            if result and ('A' <= result[-1] <= 'Z' or 'a' <= result[-1] <= 'z'):
                continue
        result.append(char)
    return unicodedata.normalize('NFC', ''.join(result))


def clean_common_text(text: Optional[str]) -> str:
    """
    Applies standard cleaning:
    - NFKC Unicode normalization
    - Latin accent stripping
    - Lowercasing
    - Ampersand expansion (& -> 'and')
    - Apostrophe stripping (Orelee's -> orelees)
    - Punctuation to space conversion via fast C-level table translation
    - Whitespace collapsing
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s or s.lower() in ("null", "none", "nan"):
        return ""
    
    # 1. NFKC Unicode normalization
    s = unicodedata.normalize('NFKC', s)
    
    # 2. Latin accent removal (safe for Indic scripts)
    s = remove_latin_accents(s)
    
    # 3. Lowercase
    s = s.lower()
    
    # 4. Ampersands (& -> 'and')
    s = AMP_PATTERN.sub(" and ", s)
    
    # 5. Remove apostrophes completely so e.g. "orelee's" -> "orelees"
    s = s.replace("'", "").replace("’", "").replace("`", "")
    
    # 6. Fast C-level translation of punctuation and symbols to spaces
    s = s.translate(_PUNCT_TABLE)
    
    # 7. Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_name(name: Optional[str]) -> Optional[str]:
    """
    Normalizes business name:
    - Strips URL prefixes (http/https/www)
    - Separates domain extensions (e.g., 'example.com' -> 'example com')
    - Normalizes punctuation, case, and whitespace
    - Preserves all digits and multilingual scripts
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s or s.lower() in ("null", "none", "nan"):
        return None
    
    # Strip URL prefixes
    s = URL_PREFIX_PATTERN.sub('', s)
    
    # Separate domain extensions so domains remain useful name tokens
    s = DOMAIN_SUFFIX_PATTERN.sub(r' \1', s)
    
    cleaned = clean_common_text(s)
    return cleaned if cleaned else None


def normalize_name_core(name: Optional[str]) -> Optional[str]:
    """
    Produces the core business name by stripping recognized legal suffixes
    (e.g., 'abc technologies pvt ltd' -> 'abc technologies').
    Safeguard: Never produces an empty string if the original name had content.
    """
    norm = normalize_name(name)
    if not norm:
        return None
    
    cur_text = norm
    modified = True
    while modified and cur_text:
        modified = False
        words = cur_text.split()
        for suffix in LEGAL_SUFFIXES:
            s_words = suffix.split()
            n_words = len(s_words)
            # Check trailing suffix
            if len(words) > n_words and words[-n_words:] == s_words:
                words = words[:-n_words]
                cur_text = ' '.join(words)
                modified = True
                break
            # Check leading suffix (e.g. 'societe ...')
            elif len(words) > n_words and words[:n_words] == s_words:
                words = words[n_words:]
                cur_text = ' '.join(words)
                modified = True
                break
    
    core = cur_text.strip()
    return core if core else norm


def tokenize_name(name: Optional[str]) -> List[str]:
    """Returns token list from normalized business name."""
    norm = normalize_name(name)
    return norm.split() if norm else []


def normalize_address(address: Optional[str]) -> Optional[str]:
    """
    Normalizes business address:
    - Removes noise words ('null' string artifacts)
    - Normalizes case, Unicode, and punctuation
    - Standardizes common address abbreviations (street -> st, road -> rd, etc.)
    - Preserves all digits, apartment numbers, building numbers, and PIN codes
    - Returns None if the input address was missing/null
    """
    if address is None:
        return None
    s = str(address).strip()
    if not s or s.lower() in ("null", "none", "nan"):
        return None
    
    # Remove literal dangling 'null' tokens observed in noisy data
    s = re.sub(r'\bnull\b', ' ', s, flags=re.IGNORECASE)
    
    cleaned = clean_common_text(s)
    if not cleaned:
        return None
    
    # Standardize address abbreviations
    def _abbr_sub(match):
        w = match.group(1).lower()
        return ADDRESS_ABBREVIATIONS.get(w, w)
    
    cleaned = ADDR_ABBR_PATTERN.sub(_abbr_sub, cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else None


def tokenize_address(address: Optional[str]) -> List[str]:
    """Returns token list from normalized business address."""
    norm = normalize_address(address)
    return norm.split() if norm else []


def extract_address_numbers(address: Optional[str]) -> List[str]:
    """
    Extracts all alphanumeric sequence tokens that contain at least one digit
    (e.g., '12', '187c', '110018', '6-29').
    These numeric tokens are critical for distinguishing distinct locations
    (e.g., 12 MG Road vs 120 MG Road) and providing script-invariant anchors.
    """
    norm = normalize_address(address)
    if not norm:
        return []
    return [t for t in norm.split() if any(c.isdigit() for c in t)]


def normalize_country(country: Optional[str]) -> Optional[str]:
    """
    Normalizes country label:
    - Trims whitespace and handles casing
    - Maps common aliases (USA -> US, etc.)
    - Open-set: preserves arbitrary country labels (e.g. France) without hard-coding
    """
    if country is None:
        return None
    c = str(country).strip()
    if not c or c.lower() in ("null", "none", "nan"):
        return None
    
    c = unicodedata.normalize('NFKC', c).strip().upper()
    if c in ("UNITED STATES", "UNITED STATES OF AMERICA", "USA", "U.S.", "U.S.A."):
        return "US"
    if c in ("INDIA", "IND", "IN"):
        return "INDIA"
    if c in ("FRANCE", "FRA", "FR"):
        return "FRANCE"
    return c


# ---------------------------------------------------------------------------
# Batch / DataFrame Normalization Utility
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pl.DataFrame,
    include_core_name: bool = True,
    include_numbers: bool = True
) -> pl.DataFrame:
    """
    Applies preprocessing to a Polars DataFrame in a memory-efficient manner.
    Generates:
    - business_name_norm
    - business_name_core (optional)
    - business_address_norm
    - country_norm
    - address_numbers (optional)
    """
    name_norm = [normalize_name(x) for x in df["business_name"].to_list()]
    addr_norm = [normalize_address(x) for x in df["business_address"].to_list()]
    country_norm = [normalize_country(x) for x in df["country"].to_list()]
    
    cols = [
        pl.Series("business_name_norm", name_norm, dtype=pl.String),
        pl.Series("business_address_norm", addr_norm, dtype=pl.String),
        pl.Series("country_norm", country_norm, dtype=pl.String),
    ]
    
    if include_core_name:
        core_norm = [normalize_name_core(x) for x in name_norm]
        cols.append(pl.Series("business_name_core", core_norm, dtype=pl.String))
        
    if include_numbers:
        addr_nums = [" ".join(extract_address_numbers(x)) for x in addr_norm]
        cols.append(pl.Series("address_numbers", addr_nums, dtype=pl.String))
        
    return df.with_columns(cols)
