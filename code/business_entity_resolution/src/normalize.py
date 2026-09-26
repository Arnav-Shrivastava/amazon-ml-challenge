"""
normalize.py — Text normalization and abbreviation expansion.

All blocking and feature code calls functions from this module before
comparing strings. Normalization is vectorized (pandas str operations) for
performance on multi-million-row DataFrames.

Key functions
-------------
preprocess_df(df)
    Add _norm_name, _norm_name_stripped, _norm_addr, _country, _postal,
    _city_tokens, _house_num, _script_key, _sorted_name_tokens, _soundex_key
    columns to a source DataFrame. Call once; reuse columns everywhere.

normalize_series(series, expand_addr)
    Pure vectorized normalize: lowercase → strip punct → expand abbreviations.

strip_legal_suffix_series(series)
    Remove trailing legal suffix from a normalized name series.
"""

from __future__ import annotations

import re

import jellyfish
import pandas as pd

# ── Regex ─────────────────────────────────────────────────────────────────────
_PUNCT      = re.compile(r"[^\w\s]")        # keep word chars and spaces
_MULTI_SPC  = re.compile(r"\s+")
_POSTAL_US  = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_POSTAL_IN  = re.compile(r"\b([1-9]\d{5})\b")  # 6-digit India PIN (starts 1-9)
_HOUSE_NUM  = re.compile(r"\b(\d{3,})\b")       # first 3+ digit number

# ── Abbreviation tables ───────────────────────────────────────────────────────
# Applied as whole-word regex replacements after lowercasing.

# Legal suffix normalisations (applied to names AND addresses)
_LEGAL_PATTERNS: list[tuple[str, str]] = [
    (r"\bpvt\.?\s+ltd\.?\b", "private limited"),
    (r"\bpvt\.?\b",          "private"),
    (r"\bltd\.?\b",          "limited"),
    (r"\bl\.l\.c\.?\b",      "llc"),
    (r"\bllc\.?\b",          "llc"),
    (r"\bl\.l\.p\.?\b",      "llp"),
    (r"\bllp\.?\b",          "llp"),
    (r"\binc\.?\b",          "incorporated"),
    (r"\bcorp\.?\b",         "corporation"),
    (r"\bco\.?\b",           "company"),
    (r"\bplc\.?\b",          "plc"),
    (r"\bpty\.?\b",          "pty"),
    (r"\bpc\.?\b",           "pc"),
    (r"\bgmbh\.?\b",         "gmbh"),
    (r"\bsa\.?\b",           "sa"),
    (r"\bsas\.?\b",          "sas"),
    (r"\bsarl\.?\b",         "sarl"),
    (r"\bnv\.?\b",           "nv"),
    (r"\bbv\.?\b",           "bv"),
    (r"\bag\.?\b",           "ag"),
    (r"&",                   "and"),
]

# Set of known legal suffix tokens (for stripping at end of name)
_LEGAL_SUFFIXES: frozenset[str] = frozenset([
    "private limited", "limited", "incorporated", "corporation", "company",
    "llc", "llp", "plc", "pty", "pc", "sa", "sas", "sarl", "nv", "bv",
    "ag", "gmbh", "private", "ltd", "pvt",
])

# Address-specific abbreviations
_ADDR_PATTERNS: list[tuple[str, str]] = [
    (r"\brd\.?\b",   "road"),
    (r"\bst\.?\b",   "street"),
    (r"\bave\.?\b",  "avenue"),
    (r"\bblvd\.?\b", "boulevard"),
    (r"\bdr\.?\b",   "drive"),
    (r"\bln\.?\b",   "lane"),
    (r"\bct\.?\b",   "court"),
    (r"\bpkwy\.?\b", "parkway"),
    (r"\bhwy\.?\b",  "highway"),
    (r"\bapt\.?\b",  "apartment"),
    (r"\bste\.?\b",  "suite"),
    (r"\bft\.?\b",   "fort"),
    (r"\bmt\.?\b",   "mount"),
    (r"\bno\.?\b",   "number"),
]

# Name-specific miscellaneous abbreviations
_NAME_PATTERNS: list[tuple[str, str]] = [
    (r"\bintl\.?\b", "international"),
    (r"\bnatl\.?\b", "national"),
    (r"\bmgmt\.?\b", "management"),
    (r"\bsvcs\.?\b", "services"),
    (r"\bsvc\.?\b",  "service"),
    (r"\bgrp\.?\b",  "group"),
    (r"\buniv\.?\b", "university"),
    (r"\btech\.?\b", "technology"),
    (r"\bsys\.?\b",  "systems"),
    (r"\bno\.?\b",   "number"),
]

# ── Core normalisation ────────────────────────────────────────────────────────

def normalize_series(series: pd.Series, expand_addr: bool = False) -> pd.Series:
    """
    Vectorized normalization:
      1. Lowercase + strip
      2. Remove punctuation (keep spaces)
      3. Expand legal + name abbreviations
      4. Optionally expand address abbreviations
      5. Collapse whitespace

    Parameters
    ----------
    series      : pd.Series of raw strings
    expand_addr : also expand address abbreviations (Rd→road, etc.)
    """
    s = series.fillna("").str.lower().str.strip()
    s = s.str.replace(_PUNCT, " ", regex=True)

    for pat, repl in _LEGAL_PATTERNS + _NAME_PATTERNS:
        s = s.str.replace(pat, repl, regex=True)

    if expand_addr:
        for pat, repl in _ADDR_PATTERNS:
            s = s.str.replace(pat, repl, regex=True)

    s = s.str.replace(_MULTI_SPC, " ", regex=True).str.strip()
    return s


def strip_legal_suffix_series(series: pd.Series) -> pd.Series:
    """
    Remove trailing legal suffix tokens from a normalized name Series.
    Tries 3-token, then 2-token, then 1-token suffix — whichever matches first.
    Returns name with suffix removed (empty series value → empty string).
    """
    def _strip(name: str) -> str:
        if not name:
            return name
        tokens = name.split()
        for n_tail in [3, 2, 1]:
            if len(tokens) <= n_tail:
                continue
            tail = " ".join(tokens[-n_tail:])
            if tail in _LEGAL_SUFFIXES:
                return " ".join(tokens[:-n_tail])
        if tokens and tokens[-1] in _LEGAL_SUFFIXES:
            return " ".join(tokens[:-1])
        return name

    return series.apply(_strip)


# ── Field extractors ──────────────────────────────────────────────────────────

def extract_postal_series(raw_address: pd.Series) -> pd.Series:
    """Extract US 5-digit ZIP or India 6-digit PIN from raw address. Returns '' if none."""
    us_zip = raw_address.str.extract(_POSTAL_US, expand=False).fillna("")
    in_pin = raw_address.str.extract(_POSTAL_IN, expand=False).fillna("")
    return us_zip.where(us_zip != "", in_pin)


def extract_house_number_series(raw_address: pd.Series) -> pd.Series:
    """Extract first 3+ digit number from raw address (building/house number)."""
    return raw_address.str.extract(_HOUSE_NUM, expand=False).fillna("")


def extract_city_tokens_series(norm_address: pd.Series) -> pd.Series:
    """
    Extract candidate city tokens from normalized address.
    Splits on commas; keeps first alpha-only token (len > 3) per segment.
    Returns pipe-separated candidates per row (e.g. 'lucknow|ansal|uttar').
    """
    def _cities(addr: str) -> str:
        if not addr.strip():
            return ""
        seen: list[str] = []
        for seg in addr.split(","):
            alpha = [t for t in seg.split() if t.isalpha() and len(t) > 3]
            if alpha and alpha[0] not in seen:
                seen.append(alpha[0])
        return "|".join(seen)

    return norm_address.apply(_cities)


def script_key_series(raw_name: pd.Series) -> pd.Series:
    """
    Extract non-ASCII character sequence as a blocking key.
    Useful for matching Devanagari / Hindi script names across S2 and S3.
    Returns '' for ASCII-only names.
    """
    def _key(name: str) -> str:
        non_ascii = "".join(c for c in name if ord(c) >= 128)
        return non_ascii[:60].strip()  # cap to avoid pathological length

    return raw_name.fillna("").apply(_key)


def sorted_name_tokens_series(norm_name_stripped: pd.Series, n: int = 3) -> pd.Series:
    """
    Sorted first-n significant tokens of a suffix-stripped normalized name.
    Ignores common stopwords and very short tokens.
    Used for B1 blocking key — robust to within-name word reordering.
    """
    _stop = frozenset(["and", "of", "the", "a", "an", "in", "for", "at", "by",
                        "on", "with", "to", "from", "la", "le", "les", "de", "du"])

    def _sorted(name: str) -> str:
        tokens = [t for t in name.split() if len(t) > 1 and t not in _stop]
        top = tokens[:n] if len(tokens) >= n else tokens
        return " ".join(sorted(top))

    return norm_name_stripped.apply(_sorted)


def soundex_key_series(norm_name: pd.Series, n_tokens: int = 2) -> pd.Series:
    """
    Space-joined Soundex codes for the first n_tokens of a normalized name.
    Used for B2 phonetic blocking key.
    """
    def _soundex(name: str) -> str:
        tokens = [t for t in name.split() if len(t) > 1][:n_tokens]
        if not tokens:
            return ""
        return " ".join(jellyfish.soundex(t) for t in tokens)

    return norm_name.apply(_soundex)


# ── Main preprocessing ────────────────────────────────────────────────────────

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived blocking/feature columns to a source DataFrame.
    Columns added (all prefixed with '_'):
        _norm_name          : normalized business_name
        _norm_name_stripped : norm_name with legal suffix removed
        _norm_addr          : normalized business_address (with address abbrevs)
        _country            : lowercased country
        _postal             : extracted postal/PIN code
        _city_tokens        : pipe-separated city token candidates from address
        _house_num          : extracted house/building number (3+ digits)
        _script_key         : non-ASCII character sequence from raw name
        _sorted_name_tokens : sorted first-3 significant tokens of stripped name
        _soundex_key        : Soundex of first 2 name tokens

    Does NOT modify the original DataFrame.
    """
    df = df.copy()

    df["_norm_name"]          = normalize_series(df["business_name"])
    df["_norm_name_stripped"] = strip_legal_suffix_series(df["_norm_name"])
    df["_norm_addr"]          = normalize_series(df["business_address"], expand_addr=True)
    df["_country"]            = df["country"].fillna("").str.lower().str.strip()

    df["_postal"]             = extract_postal_series(df["business_address"])
    df["_city_tokens"]        = extract_city_tokens_series(df["_norm_addr"])
    df["_house_num"]          = extract_house_number_series(df["business_address"])
    df["_script_key"]         = script_key_series(df["business_name"])
    df["_sorted_name_tokens"] = sorted_name_tokens_series(df["_norm_name_stripped"])
    df["_soundex_key"]        = soundex_key_series(df["_norm_name"])

    return df
