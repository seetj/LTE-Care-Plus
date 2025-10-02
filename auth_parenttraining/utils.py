# utils.py
import re
import unicodedata

# Regex for suffixes to ignore
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", flags=re.I)

def _strip_diacritics(s: str) -> str:
    """Remove accents/diacritics (e.g., 'José' -> 'Jose')."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def clean_name(raw: str) -> str:
    """Normalize name string: lowercase, strip punctuation, drop suffixes."""
    s = str(raw or "").strip()
    s = _strip_diacritics(s)
    s = s.lower()
    s = _SUFFIX_RE.sub(" ", s)                      # drop suffixes
    s = re.sub(r"[^a-z0-9 ,'\-]+", " ", s)          # keep only allowed chars
    s = re.sub(r"\s+", " ", s).strip()
    return s

def name_key(raw: str) -> tuple[str, str]:
    """
    Convert a raw name to a (first,last) tuple for matching.
    - Middle names/initials are ignored.
    - Handles 'Doe, John R' and 'John R Doe' equally → ('john','doe').
    - Hyphens/apostrophes in names are preserved.
    """
    s = clean_name(raw)
    if not s:
        return ("","")
    if "," in s:
        # 'Last, First [Middle ...]'
        last, rest = [p.strip() for p in s.split(",", 1)]
        toks = rest.split()
        first = toks[0] if toks else ""
    else:
        # 'First [Middle ...] Last'
        toks = s.split()
        if len(toks) == 1:
            return (toks[0], "")
        first, last = toks[0], toks[-1]
    return (first, last)
