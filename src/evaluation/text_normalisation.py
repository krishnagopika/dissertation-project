"""Text normalisation for WER — two explicit variants, no hidden choices.

Why this module exists
----------------------
`compute_wer.py` used a single `jiwer.RemovePunctuation()` transform, which
DELETES punctuation rather than separating on it. That gives two different
behaviours from one rule, neither of them chosen:

    'KL-5'   -> 'kl5'     hyphen deleted, tokens merged
    "don't"  -> 'dont'    apostrophe deleted, tokens merged

The first is wrong: if Voxtral transcribes "KL 5" (two tokens, arguably
correct) against a gold of "KL-5" (one token, merged to 'kl5'), that scores
WER 2.0 on a one-word reference -- one substitution plus one insertion, for a
transcription that was not wrong. The second is right, and happens to match
MELD's gold text, which has apostrophes stripped already ('mustve', 'companys').

Rather than pick one and bury it, this module defines both endpoints
explicitly, and every WER we report says which it used.

EXACT
    No normalisation at all. Split on whitespace, compare verbatim.
    Case-sensitive, punctuation-sensitive. Nothing is forgiven.
    Use this as the reproducible floor: it embeds zero judgement calls, so it
    cannot be argued with. It will read high, because "Hello." != "hello".

NORMALISED
    Standard ASR scoring practice, applied consistently:
      1. lowercase
      2. drop apostrophes WITHOUT inserting a space  ("don't" -> "dont")
         -- matches MELD's gold, which is already apostrophe-stripped
      3. replace every other punctuation mark WITH A SPACE  ("KL-5" -> "kl 5")
         -- separates rather than merges, which is the actual fix
      4. collapse runs of whitespace, strip
    Ordering matters: step 2 must precede step 3, or "don't" becomes "don t".

Report both. Where they diverge sharply, the divergence is itself the finding:
it localises how much of the error rate is formatting rather than content.
"""

from __future__ import annotations

import re
import string
from typing import List

import jiwer

#: cp1252 punctuation that was mojibake-encoded into UTF-8 and now decodes to
#: C1 control characters. MELD's CSVs carry these in ~27% of utterances: the
#: byte pair C2 92 is UTF-8 for U+0092, which was originally cp1252 0x92, a
#: right single quote. Left unrepaired, every contraction in a quarter of the
#: corpus scores as a substitution against correctly-transcribed ASR.
#:
#: Repaired BEFORE either WER policy runs, because this is file corruption, not
#: a transcription difference -- the EXACT policy should be verbatim about what
#: was *said*, not about a broken encoding.
_MOJIBAKE = {
    "\u0091": "'", "\u0092": "'",          # left/right single quote
    "\u0093": '"', "\u0094": '"',          # left/right double quote
    "\u0096": "-", "\u0097": "-",          # en/em dash
    "\u0085": "...", "\u0095": "*",        # ellipsis, bullet
    "\u2018": "'", "\u2019": "'",          # already-correct curly quotes
    "\u201c": '"', "\u201d": '"',
}


def repair_encoding(text: str) -> str:
    """Undo cp1252->UTF-8 mojibake in MELD reference text.

    Args:
        text: Raw string as read from the MELD CSV.

    Returns:
        The same string with C1 control characters mapped back to the
        punctuation they were before the bad encoding round-trip.
    """
    for bad, good in _MOJIBAKE.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


#: Punctuation that separates tokens. Everything in string.punctuation except
#: the apostrophe, which is handled first and deleted rather than separated.
_SEPARATING = "".join(c for c in string.punctuation if c not in "'’")
_SEPARATING_RE = re.compile(f"[{re.escape(_SEPARATING)}]")
_APOSTROPHE_RE = re.compile(r"['’]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Apply the NORMALISED policy to one string.

    Args:
        text: Raw transcript or reference.

    Returns:
        Lowercased string with apostrophes removed, other punctuation replaced
        by spaces, and whitespace collapsed.
    """
    t = repair_encoding(text).lower()
    t = _APOSTROPHE_RE.sub("", t)        # don't -> dont   (delete, no space)
    t = _SEPARATING_RE.sub(" ", t)       # KL-5  -> kl 5   (separate, not merge)
    return _WHITESPACE_RE.sub(" ", t).strip()


class _Normalise(jiwer.transforms.AbstractTransform):
    """jiwer transform wrapping :func:`normalise_text`."""

    def process_string(self, s: str) -> str:                   # noqa: D102
        return normalise_text(s)

    def process_list(self, inp: List[str]) -> List[str]:       # noqa: D102
        return [self.process_string(s) for s in inp]


#: Verbatim: whitespace tokenisation only. No case folding, no punctuation
#: handling. The reproducible floor.
EXACT = jiwer.Compose([
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(),
])

#: Consistent ASR-style normalisation. See module docstring.
NORMALISED = jiwer.Compose([
    _Normalise(),
    jiwer.ReduceToListOfListOfWords(),
])

#: Character-level counterparts, for CER.
EXACT_CHARS = jiwer.Compose([
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfChars(),
])

NORMALISED_CHARS = jiwer.Compose([
    _Normalise(),
    jiwer.ReduceToListOfListOfChars(),
])

#: Selectable by name from a config or CLI flag.
POLICIES = {
    "exact": (EXACT, EXACT_CHARS),
    "normalised": (NORMALISED, NORMALISED_CHARS),
}
