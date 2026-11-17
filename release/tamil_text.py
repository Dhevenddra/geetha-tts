r"""Digits to Tamil words, before the text reaches the model.

The model was trained on text with numbers already written out, so "₹500" or "15/11/2026"
sent as digits is read badly or skipped. ``verbalise`` rewrites, in this order:

    ₹1,250.50   Rs. 500   ரூ. 500     ->  ... ரூபாய் ... பைசா
    15/11/2026  15-11-2026            ->  நவம்பர் பதினைந்து, இரண்டாயிரத்து இருபத்தாறு
    3:30                              ->  மூன்று மணி முப்பது நிமிடம்
    50%                               ->  ஐம்பது சதவீதம்
    9876543210  044...                ->  digit by digit (phone numbers, codes)
    1,00,000    25.5    2026-ல்  5-ஆம் ->  cardinals, "புள்ளி" decimals, fused suffixes

Everything else is returned untouched. This file has no dependencies and is shipped next
to ``inference.py``.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["cardinal", "digits", "verbalise"]

_ZERO = "பூஜ்ஜியம்"
_UNITS = ["", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]
_TEENS = ["பத்து", "பதினொன்று", "பன்னிரண்டு", "பதின்மூன்று", "பதினான்கு", "பதினைந்து",
          "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொன்பது"]
_TENS = ["", "", "இருபது", "முப்பது", "நாற்பது", "ஐம்பது", "அறுபது", "எழுபது", "எண்பது",
         "தொண்ணூறு"]
# The tens stem a unit attaches to. It ends in a bare consonant: a unit that starts with a
# vowel fuses onto it as a vowel sign (இருபத்த + ொன்று = இருபத்தொன்று); மூன்று and நான்கு
# start with a consonant and stay a separate word after "...த்து".
_TENS_STEM = ["", "", "இருபத்த", "முப்பத்த", "நாற்பத்த", "ஐம்பத்த", "அறுபத்த", "எழுபத்த",
              "எண்பத்த", "தொண்ணூற்ற"]
_UNIT_FUSED = ["", "ொன்று", "ிரண்டு", "ு மூன்று", "ு நான்கு", "ைந்து", "ாறு", "ேழு", "ெட்டு",
               "ொன்பது"]
_HUNDREDS = ["", "நூறு", "இருநூறு", "முந்நூறு", "நானூறு", "ஐநூறு", "அறுநூறு", "எழுநூறு",
             "எண்ணூறு", "தொள்ளாயிரம்"]
_HUNDREDS_JOIN = ["", "நூற்று", "இருநூற்று", "முந்நூற்று", "நானூற்று", "ஐநூற்று", "அறுநூற்று",
                  "எழுநூற்று", "எண்ணூற்று", "தொள்ளாயிரத்து"]
_MONTHS = ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்", "ஜூலை", "ஆகஸ்ட்",
           "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"]
_DAYS_IN_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_U = "ு"          # the vowel sign every Tamil number word from 1 to 999 ends in
_LIMIT = 10 ** 14      # beyond this a number is read digit by digit


def _below_hundred(n: int) -> str:
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    tens, unit = divmod(n, 10)
    return _TENS[tens] if unit == 0 else _TENS_STEM[tens] + _UNIT_FUSED[unit]


def _thousands(count: int, standalone: bool) -> str:
    """``count`` thousand (1..99) as one word: இரண்டாயிரம், பதினோராயிரம், இருபதாயிரம்."""
    if count == 1:
        return "ஆயிரம்" if standalone else "ஓராயிரம்"
    if count == 3:
        return "மூவாயிரம்"
    word = _below_hundred(count)
    if count % 10 == 1:                       # ...ொன்று -> ...ோராயிரம்
        return word[: -len("ொன்று")] + "ோராயிரம்"
    return word[:-1] + "ாயிரம்"               # drop the final u, add -aayiram


def cardinal(n: int) -> str:
    """A non-negative whole number in Tamil words, Indian grouping (lakh, crore)."""
    if n < 0:
        raise ValueError(f"cardinal takes a non-negative number, got {n}")
    if n == 0:
        return _ZERO
    if n >= _LIMIT:
        return digits(str(n))
    parts: list[str] = []
    crores, n = divmod(n, 10_000_000)
    if crores:
        word = "ஒரு கோடி" if crores == 1 else f"{cardinal(crores)} கோடி"
        parts.append(word + "யே" if n else word)
    lakhs, n = divmod(n, 100_000)
    if lakhs:
        word = "ஒரு" if lakhs == 1 else _below_hundred(lakhs)
        parts.append(f"{word} லட்சத்து" if n else f"{word} லட்சம்")
    thousands, n = divmod(n, 1000)
    if thousands:
        word = _thousands(thousands, standalone=not parts)
        parts.append(word[:-2] + "த்து" if n else word)      # ஆயிரம் -> ஆயிரத்து
    hundreds, n = divmod(n, 100)
    if hundreds:
        parts.append(_HUNDREDS_JOIN[hundreds] if n else _HUNDREDS[hundreds])
    if n:
        parts.append(_below_hundred(n))
    return " ".join(parts)


def digits(text: str) -> str:
    """Each digit by name, for phone numbers and codes."""
    return " ".join(_ZERO if ch == "0" else _UNITS[int(ch)] for ch in text if ch.isdigit())


def _number(token: str) -> str:
    """'1,00,000' / '25.5' / '007' as words."""
    whole, _, fraction = token.replace(",", "").partition(".")
    if len(whole) > 1 and whole.startswith("0") or len(whole) >= 8 and "," not in token:
        spoken = digits(whole)                 # a code or a phone number, not a quantity
    else:
        spoken = cardinal(int(whole))
    return f"{spoken} புள்ளி {digits(fraction)}" if fraction else spoken


_SUFFIX_ON_U = {"ஆம்": "ாம்", "ம்": "ாம்", "ஆவது": "ாவது", "வது": "ாவது",
                "இல்": "ில்", "ல்": "ில்", "ல": "ில"}
_SUFFIX_ON_M = {"ஆம்": "மாம்", "ம்": "மாம்", "ஆவது": "மாவது", "வது": "மாவது",
                "இல்": "த்தில்", "ல்": "த்தில்", "ல": "த்தில", "க்கு": "த்துக்கு"}
_SUFFIX_ON_I = {"ஆம்": "யாம்", "ம்": "யாம்", "ஆவது": "யாவது", "வது": "யாவது",
                "இல்": "யில்", "ல்": "யில்", "ல": "யில", "க்கு": "க்கு"}


def _attach(word: str, suffix: str) -> str:
    """2026-ல் -> ...இருபத்தாறில்; 5-ஆம் -> ஐந்தாம்; 1000-க்கு -> ஆயிரத்துக்கு."""
    if not suffix:
        return word
    if word.endswith(_U):
        return word + "க்கு" if suffix == "க்கு" else word[:-1] + _SUFFIX_ON_U[suffix]
    if word.endswith("ம்"):
        return word[:-2] + _SUFFIX_ON_M[suffix]
    if word.endswith("ி"):
        return word + _SUFFIX_ON_I[suffix]
    return f"{word} {suffix}"


_NUM = r"\d+(?:,\d+)*(?:\.\d+)?"
_TAMIL = r"஀-௿"
_TAMIL_DIGITS = {0x0BE6 + i: ord("0") + i for i in range(10)}
_CURRENCY_BEFORE = re.compile(rf"(?:₹|\bRs\.?|ரூ\.?(?![{_TAMIL}]))\s*({_NUM})", re.IGNORECASE)
_CURRENCY_AFTER = re.compile(rf"({_NUM})\s*(?:₹|ரூ\.?(?![{_TAMIL}])|\bRs\b\.?)", re.IGNORECASE)
_DATE = re.compile(r"(?<![\d/-])(\d{1,2})([/-])(\d{1,2})\2(\d{4})(?![\d/-])")
_TIME = re.compile(r"(?<![\d:.])(\d{1,2}):(\d{2})(?![\d:])")
_PERCENT = re.compile(rf"({_NUM})\s*%")
_PHONE = re.compile(r"(?<![\d,.])\+?\d[\d -]{8,}\d(?![\d,])")
_PLAIN = re.compile(rf"({_NUM})(?:-?(ஆவது|ஆம்|வது|ம்|இல்|ல்|ல|க்கு)(?![{_TAMIL}]))?")
_SPACES = re.compile(r"[ \t]+")


def _rupees(match: re.Match[str]) -> str:
    whole, _, fraction = match.group(1).replace(",", "").partition(".")
    spoken = f"{cardinal(int(whole))} ரூபாய்"
    paise = int((fraction + "0")[:2]) if fraction else 0
    return f" {spoken} {cardinal(paise)} பைசா " if paise else f" {spoken} "


def _date(match: re.Match[str]) -> str:
    day, month, year = int(match.group(1)), int(match.group(3)), int(match.group(4))
    if not (1 <= month <= 12 and 1 <= day <= _DAYS_IN_MONTH[month - 1]):
        return match.group(0)                  # not a date: the number rule reads it
    return f" {_MONTHS[month - 1]} {cardinal(day)}, {cardinal(year)} "


def _time(match: re.Match[str]) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return match.group(0)
    if match.string[match.end():].lstrip().startswith("மணி"):
        # "3:30 மணிக்கு": the writer already says மணி, so only the numbers are read
        return f" {cardinal(hour)} {cardinal(minute)} " if minute else f" {cardinal(hour)} "
    spoken = f"{cardinal(hour)} மணி"
    return f" {spoken} {cardinal(minute)} நிமிடம் " if minute else f" {spoken} "


def _phone(match: re.Match[str]) -> str:
    token = match.group(0)
    count = sum(ch.isdigit() for ch in token)
    if count < 10 and not token.startswith(("+", "0")):
        return token                           # too short to be a phone number
    return f" {digits(token)} "


def verbalise(text: str) -> str:
    """Rewrite every number in ``text`` as Tamil words. Text without digits is unchanged."""
    text = unicodedata.normalize("NFC", text).translate(_TAMIL_DIGITS)
    if not any(ch.isdigit() for ch in text):
        return text
    text = _CURRENCY_BEFORE.sub(_rupees, text)
    text = _CURRENCY_AFTER.sub(_rupees, text)
    text = _DATE.sub(_date, text)
    text = _TIME.sub(_time, text)
    text = _PERCENT.sub(lambda m: f" {_number(m.group(1))} சதவீதம் ", text)
    text = _PHONE.sub(_phone, text)
    text = _PLAIN.sub(lambda m: f" {_attach(_number(m.group(1)), m.group(2) or '')} ", text)
    text = _SPACES.sub(" ", text)
    return re.sub(r" ([,.!?;:…])", r"\1", text).strip()
