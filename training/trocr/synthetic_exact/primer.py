"""Glyph primer: nglegena, sandhangan, pasangan, digits, common words."""

from __future__ import annotations

NGLEGENA = list("ꦲꦤꦕꦫꦏꦢꦠꦱꦮꦭꦥꦝꦗꦪꦚꦩꦒꦧꦛꦔ")
MURDA = list("ꦑꦓꦖꦟꦡꦣꦦꦨꦬꦯꦰ")
SWARA = list("ꦄꦆꦈꦌꦎ")
VOWEL = ["", "ꦴ", "ꦶ", "ꦸ", "ꦺ", "ꦼ", "ꦺꦴ"]
FINAL = ["ꦁ", "ꦂ", "ꦃ"]
MEDIAL = ["ꦿ", "ꦾ"]
PANGKON = "꧀"
DIGITS = list("꧐꧑꧒꧓꧔꧕꧖꧗꧘꧙")
PADA = ["꧈", "꧉"]

HANACARAKA = [
    "ꦲꦤꦕꦫꦏ",
    "ꦢꦠꦱꦮꦭ",
    "ꦥꦝꦗꦪꦚ",
    "ꦩꦒꦧꦛꦔ",
]

WORDS = [
    "ꦲꦏ꧀ꦱꦫ",
    "ꦲꦏ꧀ꦱꦫꦗꦮ",
    "ꦗꦮ",
    "ꦧꦱ",
    "ꦧꦱꦗꦮ",
    "ꦠꦸꦭꦶꦱ꧀",
    "ꦲꦶꦏꦶ",
    "ꦲꦶꦏꦸ",
    "ꦲꦺꦴꦫ",
    "ꦲꦤ",
    "ꦲꦶꦁ",
    "ꦏꦁ",
    "ꦱꦶꦁ",
    "ꦏꦁꦒꦺꦴ",
    "ꦱꦏ",
    "ꦩꦤꦸꦁꦱ",
    "ꦫꦕꦏꦺ",
    "ꦢꦶꦤ",
    "ꦮꦺꦴꦁ",
    "ꦧꦚꦸ",
    "ꦒꦸꦤꦸꦁ",
    "ꦏꦸꦛ",
    "ꦤꦒꦫ",
    "ꦏꦂꦠ",
    "ꦱꦸꦫ",
    "ꦭꦺꦴꦫ",
    "ꦏꦶꦢꦸꦭ꧀",
    "ꦮꦺꦠꦤ꧀",
    "ꦤꦸꦭꦶꦱ꧀",
    "ꦥꦸꦤꦶꦏ",
    "ꦱꦼꦩꦼꦤ꧀",
    "ꦲꦤꦕꦫꦏ",
]


def build_primer_texts() -> list[str]:
    """Unique short strings covering the aksara inventory."""
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    for ch in NGLEGENA + MURDA + SWARA:
        add(ch)
    for cons in NGLEGENA:
        for v in VOWEL:
            add(cons + v)
        for f in FINAL:
            add(cons + f)
        add(cons + PANGKON)
        for m in MEDIAL:
            add(cons + m)
            add(cons + m + "ꦺ")
            add(cons + m + "ꦶ")
    for a in NGLEGENA:
        for b in NGLEGENA:
            add(a + PANGKON + b)
    for d in DIGITS:
        add(d)
    for p in PADA:
        add(p)
    for line in HANACARAKA:
        add(line)
    for w in WORDS:
        add(w)
    # short stacked phrases
    add("ꦲꦤꦕꦫꦏꦢꦠꦱꦮꦭ")
    add("ꦲꦏ꧀ꦱꦫꦗꦮ꧉")
    add("ꦧꦱꦗꦮ꧈")
    return out
