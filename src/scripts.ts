/**
 * Transliteration for five Indonesian Brahmic writing traditions.
 *
 * Character inventories and shaping order are based on the Unicode Standard:
 * - Kawi: https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-17/#G41642
 * - Balinese: https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-17/#G26723
 * - Sundanese: https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-17/#G27247
 * - Sasak: Unicode Technical Note #51, section 3
 *   https://www.unicode.org/notes/tn51/UTN51-Balinese-Characters-1.pdf
 * - Buginese: https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-17/#G26977
 *
 * The API intentionally lives alongside, rather than inside, the original
 * Javanese state machine. This keeps every existing Aksara constructor and
 * static method backwards compatible.
 */

export type SupportedScript =
  | "kawi"
  | "balinese"
  | "sundanese"
  | "sasak"
  | "buginese";

interface ScriptProfile {
  consonants: { [latin: string]: string };
  independentVowels: { [latin: string]: string };
  vowelSigns: { [latin: string]: string };
  killer?: string;
  conjoiner?: string;
  medials?: { [latin: string]: string };
  finalSigns?: { [latin: string]: string };
  digits?: { [latin: string]: string };
  punctuation?: { [latin: string]: string };
  // Lontara does not traditionally record codas or consonant conjuncts.
  omitCodas?: boolean;
}

const digits = (zero: number): { [latin: string]: string } => {
  const result: { [latin: string]: string } = {};
  for (let i = 0; i <= 9; i++) result[String(i)] = String.fromCodePoint(zero + i);
  return result;
};

const KAWI: ScriptProfile = {
  consonants: {
    k: "𑼒", kh: "𑼓", g: "𑼔", gh: "𑼕", ng: "𑼖",
    c: "𑼗", ch: "𑼘", j: "𑼙", jh: "𑼚", ny: "𑼛",
    ṭ: "𑼜", ṭh: "𑼝", ḍ: "𑼞", ḍh: "𑼟", ṇ: "𑼠",
    t: "𑼡", th: "𑼢", d: "𑼣", dh: "𑼤", n: "𑼥",
    p: "𑼦", ph: "𑼧", b: "𑼨", bh: "𑼩", m: "𑼪",
    y: "𑼫", r: "𑼬", l: "𑼭", w: "𑼮", v: "𑼮",
    ś: "𑼯", ṣ: "𑼰", s: "𑼱", h: "𑼲", jñ: "𑼳",
  },
  independentVowels: {
    a: "𑼄", ā: "𑼅", i: "𑼆", ī: "𑼇", u: "𑼈", ū: "𑼉",
    é: "𑼎", ai: "𑼏", o: "𑼐", e: "𑼄𑽀",
  },
  vowelSigns: {
    a: "", ā: "𑼴", i: "𑼶", ī: "𑼷", u: "𑼸", ū: "𑼹",
    é: "𑼾", ai: "𑼿", o: "𑼾𑼴", e: "𑽀",
  },
  killer: "𑽁",
  conjoiner: "𑽂",
  digits: digits(0x11f50),
  punctuation: { ",": "𑽃", ".": "𑽄" },
};

const BALINESE_CONSONANTS: { [latin: string]: string } = {
  k: "ᬓ", kh: "ᬔ", g: "ᬕ", gh: "ᬖ", ng: "ᬗ",
  c: "ᬘ", ch: "ᬙ", j: "ᬚ", jh: "ᬛ", ny: "ᬜ",
  ṭ: "ᬝ", ṭh: "ᬞ", ḍ: "ᬟ", ḍh: "ᬠ", ṇ: "ᬡ",
  t: "ᬢ", th: "ᬣ", d: "ᬤ", dh: "ᬥ", n: "ᬦ",
  p: "ᬧ", ph: "ᬨ", b: "ᬩ", bh: "ᬪ", m: "ᬫ",
  y: "ᬬ", r: "ᬭ", l: "ᬮ", w: "ᬯ", v: "ᬯ",
  ś: "ᬰ", ṣ: "ᬱ", s: "ᬲ", h: "ᬳ",
};

const BALINESE_VOWELS = {
  a: "ᬅ", ā: "ᬆ", i: "ᬇ", ī: "ᬈ", u: "ᬉ", ū: "ᬊ",
  é: "ᬏ", ai: "ᬐ", o: "ᬑ", au: "ᬒ", e: "ᬅᭂ",
};

const BALINESE_SIGNS = {
  a: "", ā: "ᬵ", i: "ᬶ", ī: "ᬷ", u: "ᬸ", ū: "ᬹ",
  é: "ᬾ", ai: "ᬿ", o: "ᭀ", au: "ᭁ", e: "ᭂ",
};

const BALINESE: ScriptProfile = {
  consonants: BALINESE_CONSONANTS,
  independentVowels: BALINESE_VOWELS,
  vowelSigns: BALINESE_SIGNS,
  killer: "᭄",
  conjoiner: "᭄",
  finalSigns: { ng: "ᬂ", r: "ᬃ", h: "ᬄ" },
  digits: digits(0x1b50),
  punctuation: { ",": "᭚", ".": "᭞" },
};

const SASAK: ScriptProfile = {
  ...BALINESE,
  consonants: {
    ...BALINESE_CONSONANTS,
    // UTN #51 §3.2 documents KAF SASAK as /ʔ/, normally romanized q.
    q: "ᭅ",
  },
};

const SUNDANESE: ScriptProfile = {
  consonants: {
    k: "ᮊ", q: "ᮋ", g: "ᮌ", ng: "ᮍ", c: "ᮎ", j: "ᮏ",
    z: "ᮐ", ny: "ᮑ", t: "ᮒ", d: "ᮓ", n: "ᮔ", p: "ᮕ",
    f: "ᮖ", v: "ᮗ", b: "ᮘ", m: "ᮙ", y: "ᮚ", r: "ᮛ",
    l: "ᮜ", w: "ᮝ", s: "ᮞ", x: "ᮟ", h: "ᮠ", kh: "ᮮ",
    sy: "ᮯ",
  },
  independentVowels: {
    a: "ᮃ", i: "ᮄ", u: "ᮅ", é: "ᮆ", o: "ᮇ", e: "ᮈ", eu: "ᮉ",
  },
  vowelSigns: {
    a: "", i: "ᮤ", u: "ᮥ", é: "ᮦ", o: "ᮧ", e: "ᮨ", eu: "ᮩ",
  },
  // Modern pamaaeh is a visible killer and does not request a conjunct.
  killer: "᮪",
  conjoiner: "᮪",
  medials: { y: "ᮡ", r: "ᮢ", l: "ᮣ" },
  finalSigns: { ng: "ᮀ", r: "ᮁ", h: "ᮂ" },
  digits: digits(0x1bb0),
};

const BUGINESE: ScriptProfile = {
  consonants: {
    ngk: "ᨃ", mp: "ᨇ", nr: "ᨋ", nc: "ᨏ",
    k: "ᨀ", g: "ᨁ", ng: "ᨂ", p: "ᨄ", b: "ᨅ", m: "ᨆ",
    t: "ᨈ", d: "ᨉ", n: "ᨊ", c: "ᨌ", j: "ᨍ", ny: "ᨎ",
    y: "ᨐ", r: "ᨑ", l: "ᨒ", w: "ᨓ", v: "ᨓ", s: "ᨔ", h: "ᨖ",
  },
  // U+1A15 LETTER A is the carrier for all standalone vowels.
  independentVowels: {
    a: "ᨕ", i: "ᨕᨗ", u: "ᨕᨘ", é: "ᨕᨙ", o: "ᨕᨚ", e: "ᨕᨛ",
  },
  vowelSigns: {
    a: "", i: "ᨗ", u: "ᨘ", é: "ᨙ", o: "ᨚ", e: "ᨛ",
  },
  punctuation: { ",": "᨞", ".": "᨟" },
  omitCodas: true,
};

const PROFILES: { [key in SupportedScript]: ScriptProfile } = {
  kawi: KAWI,
  balinese: BALINESE,
  sundanese: SUNDANESE,
  sasak: SASAK,
  buginese: BUGINESE,
};

type LatinToken = { value: string; kind: "consonant" | "vowel" | "other" };

function latinTokens(text: string, profile: ScriptProfile): LatinToken[] {
  const normalized = text.normalize("NFC").toLowerCase();
  const consonants = Object.keys(profile.consonants);
  const vowels = Object.keys(profile.vowelSigns);
  const keys = consonants.concat(vowels).sort((a, b) => b.length - a.length);
  const result: LatinToken[] = [];

  for (let i = 0; i < normalized.length;) {
    let matched = "";
    for (const key of keys) {
      if (normalized.slice(i, i + key.length) === key) {
        matched = key;
        break;
      }
    }
    if (matched) {
      result.push({
        value: matched,
        kind: profile.consonants[matched] ? "consonant" : "vowel",
      });
      i += matched.length;
    } else {
      const codePoint = String.fromCodePoint(normalized.codePointAt(i)!);
      result.push({ value: codePoint, kind: "other" });
      i += codePoint.length;
    }
  }
  return result;
}

function isBoundary(token: LatinToken | undefined): boolean {
  return !token || token.kind === "other";
}

/**
 * Converts Latin text to a supported script.
 *
 * `e` consistently denotes schwa/pepet and `é` denotes /e/. `eu` is
 * recognized for Sundanese. Long Indic vowels may be written ā, ī and ū.
 */
export function toScript(
  text: string,
  script: SupportedScript,
  spaces: boolean = false,
): string {
  const profile = PROFILES[script];
  const tokens = latinTokens(text, profile);
  let output = "";

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const previous = tokens[i - 1];
    const next = tokens[i + 1];

    if (token.kind === "vowel") {
      output += profile.independentVowels[token.value];
      continue;
    }

    if (token.kind === "other") {
      if (token.value === "_") continue;
      if (token.value === " " && !spaces) continue;
      output +=
        (profile.punctuation && profile.punctuation[token.value]) ||
        (profile.digits && profile.digits[token.value]) ||
        token.value;
      continue;
    }

    const base = profile.consonants[token.value];
    if (next && next.kind === "vowel") {
      output += base + profile.vowelSigns[next.value];
      i++;
      continue;
    }

    const afterNext = tokens[i + 2];
    if (
      next && next.kind === "consonant" &&
      profile.medials && profile.medials[next.value] &&
      afterNext && afterNext.kind === "vowel"
    ) {
      output +=
        base + profile.medials[next.value] + profile.vowelSigns[afterNext.value];
      i += 2;
      continue;
    }

    if (next && next.kind === "consonant") {
      if (profile.omitCodas) {
        // Lontara has no conjuncts or vowel killer. A consonant following a
        // completed vowel-bearing syllable is an unexpressed coda.
        if (previous && previous.kind === "vowel") continue;
        output += base;
      } else {
        output += base + profile.conjoiner;
      }
      continue;
    }

    if (isBoundary(next)) {
      const finalSign = profile.finalSigns && profile.finalSigns[token.value];
      if (finalSign && previous && previous.kind === "vowel") {
        output += finalSign;
      } else if (profile.omitCodas && previous && previous.kind === "vowel") {
        // The Unicode Buginese repertoire intentionally does not encode codas.
      } else {
        output += base + (profile.killer || "");
      }
    }
  }

  return output;
}

function reverseMap(source: { [latin: string]: string }): Map<string, string> {
  const result = new Map<string, string>();
  for (const latin of Object.keys(source)) {
    // Prefer the first canonical spelling for aliases such as w/v.
    if (!result.has(source[latin])) result.set(source[latin], latin);
  }
  return result;
}

function entriesByLength(source: Map<string, string>): Array<[string, string]> {
  return Array.from(source.entries()).sort(
    (a, b) => Array.from(b[0]).length - Array.from(a[0]).length,
  );
}

function matchAt(
  chars: string[],
  index: number,
  entries: Array<[string, string]>,
): [string, string] | undefined {
  for (const entry of entries) {
    const glyph = Array.from(entry[0]);
    if (glyph.every((char, offset) => chars[index + offset] === char)) return entry;
  }
  return undefined;
}

/** Converts Unicode text in a supported script back to canonical Latin. */
export function fromScript(text: string, script: SupportedScript): string {
  const profile = PROFILES[script];
  const chars = Array.from(text.normalize("NFC"));
  const consonants = reverseMap(profile.consonants);
  const independent = entriesByLength(reverseMap(profile.independentVowels));
  const vowelSigns = entriesByLength(
    reverseMap(
      Object.keys(profile.vowelSigns)
        .filter((key) => profile.vowelSigns[key] !== "")
        .reduce((map, key) => {
          map[key] = profile.vowelSigns[key];
          return map;
        }, {} as { [latin: string]: string }),
    ),
  );
  const medials = profile.medials ? reverseMap(profile.medials) : new Map<string, string>();
  const finals = profile.finalSigns ? reverseMap(profile.finalSigns) : new Map<string, string>();
  const nativeDigits = profile.digits ? reverseMap(profile.digits) : new Map<string, string>();
  const punctuation = profile.punctuation
    ? reverseMap(profile.punctuation)
    : new Map<string, string>();
  let output = "";

  for (let i = 0; i < chars.length;) {
    const standalone = matchAt(chars, i, independent);
    if (standalone) {
      output += standalone[1];
      i += Array.from(standalone[0]).length;
      continue;
    }

    const consonant = consonants.get(chars[i]);
    if (consonant) {
      output += consonant;
      i++;

      const medial = medials.get(chars[i]);
      if (medial) {
        output += medial;
        i++;
      }

      const vowel = matchAt(chars, i, vowelSigns);
      if (vowel) {
        output += vowel[1];
        i += Array.from(vowel[0]).length;
      } else if (chars[i] === profile.killer || chars[i] === profile.conjoiner) {
        i++;
      } else {
        output += "a";
      }
      continue;
    }

    const final = finals.get(chars[i]);
    if (final) output += final;
    else if (nativeDigits.has(chars[i])) output += nativeDigits.get(chars[i]);
    else if (punctuation.has(chars[i])) output += punctuation.get(chars[i]);
    else output += chars[i];
    i++;
  }

  return output;
}

/** Common class API used by each named script below. */
export abstract class ScriptTransliterator {
  protected abstract readonly script: SupportedScript;

  constructor(
    public readonly text: string,
    public readonly spaces: boolean = false,
  ) {}

  getText(): string {
    return this.text;
  }

  getAksara(): string {
    return toScript(this.text, this.script, this.spaces);
  }
}

export class Kawi extends ScriptTransliterator {
  protected readonly script = "kawi" as SupportedScript;
  static fromAksara(text: string): string { return fromScript(text, "kawi"); }
}

export class Balinese extends ScriptTransliterator {
  protected readonly script = "balinese" as SupportedScript;
  static fromAksara(text: string): string { return fromScript(text, "balinese"); }
}

export class Sundanese extends ScriptTransliterator {
  protected readonly script = "sundanese" as SupportedScript;
  static fromAksara(text: string): string { return fromScript(text, "sundanese"); }
}

export class Sasak extends ScriptTransliterator {
  protected readonly script = "sasak" as SupportedScript;
  static fromAksara(text: string): string { return fromScript(text, "sasak"); }
}

export class Buginese extends ScriptTransliterator {
  protected readonly script = "buginese" as SupportedScript;
  static fromAksara(text: string): string { return fromScript(text, "buginese"); }
}
