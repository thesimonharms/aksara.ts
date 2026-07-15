import { describe, expect, test } from "bun:test";
import {
  Aksara,
  Balinese,
  Buginese,
  Kawi,
  Sasak,
  Sundanese,
  fromScript,
  toScript,
} from "../src/aksara";

describe("additional Indonesian scripts", () => {
  describe("Kawi", () => {
    test("uses Unicode Kawi consonants, vowel signs, and conjoiner", () => {
      expect(new Kawi("kawi").getAksara()).toBe("𑼒𑼮𑼶");
      expect(new Kawi("aksara").getAksara()).toBe("𑼄𑼒𑽂𑼱𑼬");
    });

    test("uses the Unicode composite dependent vowel for o", () => {
      // Unicode Core Specification table 17-8: o = U+11F3E U+11F34.
      expect(new Kawi("ko").getAksara()).toBe("𑼒𑼾𑼴");
    });

    test("round-trips astral-plane Kawi code points", () => {
      const encoded = new Kawi("aksara").getAksara();
      expect(Kawi.fromAksara(encoded)).toBe("aksara");
      expect(fromScript(encoded, "kawi")).toBe("aksara");
    });
  });

  describe("Balinese", () => {
    test("writes inherent vowels, dependent vowels, and adeg adeg", () => {
      expect(new Balinese("bali").getAksara()).toBe("ᬩᬮᬶ");
      expect(new Balinese("aksara").getAksara()).toBe("ᬅᬓ᭄ᬲᬭ");
    });

    test("uses cecek, surang, and bisah for common final consonants", () => {
      expect(new Balinese("wong").getAksara()).toBe("ᬯᭀᬂ");
      expect(new Balinese("kar").getAksara()).toBe("ᬓᬃ");
      expect(new Balinese("sah").getAksara()).toBe("ᬲᬄ");
    });

    test("round-trips Balinese text and preserves requested spaces", () => {
      const encoded = new Balinese("basa bali", true).getAksara();
      expect(encoded).toBe("ᬩᬲ ᬩᬮᬶ");
      expect(Balinese.fromAksara(encoded)).toBe("basa bali");
    });
  });

  describe("Sundanese", () => {
    test("uses modern pamaaeh rather than the Old Sundanese virama", () => {
      expect(new Sundanese("sunda").getAksara()).toBe("ᮞᮥᮔ᮪ᮓ");
      expect(new Sundanese("aksara").getAksara()).toBe("ᮃᮊ᮪ᮞᮛ");
    });

    test("supports all three Unicode medial consonant signs", () => {
      expect(new Sundanese("kriya").getAksara()).toBe("ᮊᮢᮤᮚ");
      expect(new Sundanese("klya").getAksara()).toBe("ᮊᮣᮚ");
      expect(new Sundanese("kya").getAksara()).toBe("ᮊᮡ");
    });

    test("distinguishes e, é, and eu", () => {
      expect(new Sundanese("ke ké keu", true).getAksara()).toBe(
        "ᮊᮨ ᮊᮦ ᮊᮩ",
      );
      expect(Sundanese.fromAksara("ᮊᮨ ᮊᮦ ᮊᮩ")).toBe("ke ké keu");
    });
  });

  describe("Sasak", () => {
    test("uses the Balinese-encoded Jejawan repertoire", () => {
      expect(new Sasak("sasak").getAksara()).toBe("ᬲᬲᬓ᭄");
      expect(Sasak.fromAksara("ᬲᬲᬓ᭄")).toBe("sasak");
    });

    test("supports the documented KAF SASAK q/glottal-stop spelling", () => {
      expect(new Sasak("quran").getAksara()).toBe("ᭅᬸᬭᬦ᭄");
      expect(Sasak.fromAksara("ᭅᬸᬭᬦ᭄")).toBe("quran");
    });
  });

  describe("Buginese (Lontara)", () => {
    test("writes the traditional spelling of lontara", () => {
      expect(new Buginese("lontara").getAksara()).toBe("ᨒᨚᨈᨑ");
    });

    test("uses LETTER A as the standalone vowel carrier", () => {
      expect(new Buginese("basa ugi", true).getAksara()).toBe("ᨅᨔ ᨕᨘᨁᨗ");
      expect(Buginese.fromAksara("ᨅᨔ ᨕᨘᨁᨗ")).toBe("basa ugi");
    });

    test("omits codas instead of inventing a non-Unicode virama", () => {
      // Unicode documents that sara, sara', and sarang are all ᨔᨑ.
      expect(new Buginese("sara").getAksara()).toBe("ᨔᨑ");
      expect(new Buginese("sarang").getAksara()).toBe("ᨔᨑ");
      expect(Buginese.fromAksara("ᨔᨑ")).toBe("sara");
    });
  });

  test("functional API selects a script", () => {
    expect(toScript("bali", "balinese")).toBe("ᬩᬮᬶ");
    expect(toScript("sunda", "sundanese")).toBe("ᮞᮥᮔ᮪ᮓ");
  });

  test("existing Javanese API and output remain unchanged", () => {
    expect(new Aksara("hanacaraka").getAksara()).toBe("ꦲꦤꦕꦫꦏ");
    expect(Aksara.fromAksara("ꦲꦤꦕꦫꦏ")).toBe("hanacaraka");
  });
});
