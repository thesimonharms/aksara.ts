import {describe, expect, test} from 'bun:test';
import { Aksara } from '../src/aksara';

describe('Aksara', () => {
  test('should create an instance of Aksara', () => {
    const aksara = new Aksara('hanacaraka');
    expect(aksara).toBeInstanceOf(Aksara);
  });

  test('should return the correct text', () => {
    const aksara = new Aksara('hanacaraka');
    expect(aksara.getText()).toBe('hanacaraka');
  });

  test('should return the correct aksara', () => {
    const aksara = new Aksara('hanacaraka');
    expect(aksara.getAksara()).toBe("ꦲꦤꦕꦫꦏ");
  });

  test('should return the correct aksara', () => {
    const aksara = new Aksara('aji saka');
    expect(aksara.getAksara()).toBe("ꦲꦗꦶꦱꦏ");
  });

  test('should return the correct aksara with spaces', () => {
    const aksara = new Aksara('aji saka', true);
    expect(aksara.getAksara()).toBe("ꦲꦗꦶ ꦱꦏ");
  });

  test('should be able to handle "th and dh")', () => {
    const aksara = new Aksara('tha dha');
    expect(aksara.getAksara()).toBe("ꦡꦣ");
  });

  test('should be able to handle "ng and ny")', () => {
    const aksara = new Aksara('nga nya');
    expect(aksara.getAksara()).toBe("ꦔꦚ");
  });

  test('should be able to handle longer words that are more complicated', () => {
    const aksara = new Aksara('ngaladhu');
    expect(aksara.getAksara()).toBe("ꦔꦭꦣꦸ");
  });

  test('Ng should be identifiable as a diacritic consonant', () => {
    const aksara = new Aksara('');
    expect(aksara.isDiacriticConsonant('ng')).toBe(true);
  });

  test('R should be identifiable as a diacritic consonant', () => {
    const aksara = new Aksara('');
    expect(aksara.isDiacriticConsonant('r')).toBe(true);
  });

  test('H should be identifiable as a diacritic consonant', () => {
    const aksara = new Aksara('');
    expect(aksara.isDiacriticConsonant('h')).toBe(true);
  });

  test('M should be identifiable as a diacritic consonant', () => {
    const aksara = new Aksara('');
    expect(aksara.isDiacriticConsonant('m')).toBe(true);
  });

  test('should return the correct aksara for wong jawa', () => {
    const aksara = new Aksara('wong jawa');
    expect(aksara.getAksara()).toBe("ꦮꦺꦴꦁꦗꦮ");
  });

  test('should handle consonant final words that are not diacritic consonants', () => {
    const aksara = new Aksara('orak');
    expect(aksara.getAksara()).toBe("ꦲꦺꦴꦫꦏ꧀");
  });

  test('should handle consonant final words that are diacritic consonants', () => {
    const aksara = new Aksara('ruwing');
    expect(aksara.getAksara()).toBe("ꦫꦸꦮꦶꦁ");
  });

  test('should handle words with wyanjana', () => {
    const aksara = new Aksara('wyanjana');
    expect(aksara.getAksara()).toBe("ꦮ꧀ꦪꦤ꧀ꦗꦤ");
  });

  test('all-uppercase input should produce the same result as lowercase', () => {
    const lower = new Aksara('hanacaraka');
    const upper = new Aksara('HANACARAKA');
    expect(upper.getAksara()).toBe(lower.getAksara());
  });

  test('mixed-case input should produce the same result as lowercase', () => {
    const lower = new Aksara('simon');
    const mixed = new Aksara('Simon');
    expect(mixed.getAksara()).toBe(lower.getAksara());
  });

  test('getAksara() should return the same result when called a second time', () => {
    const aksara = new Aksara('hanacaraka');
    const first = aksara.getAksara();
    const second = aksara.getAksara();
    expect(second).toBe(first);
  });

  test('getAksara() should return the same result when called many times', () => {
    const aksara = new Aksara('aji saka');
    aksara.getAksara();
    aksara.getAksara();
    expect(aksara.getAksara()).toBe("ꦲꦗꦶꦱꦏ");
  });

  test('standalone e should use the pepet diacritic (ꦲꦼ)', () => {
    const aksara = new Aksara('e');
    expect(aksara.getAksara()).toBe('ꦲꦼ');
  });

  test('standalone é should use the taling diacritic (ꦲꦺ)', () => {
    const aksara = new Aksara('é');
    expect(aksara.getAksara()).toBe('ꦲꦺ');
  });

  test('consonant + e should use the pepet diacritic (ꦏꦼ)', () => {
    const aksara = new Aksara('ke');
    expect(aksara.getAksara()).toBe('ꦏꦼ');
  });

  test('consonant + é should use the taling diacritic (ꦏꦺ)', () => {
    const aksara = new Aksara('ké');
    expect(aksara.getAksara()).toBe('ꦏꦺ');
  });

  test('word ending with a digraph should resolve nextToken correctly', () => {
    const aksara = new Aksara('ngung');
    expect(aksara.getAksara()).toBe('ꦔꦸꦁ');
  });

  test('should handle triple consonant clusters', () => {
    const aksara = new Aksara('stra');
    expect(aksara.getAksara()).toBe('ꦱ꧀ꦠꦿ');
  });

  test('should handle triple consonant clusters mid-word', () => {
    const aksara = new Aksara('astra');
    expect(aksara.getAksara()).toBe('ꦲꦱ꧀ꦠꦿ');
  });

  test('digits should be converted to Javanese numerals', () => {
    expect(new Aksara('1').getAksara()).toBe('꧑');
    expect(new Aksara('0').getAksara()).toBe('꧐');
    expect(new Aksara('9').getAksara()).toBe('꧙');
    expect(new Aksara('1234567890').getAksara()).toBe('꧑꧒꧓꧔꧕꧖꧗꧘꧙꧐');
  });

  test('punctuation should pass through unchanged', () => {
    const aksara = new Aksara('aji, saka.', true);
    expect(aksara.getAksara()).toBe('ꦲꦗꦶ꧈ ꦱꦏ꧉');
  });

  test('_ is an explicit syllable break that produces no output', () => {
    expect(new Aksara('a_ka').getAksara()).toBe('ꦲꦏ');
  });

  test('_ forces a diacritic consonant into onset cluster position', () => {
    // Without break: ng is treated as a coda diacritic of 'a'
    expect(new Aksara('angkra').getAksara()).toBe('ꦲꦁꦏꦿ');
    // With break: 'a' is an open syllable; ngkra forms a full onset cluster
    expect(new Aksara('a_ngkra').getAksara()).toBe('ꦲꦔ꧀ꦏꦿ');
  });

  test('_ allows multi-consonant onset clusters starting with r', () => {
    expect(new Aksara('a_rsa').getAksara()).toBe('ꦲꦫ꧀ꦱ');
  });

  test('_ works mid-word between two consonant clusters', () => {
    expect(new Aksara('ang_kra').getAksara()).toBe('ꦲꦁꦏꦿ');
  });

  test('th in wyanjana position should use the same aksara as th in initial position', () => {
    const initial = new Aksara('tha').getAksara();     // ꦡ as initial
    const wyanjana = new Aksara('ntha').getAksara();   // ꦤ + pangkon + ꦡ
    expect(wyanjana).toBe('ꦤ' + '꧀' + initial);
  });

  test('dh in wyanjana position should use the same aksara as dh in initial position', () => {
    const initial = new Aksara('dha').getAksara();     // ꦣ as initial
    const wyanjana = new Aksara('ndha').getAksara();   // ꦤ + pangkon + ꦣ
    expect(wyanjana).toBe('ꦤ' + '꧀' + initial);
  });

  test('th wyanjana works in post-vowel cluster position', () => {
    const initial = new Aksara('tha').getAksara();
    const wyanjana = new Aksara('antha').getAksara();  // ꦲꦤ + pangkon + ꦡ
    expect(wyanjana).toBe('ꦲꦤ' + '꧀' + initial);
  });

  test('dh wyanjana works in post-vowel cluster position', () => {
    const initial = new Aksara('dha').getAksara();
    const wyanjana = new Aksara('andha').getAksara();  // ꦲꦤ + pangkon + ꦣ
    expect(wyanjana).toBe('ꦲꦤ' + '꧀' + initial);
  });

  // Fix 1: word-final consonant in wyanjana chain missing trailing pangkon
  test('word-final wyanjana consonant should get trailing pangkon', () => {
    expect(new Aksara('aks').getAksara()).toBe('ꦲꦏ꧀ꦱ꧀');
    expect(new Aksara('ants').getAksara()).toBe('ꦲꦤ꧀ꦠ꧀ꦱ꧀');
  });

  test('wyanjana consonant followed by a vowel should not get trailing pangkon', () => {
    expect(new Aksara('ntsa').getAksara()).toBe('ꦤ꧀ꦠ꧀ꦱ');
  });

  // Fix 2: word-initial consonant with no following vowel missing pangkon
  test('lone consonant with no following vowel should get pangkon', () => {
    expect(new Aksara('k').getAksara()).toBe('ꦏ꧀');
    expect(new Aksara('ng').getAksara()).toBe('ꦔ꧀');
  });

  // Fix 3: explicit aksara vowels
  test('explicit vowel mode uses the unique aksara vowel letters for standalone vowels', () => {
    expect(new Aksara('a', false, true).getAksara()).toBe('ꦄ');
    expect(new Aksara('i', false, true).getAksara()).toBe('ꦆ');
    expect(new Aksara('u', false, true).getAksara()).toBe('ꦈ');
    expect(new Aksara('e', false, true).getAksara()).toBe('ꦌ');
    expect(new Aksara('o', false, true).getAksara()).toBe('ꦎ');
  });

  test('explicit vowel mode: é falls back to ꦲꦺ (no standalone form exists)', () => {
    expect(new Aksara('é', false, true).getAksara()).toBe('ꦲꦺ');
  });

  test('explicit vowel mode: "aksara" initial vowel is explicit, mid-word vowels are unchanged', () => {
    expect(new Aksara('aksara', false, true).getAksara()).toBe('ꦄꦏ꧀ꦱꦫ');
    expect(new Aksara('aksara').getAksara()).toBe('ꦲꦏ꧀ꦱꦫ');
  });

  test('explicit vowel mode does not affect vowels that follow a consonant', () => {
    expect(new Aksara('ki', false, true).getAksara()).toBe('ꦏꦶ');
    expect(new Aksara('ku', false, true).getAksara()).toBe('ꦏꦸ');
  });

  // Fix 4: cakra (ꦿ) for medial r
  test('r in wyanjana position should use cakra (ꦿ)', () => {
    expect(new Aksara('kra').getAksara()).toBe('ꦏꦿ');
    expect(new Aksara('ngra').getAksara()).toBe('ꦔꦿ');
  });

  test('cakra combines correctly with vowel diacritics', () => {
    expect(new Aksara('kri').getAksara()).toBe('ꦏꦿꦶ');
    expect(new Aksara('kru').getAksara()).toBe('ꦏꦿꦸ');
  });

  test('word-final cakra gets trailing pangkon', () => {
    expect(new Aksara('kr').getAksara()).toBe('ꦏꦿ꧀');
  });

  test('initial r is still ꦫ (ra), not cakra', () => {
    expect(new Aksara('ra').getAksara()).toBe('ꦫ');
  });

  test('coda r is still ꦂ (layar), not cakra', () => {
    expect(new Aksara('ar').getAksara()).toBe('ꦲꦂ');
  });

  // Fix 5: Javanese punctuation
  test('period maps to pada lungsi (꧉)', () => {
    expect(new Aksara('.').getAksara()).toBe('꧉');
  });

  test('comma maps to pada lingsa (꧈)', () => {
    expect(new Aksara(',').getAksara()).toBe('꧈');
  });

  test('unknown punctuation passes through unchanged', () => {
    expect(new Aksara('?').getAksara()).toBe('?');
  });
});

describe('Aksara.fromAksara', () => {
  test('single consonant with inherent a', () => {
    expect(Aksara.fromAksara('ꦏ')).toBe('ka');
  });

  test('consonant + vowel diacritics', () => {
    expect(Aksara.fromAksara('ꦏꦶ')).toBe('ki');
    expect(Aksara.fromAksara('ꦏꦸ')).toBe('ku');
    expect(Aksara.fromAksara('ꦏꦼ')).toBe('ke');
  });

  test('taling alone is é', () => {
    expect(Aksara.fromAksara('ꦏꦺ')).toBe('ké');
  });

  test('taling + tarung is o', () => {
    expect(Aksara.fromAksara('ꦏꦺꦴ')).toBe('ko');
  });

  test('pangkon suppresses inherent a', () => {
    expect(Aksara.fromAksara('ꦏ꧀')).toBe('k');
  });

  test('coda diacritics decode correctly', () => {
    expect(Aksara.fromAksara('ꦁ')).toBe('ng');
    expect(Aksara.fromAksara('ꦀ')).toBe('m');
    expect(Aksara.fromAksara('ꦂ')).toBe('r');
    expect(Aksara.fromAksara('ꦃ')).toBe('h');
  });

  test('cakra decodes as medial r', () => {
    expect(Aksara.fromAksara('ꦏꦿ')).toBe('kra');
    expect(Aksara.fromAksara('ꦏꦿꦶ')).toBe('kri');
    expect(Aksara.fromAksara('ꦏꦿ꧀')).toBe('kr');
  });

  test('wyanjana (pangkon + consonant) decodes correctly', () => {
    expect(Aksara.fromAksara('ꦮ꧀ꦪꦤ꧀ꦗꦤ')).toBe('wyanjana');
  });

  test('roundtrip: hanacaraka', () => {
    expect(Aksara.fromAksara(new Aksara('hanacaraka').getAksara())).toBe('hanacaraka');
  });

  test('roundtrip: wong jawa with spaces', () => {
    expect(Aksara.fromAksara(new Aksara('wong jawa', true).getAksara())).toBe('wong jawa');
  });

  test('roundtrip: ruwing (coda diacritic ng)', () => {
    expect(Aksara.fromAksara(new Aksara('ruwing').getAksara())).toBe('ruwing');
  });

  test('roundtrip: orak (non-diacritic final consonant)', () => {
    expect(Aksara.fromAksara(new Aksara('orak').getAksara())).toBe('horak');
  });

  test('roundtrip: kra (cakra)', () => {
    expect(Aksara.fromAksara(new Aksara('kra').getAksara())).toBe('kra');
  });

  test('roundtrip: stra (triple cluster with cakra)', () => {
    expect(Aksara.fromAksara(new Aksara('stra').getAksara())).toBe('stra');
  });

  test('ꦲ decodes as h (the h+vowel ambiguity)', () => {
    expect(Aksara.fromAksara('ꦲꦗꦶ')).toBe('haji');
  });

  test('spaces pass through', () => {
    expect(Aksara.fromAksara('ꦏ ꦭ')).toBe('ka la');
  });

  test('unknown characters pass through unchanged', () => {
    expect(Aksara.fromAksara('꧉')).toBe('꧉');
  });

  test('Na Murda (ꦟ) decodes as n', () => {
    expect(Aksara.fromAksara('ꦟ꧀')).toBe('n');
    expect(Aksara.fromAksara('ꦟꦸ')).toBe('nu');
  });

  test('Dda (ꦝ, retroflex d) decodes as ḍ', () => {
    expect(Aksara.fromAksara('ꦝꦼ')).toBe('ḍe');
    expect(Aksara.fromAksara('ꦝ꧀')).toBe('ḍ');
  });

  test('murda consonants decode to the same Latin as their base form', () => {
    expect(Aksara.fromAksara('ꦑ')).toBe('ka');   // Ka Murda
    expect(Aksara.fromAksara('ꦓ')).toBe('ga');   // Ga Murda
    expect(Aksara.fromAksara('ꦦ')).toBe('pa');   // Pa Murda
    expect(Aksara.fromAksara('ꦯ')).toBe('sa');   // Sa Murda
    expect(Aksara.fromAksara('ꦬ')).toBe('ra');   // Ra Agung
    expect(Aksara.fromAksara('ꦛ')).toBe('ṭa');   // Tta (retroflex t)
  });

  test('Re (ꦉ) decodes as standalone re', () => {
    expect(Aksara.fromAksara('ꦉ')).toBe('re');
  });

  test('Le (ꦊ) decodes as standalone le', () => {
    expect(Aksara.fromAksara('ꦊ')).toBe('le');
  });

  test('explicit standalone vowel letters decode correctly', () => {
    expect(Aksara.fromAksara('ꦄ')).toBe('a');
    expect(Aksara.fromAksara('ꦆ')).toBe('i');
    expect(Aksara.fromAksara('ꦈ')).toBe('u');
    expect(Aksara.fromAksara('ꦌ')).toBe('e');
    expect(Aksara.fromAksara('ꦎ')).toBe('o');
  });

  test('pengkal (ꦾ) is medial y, analogous to cakra for r', () => {
    expect(Aksara.fromAksara('ꦢꦾ')).toBe('dya');
    expect(Aksara.fromAksara('ꦢꦾꦸ')).toBe('dyu');
    expect(Aksara.fromAksara('ꦢꦾ꧀')).toBe('dy');
  });

  test('roundtrip fragment from manuscript: bunḍel', () => {
    // ꦧꦸꦟ꧀ꦝꦼꦭ꧀ = bu + na-murda(n) + dda+pepet(ḍe) + la+pangkon(l)
    expect(Aksara.fromAksara('ꦧꦸꦟ꧀ꦝꦼꦭ꧀')).toBe('bunḍel');
  });
});
