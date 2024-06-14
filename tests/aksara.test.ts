import {describe, expect, test} from '@jest/globals';
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
});