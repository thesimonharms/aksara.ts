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
    expect(aksara.getAksara()).toBe("ꦡ");
  });
});