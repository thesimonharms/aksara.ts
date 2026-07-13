import { readFileSync, writeFileSync } from "fs";

export interface CharNgramLMJSON {
  n: number;
  smoothing: number;
  vocab: string[];
  counts: Record<string, Record<string, number>>;
}

/**
 * Character-level n-gram language model with Laplace smoothing.
 * Mirrors Python `CharNgramLM` in `training/javanese_ocr.py`.
 */
export class CharNgramLM {
  readonly n: number;
  readonly smoothing: number;
  private counts = new Map<string, Map<string, number>>();
  private vocab = new Set<string>();

  constructor(n = 3, smoothing = 0.1) {
    this.n = n;
    this.smoothing = smoothing;
  }

  /**
   * Train n-gram counts from input text.
   */
  train(text: string): void {
    const chars = [...text];
    for (const ch of chars) {
      this.vocab.add(ch);
    }

    const pad = "\x00".repeat(Math.max(this.n - 1, 0));
    const padded = [...pad, ...chars];

    for (let i = 0; i <= padded.length - this.n; i++) {
      const ctx = padded.slice(i, i + this.n - 1).join("");
      const ch = padded[i + this.n - 1];

      let ctxMap = this.counts.get(ctx);
      if (!ctxMap) {
        ctxMap = new Map<string, number>();
        this.counts.set(ctx, ctxMap);
      }
      ctxMap.set(ch, (ctxMap.get(ch) ?? 0) + 1);
    }
  }

  /**
   * Laplace-smoothed log P(char | last n-1 chars of prefix).
   */
  logProb(prefix: string, char: string): number {
    const padCount = Math.max(this.n - 1, 0);
    const prefixChars = [...prefix];
    let ctxChars: string[];
    if (prefixChars.length >= padCount) {
      ctxChars = prefixChars.slice(prefixChars.length - padCount);
    } else {
      const pad = Array<string>(padCount - prefixChars.length).fill("\x00");
      ctxChars = [...pad, ...prefixChars];
    }
    const ctx = ctxChars.join("");

    const ctxMap = this.counts.get(ctx);
    const charCount = ctxMap?.get(char) ?? 0;
    let totalCount = 0;
    if (ctxMap) {
      for (const count of ctxMap.values()) {
        totalCount += count;
      }
    }

    const vocabSize = Math.max(this.vocab.size, 1);
    const total = totalCount + this.smoothing * vocabSize;
    return Math.log((charCount + this.smoothing) / total);
  }

  toJSON(): CharNgramLMJSON {
    const countsRecord: Record<string, Record<string, number>> = {};
    for (const [ctx, map] of this.counts.entries()) {
      const inner: Record<string, number> = {};
      for (const [ch, c] of map.entries()) {
        inner[ch] = c;
      }
      countsRecord[ctx] = inner;
    }

    return {
      n: this.n,
      smoothing: this.smoothing,
      vocab: Array.from(this.vocab),
      counts: countsRecord,
    };
  }

  static fromJSON(data: CharNgramLMJSON): CharNgramLM {
    const lm = new CharNgramLM(data.n, data.smoothing);
    for (const ch of data.vocab) {
      lm.vocab.add(ch);
    }
    for (const [ctx, map] of Object.entries(data.counts)) {
      const ctxMap = new Map<string, number>();
      for (const [ch, c] of Object.entries(map)) {
        ctxMap.set(ch, c);
      }
      lm.counts.set(ctx, ctxMap);
    }
    return lm;
  }

  save(path: string): void {
    writeFileSync(path, JSON.stringify(this.toJSON()), "utf-8");
  }

  static load(path: string): CharNgramLM {
    const content = readFileSync(path, "utf-8");
    return CharNgramLM.fromJSON(JSON.parse(content) as CharNgramLMJSON);
  }
}
