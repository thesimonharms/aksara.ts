import * as ort from "onnxruntime-node";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { PNG } from "pngjs";
import { CharNgramLM } from "./lm";

declare const __dirname: string;

const DEFAULT_MODEL_PATH = join(__dirname, "../model/javanese_ocr.onnx");
const DEFAULT_VOCAB_PATH = join(__dirname, "../model/ocr_vocab.json");

export const DEFAULT_OCR_ALPHABET = [
  "[blank]",
  ...Array.from({ length: 50 }, (_, i) => String.fromCharCode(0xA98F + i)),
];

export interface ImageDataLike {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray | Buffer | number[];
}

export interface OcrRecognizeOptions {
  lm?: CharNgramLM;
  beamWidth?: number;
  lmWeight?: number;
  tileWidth?: number;
  tileHeight?: number;
}

export interface OcrTileResult {
  text: string;
  confidence: number;
}

export interface OcrRecognizeResult {
  text: string;
  confidence: number;
  tiles: OcrTileResult[];
}

function logAddExp(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const max = Math.max(a, b);
  return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
}

function resizeGreyscale(
  src: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Float32Array {
  const dst = new Float32Array(dstW * dstH);
  for (let dy = 0; dy < dstH; dy++) {
    const sy = (dy + 0.5) * (srcH / dstH) - 0.5;
    const y0 = Math.max(0, Math.min(srcH - 1, Math.floor(sy)));
    const y1 = Math.max(0, Math.min(srcH - 1, y0 + 1));
    const wy = Math.max(0, Math.min(1, sy - y0));

    for (let dx = 0; dx < dstW; dx++) {
      const sx = (dx + 0.5) * (srcW / dstW) - 0.5;
      const x0 = Math.max(0, Math.min(srcW - 1, Math.floor(sx)));
      const x1 = Math.max(0, Math.min(srcW - 1, x0 + 1));
      const wx = Math.max(0, Math.min(1, sx - x0));

      const v00 = src[y0 * srcW + x0];
      const v10 = src[y0 * srcW + x1];
      const v01 = src[y1 * srcW + x0];
      const v11 = src[y1 * srcW + x1];

      const top = v00 * (1 - wx) + v10 * wx;
      const bot = v01 * (1 - wx) + v11 * wx;
      dst[dy * dstW + dx] = top * (1 - wy) + bot * wy;
    }
  }
  return dst;
}

export class OcrModel {
  private session: ort.InferenceSession;
  private alphabet: string[];

  private constructor(session: ort.InferenceSession, alphabet: string[]) {
    this.session = session;
    this.alphabet = alphabet;
  }

  static async load(
    modelPath = DEFAULT_MODEL_PATH,
    vocabPath = DEFAULT_VOCAB_PATH,
  ): Promise<OcrModel> {
    const session = await ort.InferenceSession.create(modelPath);
    let alphabet = DEFAULT_OCR_ALPHABET;
    if (vocabPath && existsSync(vocabPath)) {
      alphabet = JSON.parse(readFileSync(vocabPath, "utf-8")) as string[];
    }
    return new OcrModel(session, alphabet);
  }

  /**
   * Preprocess and recognize Aksara Javanese text from an image line strip.
   */
  async recognize(
    image: ImageDataLike | Buffer | Uint8Array,
    options: OcrRecognizeOptions = {},
  ): Promise<OcrRecognizeResult> {
    const tileWidth = options.tileWidth ?? 128;
    const tileHeight = options.tileHeight ?? 32;
    const lm = options.lm;
    const beamWidth = options.beamWidth ?? (lm ? 10 : 1);
    const lmWeight = options.lmWeight ?? 0.5;

    const greyscale = this.toGreyscale(image);
    const tiles = this.tileImage(greyscale, tileWidth, tileHeight);

    const tileResults: OcrTileResult[] = [];
    for (const tile of tiles) {
      const result = await this.recognizeTile(tile, tileWidth, tileHeight, {
        lm,
        beamWidth,
        lmWeight,
      });
      tileResults.push(result);
    }

    const fullText = tileResults.map((t) => t.text).join("");
    const meanConfidence =
      tileResults.length > 0
        ? tileResults.reduce((sum, t) => sum + t.confidence, 0) /
          tileResults.length
        : 0;

    return {
      text: fullText,
      confidence: meanConfidence,
      tiles: tileResults,
    };
  }

  private toGreyscale(
    image: ImageDataLike | Buffer | Uint8Array,
  ): { width: number; height: number; data: Float32Array } {
    if (
      (Buffer.isBuffer(image) || image instanceof Uint8Array) &&
      this.isPng(image)
    ) {
      const png = PNG.sync.read(
        Buffer.isBuffer(image) ? image : Buffer.from(image),
      );
      return this.extractGreyscale(png.width, png.height, png.data, 4);
    }

    if (
      typeof image === "object" &&
      "width" in image &&
      "height" in image &&
      "data" in image
    ) {
      const width = image.width;
      const height = image.height;
      const raw = image.data;
      const totalPixels = width * height;
      const channels = Math.max(1, Math.round(raw.length / totalPixels));
      return this.extractGreyscale(width, height, raw, channels);
    }

    throw new TypeError(
      "Unsupported image input. Provide ImageDataLike { width, height, data } or a valid PNG Buffer.",
    );
  }

  private isPng(buf: Uint8Array): boolean {
    return (
      buf.length >= 8 &&
      buf[0] === 0x89 &&
      buf[1] === 0x50 &&
      buf[2] === 0x4e &&
      buf[3] === 0x47
    );
  }

  private extractGreyscale(
    width: number,
    height: number,
    raw: ArrayLike<number>,
    channels: number,
  ): { width: number; height: number; data: Float32Array } {
    const data = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const idx = i * channels;
      if (channels >= 3) {
        const r = raw[idx] ?? 0;
        const g = raw[idx + 1] ?? 0;
        const b = raw[idx + 2] ?? 0;
        data[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      } else {
        data[i] = raw[idx] ?? 0;
      }
    }
    return { width, height, data };
  }

  private tileImage(
    greyscale: { width: number; height: number; data: Float32Array },
    tileWidth: number,
    tileHeight: number,
  ): Float32Array[] {
    const scale = tileHeight / greyscale.height;
    const scaledWidth = Math.max(tileWidth, Math.round(greyscale.width * scale));
    const scaled = resizeGreyscale(
      greyscale.data,
      greyscale.width,
      greyscale.height,
      scaledWidth,
      tileHeight,
    );

    const tiles: Float32Array[] = [];
    for (let x = 0; x < scaledWidth; x += tileWidth) {
      const tile = new Float32Array(tileWidth * tileHeight);
      tile.fill(255.0); // white padding

      const copyWidth = Math.min(tileWidth, scaledWidth - x);
      for (let y = 0; y < tileHeight; y++) {
        for (let dx = 0; dx < copyWidth; dx++) {
          tile[y * tileWidth + dx] = scaled[y * scaledWidth + (x + dx)];
        }
      }
      tiles.push(tile);
    }
    return tiles;
  }

  private async recognizeTile(
    tileData: Float32Array,
    tileWidth: number,
    tileHeight: number,
    options: {
      lm?: CharNgramLM;
      beamWidth: number;
      lmWeight: number;
    },
  ): Promise<OcrTileResult> {
    const normalized = new Float32Array(tileData.length);
    for (let i = 0; i < tileData.length; i++) {
      normalized[i] = tileData[i] / 255.0;
    }

    const inputTensor = new ort.Tensor("float32", normalized, [
      1,
      1,
      tileHeight,
      tileWidth,
    ]);
    const inputName = this.session.inputNames[0] ?? "image";
    const results = await this.session.run({ [inputName]: inputTensor });

    const outputName = this.session.outputNames[0] ?? "logits";
    const outputTensor = results[outputName];
    const logits = outputTensor.data as Float32Array;
    const dims = outputTensor.dims; // [1, T, NUM_CLASSES]
    const timeSteps = dims[1];
    const numClasses = dims[2];

    if (options.lm || options.beamWidth > 1) {
      return this.ctcBeamDecode(
        logits,
        timeSteps,
        numClasses,
        options.lm,
        options.beamWidth,
        options.lmWeight,
      );
    }
    return this.ctcGreedyDecode(logits, timeSteps, numClasses);
  }

  private ctcGreedyDecode(
    logits: Float32Array,
    timeSteps: number,
    numClasses: number,
  ): OcrTileResult {
    let totalMaxProb = 0;
    const decodedChars: string[] = [];
    let prevClass = -1;

    for (let t = 0; t < timeSteps; t++) {
      const offset = t * numClasses;
      let maxLogit = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        if (logits[offset + c] > maxLogit) {
          maxLogit = logits[offset + c];
        }
      }

      let sumExp = 0;
      let bestClass = 0;
      let bestProb = 0;
      for (let c = 0; c < numClasses; c++) {
        const prob = Math.exp(logits[offset + c] - maxLogit);
        sumExp += prob;
        if (prob > bestProb) {
          bestProb = prob;
          bestClass = c;
        }
      }

      const normalizedBestProb = bestProb / sumExp;
      totalMaxProb += normalizedBestProb;

      if (bestClass !== 0 && bestClass !== prevClass) {
        decodedChars.push(this.alphabet[bestClass] ?? "");
      }
      prevClass = bestClass;
    }

    const confidence = timeSteps > 0 ? totalMaxProb / timeSteps : 0;
    return {
      text: decodedChars.join(""),
      confidence,
    };
  }

  private ctcBeamDecode(
    logits: Float32Array,
    timeSteps: number,
    numClasses: number,
    lm: CharNgramLM | undefined,
    beamWidth: number,
    lmWeight: number,
  ): OcrTileResult {
    interface BeamEntry {
      pBlank: number;
      pNonBlank: number;
    }

    let beams = new Map<string, BeamEntry>();
    beams.set("\x00", { pBlank: 0.0, pNonBlank: -Infinity });

    const getPrefixStr = (key: string) => key.slice(0, -1);
    const getLastClass = (key: string) => key.charCodeAt(key.length - 1);

    for (let t = 0; t < timeSteps; t++) {
      const offset = t * numClasses;
      let maxLogit = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        if (logits[offset + c] > maxLogit) {
          maxLogit = logits[offset + c];
        }
      }

      let sumExp = 0;
      for (let c = 0; c < numClasses; c++) {
        sumExp += Math.exp(logits[offset + c] - maxLogit);
      }

      const logProbs = new Float32Array(numClasses);
      const logSumExp = maxLogit + Math.log(sumExp);
      for (let c = 0; c < numClasses; c++) {
        logProbs[c] = logits[offset + c] - logSumExp;
      }

      const nextBeams = new Map<string, BeamEntry>();
      const merge = (key: string, pB = -Infinity, pNb = -Infinity) => {
        let entry = nextBeams.get(key);
        if (!entry) {
          entry = { pBlank: -Infinity, pNonBlank: -Infinity };
          nextBeams.set(key, entry);
        }
        entry.pBlank = logAddExp(entry.pBlank, pB);
        entry.pNonBlank = logAddExp(entry.pNonBlank, pNb);
      };

      for (const [key, { pBlank, pNonBlank }] of beams.entries()) {
        const prefix = getPrefixStr(key);
        const lastClass = getLastClass(key);
        const pTotal = logAddExp(pBlank, pNonBlank);

        // Blank transition
        const blankKey = `${prefix}\x00`;
        merge(blankKey, pTotal + logProbs[0], -Infinity);

        // Non-blank transitions
        for (let c = 1; c < numClasses; c++) {
          const char = this.alphabet[c] ?? "";
          const ctcLp = logProbs[c];
          const lmLp = lm ? lm.logProb(prefix, char) * lmWeight : 0;
          const newPrefix = prefix + char;
          const newKey = `${newPrefix}${String.fromCharCode(c)}`;

          if (c === lastClass) {
            merge(newKey, -Infinity, pBlank + ctcLp + lmLp);
            const keepKey = `${prefix}${String.fromCharCode(c)}`;
            merge(keepKey, -Infinity, pNonBlank + ctcLp);
          } else {
            merge(newKey, -Infinity, pTotal + ctcLp + lmLp);
          }
        }
      }

      const sorted = Array.from(nextBeams.entries()).sort((a, b) => {
        const scoreA = logAddExp(a[1].pBlank, a[1].pNonBlank);
        const scoreB = logAddExp(b[1].pBlank, b[1].pNonBlank);
        return scoreB - scoreA;
      });

      beams = new Map(sorted.slice(0, beamWidth));
    }

    let bestKey = "";
    let bestScore = -Infinity;
    for (const [key, entry] of beams.entries()) {
      const score = logAddExp(entry.pBlank, entry.pNonBlank);
      if (score > bestScore) {
        bestScore = score;
        bestKey = key;
      }
    }

    const prefix = bestKey.length > 0 ? getPrefixStr(bestKey) : "";
    const len = Math.max(prefix.length, 1);
    const confidence = Math.exp(Math.max(-100, Math.min(0, bestScore / len)));

    return { text: prefix, confidence };
  }
}
