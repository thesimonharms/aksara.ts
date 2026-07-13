import { describe, expect, test } from "bun:test";
import { unlinkSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { CharNgramLM } from "../src/lm";
import { OcrModel } from "../src/ocr";

describe("CharNgramLM", () => {
  test("should train and calculate log probabilities with Laplace smoothing", () => {
    const lm = new CharNgramLM(3, 0.1);
    lm.train("ꦲꦤꦕꦫꦏꦲꦤꦕꦫꦏ");

    const probSeen = lm.logProb("ꦲꦤ", "ꦕ");
    const probUnseen = lm.logProb("ꦲꦤ", "ꦢ");

    expect(probSeen).toBeGreaterThan(probUnseen);
    expect(Number.isFinite(probSeen)).toBeTrue();
    expect(Number.isFinite(probUnseen)).toBeTrue();
  });

  test("should serialize and deserialize cleanly via JSON and file save/load", () => {
    const lm = new CharNgramLM(3, 0.2);
    lm.train("ꦲꦤꦕꦫꦏ");

    const json = lm.toJSON();
    const loadedFromJSON = CharNgramLM.fromJSON(json);
    expect(loadedFromJSON.n).toBe(3);
    expect(loadedFromJSON.smoothing).toBe(0.2);
    expect(loadedFromJSON.logProb("ꦲꦤ", "ꦕ")).toBeCloseTo(
      lm.logProb("ꦲꦤ", "ꦕ"),
      6,
    );

    const tmpPath = join(tmpdir(), `test-lm-${Date.now()}.json`);
    lm.save(tmpPath);
    const loadedFromFile = CharNgramLM.load(tmpPath);
    expect(loadedFromFile.logProb("ꦲꦤ", "ꦕ")).toBeCloseTo(
      lm.logProb("ꦲꦤ", "ꦕ"),
      6,
    );
    unlinkSync(tmpPath);
  });
});

describe("OcrModel", () => {
  test("should load OcrModel from canonical ONNX model path", async () => {
    const model = await OcrModel.load();
    expect(model).toBeInstanceOf(OcrModel);
  });

  test("should recognize synthetic ImageData with greedy decoding", async () => {
    const model = await OcrModel.load();
    const imgData = {
      width: 128,
      height: 32,
      data: new Uint8Array(128 * 32 * 4).fill(255),
    };

    const result = await model.recognize(imgData);
    expect(result).toHaveProperty("text");
    expect(typeof result.text).toBe("string");
    expect(typeof result.confidence).toBe("number");
    expect(result.tiles.length).toBe(1);
  });

  test("should recognize synthetic ImageData with LM beam search decoding", async () => {
    const model = await OcrModel.load();
    const lm = new CharNgramLM(3, 0.1);
    lm.train("ꦲꦤꦕꦫꦏ");

    const imgData = {
      width: 128,
      height: 32,
      data: new Uint8Array(128 * 32 * 4).fill(255),
    };

    const result = await model.recognize(imgData, {
      lm,
      beamWidth: 5,
      lmWeight: 0.3,
    });
    expect(result).toHaveProperty("text");
    expect(typeof result.text).toBe("string");
    expect(typeof result.confidence).toBe("number");
    expect(result.tiles.length).toBe(1);
  });
  test("should recognize variable-width ImageData using fullLine option", async () => {
    const model = await OcrModel.load();
    const imgData = {
      width: 256,
      height: 32,
      data: new Uint8Array(256 * 32 * 4).fill(255),
    };

    const result = await model.recognize(imgData, {
      fullLine: true,
    });
    expect(result).toHaveProperty("text");
    expect(typeof result.text).toBe("string");
    expect(typeof result.confidence).toBe("number");
    expect(result.tiles.length).toBe(1);
  });
});
