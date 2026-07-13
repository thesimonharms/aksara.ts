import { existsSync, readFileSync } from "fs";
import { OcrModel } from "../src/ocr";
import { CharNgramLM } from "../src/lm";

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`
Aksara.ts — OCR CLI Demo

Usage:
  bun run scripts/ocr-demo.ts <path/to/image.png> [options]

Options:
  --lm <path>         Path to CharNgramLM JSON file for beam search reranking
  --beam <width>      Beam width for CTC decoding (default: 10 with LM, 1 without)
  --lm-weight <w>     LM contribution weight (default: 0.5)

Example:
  bun run scripts/ocr-demo.ts training/ocr_corpus/corpus_29999.png
`);
    process.exit(1);
  }

  const imagePath = args[0];
  if (!imagePath || !existsSync(imagePath)) {
    console.error(`Error: image file not found at '${imagePath ?? ""}'`);
    process.exit(1);
  }

  let lm: CharNgramLM | undefined;
  let beamWidth = 1;
  let lmWeight = 0.5;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === "--lm" && args[i + 1]) {
      const lmPath = args[++i]!;
      if (existsSync(lmPath)) {
        lm = CharNgramLM.load(lmPath);
        beamWidth = 10;
      } else {
        console.warn(`Warning: LM file not found at '${lmPath}'`);
      }
    } else if (args[i] === "--beam" && args[i + 1]) {
      beamWidth = parseInt(args[++i]!, 10) || 1;
    } else if (args[i] === "--lm-weight" && args[i + 1]) {
      lmWeight = parseFloat(args[++i]!) || 0.5;
    }
  }

  console.log(`Loading OCR model...`);
  const model = await OcrModel.load();

  console.log(`Reading image: ${imagePath}`);
  const imageBuffer = readFileSync(imagePath);

  const startTime = performance.now();
  const result = await model.recognize(imageBuffer, {
    lm,
    beamWidth,
    lmWeight,
  });
  const elapsed = (performance.now() - startTime).toFixed(1);

  console.log(`\n=== Aksara.ts OCR Result ===`);
  console.log(`Image       : ${imagePath}`);
  console.log(`Decoder     : ${lm || beamWidth > 1 ? `beam (w=${beamWidth})` : "greedy"}`);
  console.log(`Tiles       : ${result.tiles.length}`);
  console.log(`Confidence  : ${(result.confidence * 100).toFixed(1)}%`);
  console.log(`Elapsed     : ${elapsed} ms`);
  console.log(`----------------------------`);
  console.log(`Decoded Aksara:`);
  console.log(result.text);

  if (result.tiles.length > 1) {
    console.log(`\nPer-tile breakdown:`);
    result.tiles.forEach((tile, idx) => {
      console.log(
        `  Tile #${idx + 1} (${(tile.confidence * 100).toFixed(1)}%): ${tile.text}`,
      );
    });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
