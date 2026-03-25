#!/usr/bin/env node
// training/transliterate_corpus.js
//
// Transliterates a plain-text Latin Javanese file to Aksara Jawa Unicode,
// ready to use as a corpus for --mode generate_from_corpus and --mode train_lm.
//
// Usage:
//   node transliterate_corpus.js <input.txt> <output.txt>
//
// Example:
//   node transliterate_corpus.js javanese_wiki.txt javanese_aksara.txt

"use strict";

const fs   = require("fs");
const path = require("path");
const { Aksara } = require("../dist/aksara.js");

const [,, inputPath, outputPath] = process.argv;

if (!inputPath || !outputPath) {
  console.error("Usage: node transliterate_corpus.js <input.txt> <output.txt>");
  process.exit(1);
}

const inputAbs  = path.resolve(inputPath);
const outputAbs = path.resolve(outputPath);

if (!fs.existsSync(inputAbs)) {
  console.error(`Input file not found: ${inputAbs}`);
  process.exit(1);
}

const raw   = fs.readFileSync(inputAbs, "utf-8");
const lines = raw.split("\n");

console.log(`Input : ${inputAbs}`);
console.log(`Lines : ${lines.length.toLocaleString()}`);
console.log("Transliterating…");

const out    = fs.createWriteStream(outputAbs, { encoding: "utf-8" });
let written  = 0;   // lines with at least one Javanese character
let total    = 0;   // total Aksara characters written
const JAVANESE_RE = /[\uA980-\uA9DF]/;

for (let i = 0; i < lines.length; i++) {
  const line = lines[i].trim();
  if (!line) continue;

  let aksara;
  try {
    aksara = new Aksara(line, /* spaces= */ false).getAksara();
  } catch {
    // Skip lines that fail (stray markup, URLs, etc.)
    continue;
  }

  // Only keep lines that produced actual Javanese characters
  if (!JAVANESE_RE.test(aksara)) continue;

  out.write(aksara + "\n");
  written++;
  total += aksara.length;

  if ((i + 1) % 50_000 === 0) {
    process.stdout.write(`  ${(i + 1).toLocaleString()} / ${lines.length.toLocaleString()} lines processed…\r`);
  }
}

out.end();

console.log(`\nDone.`);
console.log(`Output       : ${outputAbs}`);
console.log(`Lines written: ${written.toLocaleString()}`);
console.log(`Total chars  : ${total.toLocaleString()} Javanese characters`);
console.log();
console.log("Next steps:");
console.log("  # Train language model");
console.log(`  python javanese_ocr.py --mode train_lm --corpus ${outputPath} --output_path javanese_lm.pkl`);
console.log();
console.log("  # Generate corpus-based training images (with manuscript backgrounds)");
console.log(`  python javanese_ocr.py --mode generate_from_corpus \\`);
console.log(`      --corpus ${outputPath} --background_pdf PDFA.pdf \\`);
console.log(`      --data_dir ./ocr_corpus --num_samples 5000`);
