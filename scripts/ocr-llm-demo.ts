import { existsSync, readFileSync, unlinkSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { execSync } from "child_process";
import { OcrModel } from "../src/ocr";
import { Aksara } from "../src/aksara";
import { Segmenter } from "../src/segmenter";

function renderPdfFirstPageToPng(pdfPath: string): string {
  const tmpPng = join(
    tmpdir(),
    `aksara-pdf-${Date.now()}-${Math.random().toString(36).slice(2)}.png`,
  );

  const pythonCandidates = [
    join(__dirname, "../training/venv/Scripts/python.exe"),
    join(__dirname, "../training/venv/bin/python"),
    "python3",
    "python",
  ];

  let pyExe: string | undefined;
  for (const candidate of pythonCandidates) {
    try {
      execSync(`"${candidate}" -c "import fitz"`, { stdio: "ignore" });
      pyExe = candidate;
      break;
    } catch {
      // try next candidate
    }
  }

  if (!pyExe) {
    throw new Error(
      "PyMuPDF (fitz) is required to render PDF files. Please run training/setup.bat or install PyMuPDF.",
    );
  }

  const script = `
import fitz
doc = fitz.open(r'''${pdfPath}''')
page = doc[0]
pix = page.get_pixmap(dpi=150)
pix.save(r'''${tmpPng}''')
`;

  execSync(`"${pyExe}" -c "${script.trim().replace(/\r?\n/g, "; ")}"`, {
    stdio: "inherit",
  });

  if (!existsSync(tmpPng)) {
    throw new Error(`Failed to render PDF page to temporary PNG at ${tmpPng}`);
  }

  return tmpPng;
}

async function main() {
  const args = process.argv.slice(2);
  const inputPath = args[0] || join(__dirname, "../training/ocr_corpus/corpus_29999.png");

  if (!existsSync(inputPath)) {
    console.error(`Error: Input file not found at '${inputPath}'`);
    process.exit(1);
  }

  console.log(`=== Aksara.ts End-to-End OCR → LLM Pipeline Demo ===\n`);
  console.log(`Input source: ${inputPath}`);

  let imagePath = inputPath;
  let tmpCleanPath: string | undefined;

  if (inputPath.toLowerCase().endsWith(".pdf")) {
    console.log(`[0] Rendering PDF page 1 to PNG via PyMuPDF...`);
    imagePath = renderPdfFirstPageToPng(inputPath);
    tmpCleanPath = imagePath;
  }

  try {
    // [1] Rendered page -> OCR tiles -> Aksara strings
    console.log(`\n[1] Running Aksara OCR Model...`);
    const ocrModel = await OcrModel.load();
    const imageBuffer = readFileSync(imagePath);
    const ocrResult = await ocrModel.recognize(imageBuffer);

    console.log(`    Tiles processed : ${ocrResult.tiles.length}`);
    console.log(`    Mean confidence : ${(ocrResult.confidence * 100).toFixed(1)}%`);
    console.log(`    Aksara output   : ${ocrResult.text}`);

    // [2] Aksara strings -> fromAksara -> Latin strings
    console.log(`\n[2] Transliterating Aksara → Latin...`);
    const latinText = Aksara.fromAksara(ocrResult.text);
    console.log(`    Latin string    : ${latinText}`);

    // [3] Latin strings -> Segmenter -> segmented Javanese
    console.log(`\n[3] Segmenting Latin text into words...`);
    const segmenter = await Segmenter.load();
    const segmentedText = await segmenter.segment(latinText);
    console.log(`    Segmented output: ${segmentedText}`);

    // [4] Segmented Javanese -> "would be sent to an LLM"
    console.log(`\n[4] Preparing prompt payload for downstream LLM...`);
    const llmPayload = {
      source: inputPath,
      stage: "ready_for_llm",
      confidence: Number(ocrResult.confidence.toFixed(4)),
      raw_aksara: ocrResult.text,
      transliterated_latin: latinText,
      segmented_javanese: segmentedText,
      prompt: `Translate and summarize the following segmented Javanese manuscript text:\n\n"${segmentedText}"`,
    };

    console.log(JSON.stringify(llmPayload, null, 2));
  } finally {
    if (tmpCleanPath && existsSync(tmpCleanPath)) {
      try {
        unlinkSync(tmpCleanPath);
      } catch {
        // ignore cleanup error
      }
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
