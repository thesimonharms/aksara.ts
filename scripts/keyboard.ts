/**
 * keyboard.ts — Javanese script (Aksara Jawa) virtual keyboard
 * Run with: bun run keyboard
 * Opens a web keyboard at http://localhost:3000
 */

const HTML = `<!DOCTYPE html>
<html lang="jv">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Aksara Jawa Keyboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: system-ui, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    min-height: 100vh;
    padding: 20px;
  }

  h1 {
    text-align: center;
    font-size: 1.2rem;
    color: #a0a0c0;
    margin-bottom: 16px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  #output-box {
    background: #0d0d1a;
    border: 1px solid #3a3a5c;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    min-height: 80px;
    font-size: 2rem;
    line-height: 1.6;
    word-break: break-all;
    color: #f0f0ff;
    letter-spacing: 0.05em;
  }
  #output-box.empty::before {
    content: 'Compose Javanese text here\u2026';
    color: #444466;
    font-size: 1rem;
  }

  .actions {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
  }
  .actions button {
    flex: 1;
    padding: 10px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
    transition: opacity 0.1s;
  }
  .actions button:active { opacity: 0.7; }
  #btn-copy   { background: #2d7d46; color: #fff; }
  #btn-back   { background: #5a3a1a; color: #ffcc88; }
  #btn-clear  { background: #5a1a1a; color: #ff9999; }
  #btn-space  { background: #2a2a4a; color: #aaaacc; flex: 2; }

  .section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #5a5a7a;
    margin: 16px 0 6px;
  }

  .key-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 4px;
  }

  .key {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #2a2a4a;
    border: 1px solid #3a3a6a;
    border-radius: 6px;
    cursor: pointer;
    padding: 6px 4px 4px;
    min-width: 52px;
    transition: background 0.08s, transform 0.08s;
    user-select: none;
  }
  .key:hover  { background: #3a3a6a; }
  .key:active { background: #4a4a8a; transform: scale(0.93); }

  .key .aksara {
    font-size: 1.5rem;
    line-height: 1.4;
    color: #e8e8ff;
  }
  .key .roman {
    font-size: 0.6rem;
    color: #6a6a9a;
    margin-top: 2px;
    white-space: nowrap;
  }

  .key.diac {
    min-width: 44px;
    background: #1e2a3a;
    border-color: #2a4a6a;
  }
  .key.diac:hover  { background: #2a3a5a; }
  .key.diac:active { background: #3a4a7a; }
  .key.diac .aksara { color: #88ccff; }

  .key.arabic {
    background: #2a1a2a;
    border-color: #5a2a5a;
  }
  .key.arabic:hover { background: #3a2a3a; }
  .key.arabic .aksara { color: #ffaaff; }

  .key.pasangan {
    background: #1a2a1a;
    border-color: #2a5a2a;
  }
  .key.pasangan:hover { background: #2a3a2a; }
  .key.pasangan .aksara { color: #88ffaa; font-size: 1.2rem; }

  .key.digit { background: #2a1e3a; border-color: #4a2a6a; }
  .key.digit:hover { background: #3a2a5a; }
  .key.digit .aksara { color: #cc88ff; }

  #toast {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    background: #2d7d46;
    color: #fff;
    padding: 10px 24px;
    border-radius: 20px;
    font-size: 0.9rem;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
  }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<h1>Aksara Jawa Keyboard</h1>

<div id="output-box" class="empty"></div>

<div class="actions">
  <button id="btn-copy"  onclick="copyText()">Copy</button>
  <button id="btn-space" onclick="type('\u200B')">Space (ZWSP)</button>
  <button id="btn-back"  onclick="backspace()">&#9003;</button>
  <button id="btn-clear" onclick="clearText()">Clear</button>
</div>

<div class="section-label">Aksara (consonants — Hanacaraka order)</div>
<div class="key-grid" id="grid-aksara"></div>

<div class="section-label">Murda &amp; Mahaprana</div>
<div class="key-grid" id="grid-murda"></div>

<div class="section-label">Arabic-derived (cecak telu)</div>
<div class="key-grid" id="grid-arabic"></div>

<div class="section-label">Sandhangan (vowel diacritics)</div>
<div class="key-grid" id="grid-sandh"></div>

<div class="section-label">Finals &amp; marks</div>
<div class="key-grid" id="grid-marks"></div>

<div class="section-label">Pasangan (subscript consonants)</div>
<div class="key-grid" id="grid-pasangan"></div>

<div class="section-label">Digits (pada angka)</div>
<div class="key-grid" id="grid-digits"></div>

<div class="section-label">Punctuation</div>
<div class="key-grid" id="grid-punct"></div>

<div id="toast">Copied!</div>

<script>
let text = '';
const output = document.getElementById('output-box');

function render() {
  if (text === '') {
    output.textContent = '';
    output.classList.add('empty');
  } else {
    output.textContent = text;
    output.classList.remove('empty');
  }
}

function type(ch) { text += ch; render(); }

function backspace() {
  const arr = [...text];
  arr.pop();
  text = arr.join('');
  render();
}

function clearText() { text = ''; render(); }

function copyText() {
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const t = document.getElementById('toast');
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1800);
  });
}

// ── Character tables ────────────────────────────────────────────────────────

const AKSARA = [
  ['\uA9B2','ha'], ['\uA9A4','na'], ['\uA995','ca'], ['\uA9AB','ra'], ['\uA98F','ka'],
  ['\uA9A2','da'], ['\uA9A0','ta'], ['\uA9B1','sa'], ['\uA9AE','wa'], ['\uA9AD','la'],
  ['\uA9A5','pa'], ['\uA99D','dha'],  ['\uA997','ja'], ['\uA9AA','ya'], ['\uA99A','nya'],
  ['\uA9A9','ma'], ['\uA992','ga'], ['\uA9A7','ba'], ['\uA99B','tha'], ['\uA994','nga'],
];

const MURDA = [
  ['\uA991','ka murda'], ['\uA993','ga murda'], ['\uA996','ca murda'],
  ['\uA998','ja murda'], ['\uA99F','na murda'], ['\uA9A1','ta murda'],
  ['\uA9A6','pa murda'], ['\uA9A8','ba murda'], ['\uA9AF','sa murda'],
  ['\uA9B0','sa mhp'],   ['\uA9AC','ra agung'],
  ['\uA99E','da mhp'],   ['\uA999','ja mhp'],
  ['\uA99C','da m.mhp'], ['\uA9A3','da murda'],
];

// Cecak telu (U+A9B3) added to base aksara for Arabic loanwords
const ARABIC = [
  ['\uA9B2\uA9B3','kha'],
  ['\uA98F\uA9B3','qa'],
  ['\uA990','dza'],        // ka sasak — standalone char
  ['\uA9A2\uA9B3','sya'],
  ['\uA9B1\uA9B3','fa/va'],
  ['\uA9A5\uA9B3','za'],
  ['\uA997\uA9B3','gha'],
  ['\uA992\uA9B3','\u02BEa (ain)'],
  ['\uA994\uA9B3','panyangga'],
];

const SANDHANGAN = [
  ['\uA9B4','aa (tarung)'],
  ['\uA9B5','tolong'],
  ['\uA9B6','i (wulu)'],
  ['\uA9B7','ii (wulu melik)'],
  ['\uA9B8','u (suku)'],
  ['\uA9B9','uu (suku mendut)'],
  ['\uA9BA','e (taling)'],
  ['\uA9BB','o (taling tarung)'],
  ['\uA9BC','schwa (pepet)'],
  ['\uA9BD','keret'],
  ['\uA9BE','ya sub.'],
  ['\uA9BF','ra sub.'],
];

const MARKS = [
  ['\uA980','nasal'],
  ['\uA981','cecek (-ng)'],
  ['\uA982','layar (-r)'],
  ['\uA983','wignyan (-h)'],
  ['\uA9B3','cecak telu'],
  ['\uA9C0','pangkon'],
  ['\uA9CF','avagraha'],
];

// Pasangan = pangkon (U+A9C0) + aksara — renders as subscript form in font
const PANGKON = '\uA9C0';
const PASANGAN = [
  ['\uA9B2','ha'], ['\uA9A4','na'], ['\uA995','ca'], ['\uA9AB','ra'], ['\uA98F','ka'],
  ['\uA9A2','da'], ['\uA9A0','ta'], ['\uA9B1','sa'], ['\uA9AE','wa'], ['\uA9AD','la'],
  ['\uA9A5','pa'], ['\uA99D','dha'], ['\uA997','ja'], ['\uA9AA','ya'], ['\uA99A','nya'],
  ['\uA9A9','ma'], ['\uA992','ga'], ['\uA9A7','ba'], ['\uA99B','tha'], ['\uA994','nga'],
  ['\uA99F','na murda'], ['\uA996','ca murda'], ['\uA9AC','ra agung'], ['\uA991','ka murda'],
  ['\uA9A1','ta murda'], ['\uA9AF','sa murda'], ['\uA9A6','pa murda'], ['\uA998','ja murda'],
  ['\uA993','ga murda'], ['\uA9A8','ba murda'], ['\uA9A3','da murda'], ['\uA9B0','sa mhp'],
  ['\uA99E','da mhp'],   ['\uA999','ja mhp'],   ['\uA99C','da m.mhp'],
];

const DIGITS = [
  ['\uA9D0','0'],['\uA9D1','1'],['\uA9D2','2'],['\uA9D3','3'],['\uA9D4','4'],
  ['\uA9D5','5'],['\uA9D6','6'],['\uA9D7','7'],['\uA9D8','8'],['\uA9D9','9'],
];

const PUNCT = [
  ['\uA9C1','pada lingsa'],
  ['\uA9C2','pada lungsi'],
  ['\uA9C3','pada pangkat'],
  ['\uA9C4','pada wignyan'],
  ['\uA9C5','pada isen-isen'],
  ['\uA9C6','pada windu'],
  ['\uA9C7','pada pangrawit'],
  ['\uA9C8','pada sawiji'],
  ['\uA9C9','pada adeg'],
  ['\uA9CA','pada adeg adeg'],
  ['\uA9CB','pada piseleh'],
  ['\uA9CC','pada piseleh inv.'],
  ['\uA9CD','pada tirta tumetes'],
  ['\uA9DE','pada carik sepisan'],
  ['\uA9DF','pada carik kalih'],
];

function makeKey(ch, label, extraClass='') {
  const div = document.createElement('div');
  div.className = 'key' + (extraClass ? ' ' + extraClass : '');
  div.innerHTML = '<span class="aksara">' + ch + '</span><span class="roman">' + label + '</span>';
  div.addEventListener('click', () => type(ch));
  return div;
}

function buildGrid(id, chars, extraClass='') {
  const grid = document.getElementById(id);
  for (const [ch, label] of chars) {
    grid.appendChild(makeKey(ch, label, extraClass));
  }
}

function buildPasanganGrid() {
  const grid = document.getElementById('grid-pasangan');
  for (const [aksara, label] of PASANGAN) {
    const ch = PANGKON + aksara;
    const div = document.createElement('div');
    div.className = 'key pasangan';
    div.innerHTML = '<span class="aksara">' + ch + '</span><span class="roman">' + label + '</span>';
    div.addEventListener('click', () => type(ch));
    grid.appendChild(div);
  }
}

buildGrid('grid-aksara',  AKSARA);
buildGrid('grid-murda',   MURDA);
buildGrid('grid-arabic',  ARABIC,  'arabic');
buildGrid('grid-sandh',   SANDHANGAN, 'diac');
buildGrid('grid-marks',   MARKS,      'diac');
buildPasanganGrid();
buildGrid('grid-digits',  DIGITS,  'digit');
buildGrid('grid-punct',   PUNCT,   'digit');
</script>
</body>
</html>`;

const server = Bun.serve({
  port: 3000,
  fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === "/" || url.pathname === "/keyboard") {
      return new Response(HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }
    return new Response("Not found", { status: 404 });
  },
});

console.log(`Aksara Jawa keyboard running at http://localhost:${server.port}`);
console.log("Open that URL in your browser, compose text, then click Copy.");
console.log("Press Ctrl+C to stop.");
