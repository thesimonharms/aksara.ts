import { Aksara } from '../src/aksara';
import { Segmenter } from '../src/segmenter';

const segmenter = await Segmenter.load();

// ── Reverse: Aksara → Latin → segmented ─────────────────────────────────────

const samples = [
    'ꦭꦩꦸꦤ꧀ꦱꦶꦫꦔꦶꦔꦸꦏꦸꦕꦶꦁ',
    'ꦲꦮꦏ꧀ꦏꦺꦲꦶꦉꦁꦱꦢꦪ',
    'ꦭꦩ꧀ꦧꦸꦁꦏꦶꦮꦠꦺꦩ꧀ꦧꦺꦴꦁꦥꦸꦠꦶꦃ',
    'ꦪꦺꦤ꧀ꦧꦸꦟ꧀ꦝꦼꦭ꧀ꦭꦁꦏꦸꦁꦲꦸꦠꦩ',
];

console.log('Aksara → Latin → segmented\n');
for (const aksara of samples) {
    const latin     = Aksara.fromAksara(aksara);
    const segmented = await segmenter.segment(latin);
    console.log(`  aksara   : ${aksara}`);
    console.log(`  latin    : ${latin}`);
    console.log(`  segmented: ${segmented}`);
    console.log();
}

// ── Forward: Latin → Aksara ──────────────────────────────────────────────────

console.log('Latin → Aksara\n');
const words = ['lamun', 'sira', 'nginguk', 'ucing'];
for (const word of words) {
    console.log(`  ${word.padEnd(10)} → ${new Aksara(word).getAksara()}`);
}
