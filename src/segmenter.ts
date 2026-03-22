import * as ort from 'onnxruntime-node';
import { readFileSync } from 'fs';

const THRESHOLD = 0.5;

export class Segmenter {
    private session:   ort.InferenceSession;
    private charToIdx: Map<string, number>;

    private constructor(session: ort.InferenceSession, vocab: string[]) {
        this.session   = session;
        this.charToIdx = new Map(vocab.map((c, i) => [c, i]));
    }

    static async load(
        modelPath = './model/segmenter.onnx',
        vocabPath = './model/vocab.json',
    ): Promise<Segmenter> {
        const session = await ort.InferenceSession.create(modelPath);
        const vocab   = JSON.parse(readFileSync(vocabPath, 'utf-8')) as string[];
        return new Segmenter(session, vocab);
    }

    async segment(text: string, threshold = THRESHOLD): Promise<string> {
        const chars = [...text.replace(/\s+/g, '')];
        if (chars.length === 0) return text;

        // onnxruntime-node requires BigInt64Array for int64 tensors
        const indices = BigInt64Array.from(
            chars.map(c => BigInt(this.charToIdx.get(c) ?? 1))
        );
        const input   = new ort.Tensor('int64', indices, [1, chars.length]);
        const results = await this.session.run({ input });

        // Model outputs raw logits — apply sigmoid manually
        const logits = results['logits'].data as Float32Array;

        let result = '';
        for (let i = 0; i < chars.length; i++) {
            result += chars[i];
            const prob = 1 / (1 + Math.exp(-logits[i]));
            if (prob > threshold && i < chars.length - 1) result += ' ';
        }
        return result;
    }
}
