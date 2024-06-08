# Aksara.ts

This is Aksara.ts , a project that currently is doing Javanese script to Latin script, and hopefully in the future includes some other indigenous scripts of Indonesia. Currently, only words without complex vowels will work (ex. CV only words or V initial words). I am planning on implementing someway to combine the elements at the unicode level though.

## How to use
```bash
npm install aksara-ts
```

I haven't actually packaged an NPM package yet though...

```typescript
// Import Aksara
import { Aksara } from "aksara-ts";

// Create an Aksara object and save the converted text as carakan.
const name = new Aksara("Simon");
const carakan = name.getAksara(); // Simon -> ꦱꦶꦩꦺꦴꦤ꧀
```