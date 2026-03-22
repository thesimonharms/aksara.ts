class Aksara {
    // Eventually becomes returned aksara jawa text
    hanacaraka: string;

    // Tokens
    tokens: string[] = [];

    // List of all vowels used in Javanese
    vowels: Set<string> = new Set(['a', 'i', 'u', 'e', 'é', 'o']);

    // List of all consonants used in Javanese
    consonants: Set<string> = new Set(['h', 'n', 'c', 'r', 'k', 'd', 't', 's', 'w', 'l', 'p', 'j', 'y', 'm', 'g', 'b', 'th', 'dh', 'ng', 'ny']);

    // List of consonants that as a final become a diacritic (or modifying symbol)
    diacriticConsonants: Set<string> = new Set(['r', 'h', 'm', 'ng']);

    // Pangkon variable used for sentence or word final consonants
    pangkon: string = '꧀';

    // The diacritics used for consonants in Javanese
    consonantDiacritics: { [key: string]: string } = {
        'ng': 'ꦁ',
        'm': 'ꦀ',
        'r': 'ꦂ',
        'h': 'ꦃ'
    };

    // The diacritics used for vowels in Javanese
    vowelDiacritics: { [key: string]: string } = {
        'a': '',
        'i': 'ꦶ',
        'u': 'ꦸ',
        'e': 'ꦼ',
        'é': 'ꦺ',
        'o': 'ꦴ'
    };

    // The initial consonants used in Javanese
    initialConsonantAksara: { [key: string]: string } = {
        'ng': 'ꦔ',
        'ny': 'ꦚ',
        'th': 'ꦡ',
        'dh': 'ꦣ',
        'h': 'ꦲ',
        'n': 'ꦤ',
        'c': 'ꦕ',
        'r': 'ꦫ',
        'k': 'ꦏ',
        'd': 'ꦢ',
        't': 'ꦠ',
        's': 'ꦱ',
        'w': 'ꦮ',
        'l': 'ꦭ',
        'p': 'ꦥ',
        'j': 'ꦗ',
        'y': 'ꦪ',
        'm': 'ꦩ',
        'g': 'ꦒ',
        'b': 'ꦧ',
    };

    // The consonants used when two consonants touch each other.
    // 'r' uses cakra (ꦿ) — a subscript diacritic specific to medial r — rather than pangkon+ra.
    wyanjanaAksara: { [key: string]: string } = {
        'ng': '꧀ꦔ',
        'ny': '꧀ꦚ',
        'th': '꧀ꦡ',
        'dh': '꧀ꦣ',
        'h': '꧀ꦲ',
        'n': '꧀ꦤ',
        'c': '꧀ꦕ',
        'r': 'ꦿ',
        'k': '꧀ꦏ',
        'd': '꧀ꦢ',
        't': '꧀ꦠ',
        's': '꧀ꦱ',
        'w': '꧀ꦮ',
        'l': '꧀ꦭ',
        'p': '꧀ꦥ',
        'j': '꧀ꦗ',
        'y': '꧀ꦪ',
        'm': '꧀ꦩ',
        'g': '꧀ꦒ',
        'b': '꧀ꦧ',
    };

    // Vowels used when a vowel is by itself without consonant (note these are all h+vowel)
    normalAksaraVowels: { [key: string]: string } = {
        'a': 'ꦲ',
        'i': 'ꦲꦶ',
        'u': 'ꦲꦸ',
        'e': 'ꦲꦼ',
        'é': 'ꦲꦺ',
        'o': 'ꦲꦺꦴ'
    };

    // Standalone vowel letters — unambiguously vowels, not h+vowel.
    // Use by passing explicitVowels=true to the constructor.
    // é has no standalone form in Javanese script; it falls back to h+taling (ꦲꦺ).
    explicitAksaraVowels: { [key: string]: string } = {
        'a': 'ꦄ',
        'i': 'ꦆ',
        'u': 'ꦈ',
        'e': 'ꦌ',
        'é': 'ꦲꦺ',
        'o': 'ꦎ'
    }

    // Javanese numerals
    javaneseNumerals: { [key: string]: string } = {
        '0': '꧐', '1': '꧑', '2': '꧒', '3': '꧓', '4': '꧔',
        '5': '꧕', '6': '꧖', '7': '꧗', '8': '꧘', '9': '꧙'
    };

    // Javanese punctuation marks
    javanesePunctuation: { [key: string]: string } = {
        '.': '꧉',
        ',': '꧈'
    };

    // The text to be converted to Aksara, and whether or not to include spaces.
    // Set explicitVowels=true to use standalone vowel letters (ꦄ ꦆ ꦈ ꦌ ꦎ) instead of
    // the h+vowel convention (ꦲ ꦲꦶ ꦲꦸ ...) for vowels that appear without a preceding consonant.
    constructor(public readonly text: string, public readonly spaces: boolean = false, public readonly explicitVowels: boolean = false) {
        this.text = text.toLowerCase();
        this.hanacaraka = '';
        this.spaces = spaces;
        this.tokens = this.tokenize();
    }

    // Function to get supplied text
    getText(): string {
        return this.text;
    }

    // Internally used function to check if a character is a vowel
    isVowel(char: string): boolean {
        return this.vowels.has(char);
    }

    // Internally used function to check if a character is a consonant
    isConsonant(char: string): boolean {
        return this.consonants.has(char);
    }

    // Internally used function to check if a character is a diacritic consonant (or if wyanjana should be used instead)
    isDiacriticConsonant(char: string): boolean {
        return this.diacriticConsonants.has(char);
    }

    /**
     * Checks if a given character and the next character form a two-character consonant.
     * @param char - The current character.
     * @param nextChar - The next character.
     * @returns A boolean indicating whether the two characters form a two-character consonant.
     */
    isTwoCharacterConsonant(char: string, nextChar: string): boolean {
        if (char === 'n' && (nextChar === 'g' || nextChar === 'y')) {
            return true;
        } else if (char === 'd' && nextChar === 'h') {
            return true;
        } else if (char === 't' && nextChar === 'h') {
            return true;
        } else {
            return false;
        }
    }

    tokenize(): string[] {
        let tokenizedText: string[] = [];
        let i = 0;

        while (i < this.text.length) {
            let char = this.text[i];
            let nextChar = (i + 1 < this.text.length) ? this.text[i + 1] : undefined;
            if (this.isVowel(char)) {
                tokenizedText.push(char);
                i++;
            } else if (this.isConsonant(char)) {
                if (nextChar && this.isTwoCharacterConsonant(char, nextChar)) {
                    tokenizedText.push(char + nextChar);
                    i += 2;
                } else {
                    tokenizedText.push(char);
                    i++;
                }
            } else if (char === ' ') {
                tokenizedText.push(' ');
                i++;
            } else {
                tokenizedText.push(char);
                i++;
            }
        }

        return tokenizedText;
    }

    getAksara(): string {
        this.hanacaraka = '';

        let wyanjanaFlag = false;
        let postVowelFlag = false;

        let i = 0;
        while (i < this.tokens.length) {
            let previousToken = (i - 1 >= 0) ? this.tokens[i - 1] : undefined;
            let token = this.tokens[i];
            let nextToken = (i + 1 < this.tokens.length) ? this.tokens[i + 1] : undefined;

            if (this.isVowel(token)) {
                if (previousToken && this.isConsonant(previousToken)) {
                    if (token !== 'o') {
                        this.hanacaraka += this.vowelDiacritics[token];
                        postVowelFlag = true;
                        wyanjanaFlag = false;
                    } else {
                        this.hanacaraka += this.vowelDiacritics['é'] + this.vowelDiacritics['o'];
                        postVowelFlag = true;
                        wyanjanaFlag = false;
                    }
                } else {
                    this.hanacaraka += this.explicitVowels
                        ? this.explicitAksaraVowels[token]
                        : this.normalAksaraVowels[token];
                    postVowelFlag = true;
                    wyanjanaFlag = false;
                }
            } else if (this.isConsonant(token)) {
                if (postVowelFlag === true && wyanjanaFlag === false) {
                    if (nextToken && this.isVowel(nextToken)) {
                        this.hanacaraka += this.initialConsonantAksara[token];
                        postVowelFlag = false;
                    } else if (nextToken && this.isConsonant(nextToken)) {
                        if (this.isDiacriticConsonant(token)) {
                            this.hanacaraka += this.consonantDiacritics[token];
                            postVowelFlag = false;
                        } else {
                            this.hanacaraka += this.initialConsonantAksara[token];
                            postVowelFlag = false;
                            wyanjanaFlag = true;
                        }
                    } else {
                        if (this.isDiacriticConsonant(token)) {
                            this.hanacaraka += this.consonantDiacritics[token];
                        } else {
                            this.hanacaraka += this.initialConsonantAksara[token] + this.pangkon;
                        }
                    }
                } else if (postVowelFlag === false && wyanjanaFlag === false) {
                    if (nextToken && this.isConsonant(nextToken)) {
                        wyanjanaFlag = true;
                    } else {
                        wyanjanaFlag = false;
                    }
                    postVowelFlag = false;
                    this.hanacaraka += this.initialConsonantAksara[token];
                    if (!wyanjanaFlag && !(nextToken !== undefined && this.isVowel(nextToken))) {
                        this.hanacaraka += this.pangkon;
                    }
                } else if (wyanjanaFlag === true) {
                    this.hanacaraka += this.wyanjanaAksara[token];
                    wyanjanaFlag = nextToken !== undefined && this.isConsonant(nextToken);
                    if (!wyanjanaFlag && !(nextToken !== undefined && this.isVowel(nextToken))) {
                        this.hanacaraka += this.pangkon;
                    }
                }
            } else if (token === '_') {
                postVowelFlag = false;
                wyanjanaFlag = false;
            } else if (token === ' ') {
                if (this.spaces === true) {
                    this.hanacaraka += ' ';
                }
            } else {
                this.hanacaraka += this.javanesePunctuation[token] ?? this.javaneseNumerals[token] ?? token;
            }
            i++;

        }
        return this.hanacaraka;
    }

    static fromAksara(text: string): string {
        const consonantMap: { [key: string]: string } = {
            // Basic consonants
            'ꦔ': 'ng', 'ꦚ': 'ny', 'ꦡ': 'th', 'ꦣ': 'dh',
            'ꦲ': 'h',  'ꦤ': 'n',  'ꦕ': 'c',  'ꦫ': 'r',
            'ꦏ': 'k',  'ꦢ': 'd',  'ꦠ': 't',  'ꦱ': 's',
            'ꦮ': 'w',  'ꦭ': 'l',  'ꦥ': 'p',  'ꦗ': 'j',
            'ꦪ': 'y',  'ꦩ': 'm',  'ꦒ': 'g',  'ꦧ': 'b',
            // Murda (prestige) consonants — same phonological value as base form
            'ꦑ': 'k',  'ꦓ': 'g',  'ꦟ': 'n',  'ꦦ': 'p',
            'ꦯ': 's',  'ꦬ': 'r',
            // Retroflex consonants (Sanskrit-origin)
            'ꦛ': 'ṭ',  'ꦝ': 'ḍ',
        };

        const standaloneVowelMap: { [key: string]: string } = {
            'ꦄ': 'a',  'ꦆ': 'i',  'ꦈ': 'u',  'ꦌ': 'e',  'ꦎ': 'o',
            'ꦉ': 're', 'ꦊ': 'le',
        };

        const vowelDiacriticMap: { [key: string]: string } = {
            'ꦶ': 'i', 'ꦸ': 'u', 'ꦼ': 'e',
        };

        const codaDiacriticMap: { [key: string]: string } = {
            'ꦁ': 'ng', 'ꦀ': 'm', 'ꦂ': 'r', 'ꦃ': 'h',
        };

        const PANGKON = '꧀';
        const CAKRA   = 'ꦿ';
        const PENGKAL = 'ꦾ';
        const TALING  = 'ꦺ';
        const TARUNG  = 'ꦴ';

        let result = '';
        let i = 0;

        while (i < text.length) {
            const ch = text[i];

            if (ch in standaloneVowelMap) {
                result += standaloneVowelMap[ch];
                i++;
            } else if (ch in consonantMap) {
                const latin = consonantMap[ch];
                let j = i + 1;

                let medial = '';
                if (j < text.length && text[j] === CAKRA) {
                    medial = 'r';
                    j++;
                } else if (j < text.length && text[j] === PENGKAL) {
                    medial = 'y';
                    j++;
                }

                const next = j < text.length ? text[j] : '';

                if (next === PANGKON) {
                    result += latin + medial;
                    i = j + 1;
                } else if (next === TALING) {
                    const afterTaling = j + 1 < text.length ? text[j + 1] : '';
                    if (afterTaling === TARUNG) {
                        result += latin + medial + 'o';
                        i = j + 2;
                    } else {
                        result += latin + medial + 'é';
                        i = j + 1;
                    }
                } else if (next in vowelDiacriticMap) {
                    result += latin + medial + vowelDiacriticMap[next];
                    i = j + 1;
                } else {
                    result += latin + medial + 'a';
                    i = j;
                }
            } else if (ch in codaDiacriticMap) {
                result += codaDiacriticMap[ch];
                i++;
            } else if (ch === ' ') {
                result += ' ';
                i++;
            } else {
                result += ch;
                i++;
            }
        }

        return result;
    }
}


export { Aksara };
