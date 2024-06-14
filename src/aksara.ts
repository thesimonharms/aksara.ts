type PreSyllable = {
    initial: string,
    vowel: string,
    space: boolean
}

type CompleteSyllable = {
    initial: string,
    vowel: string,
    final: string,
    space: boolean
}

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

    emptyPreSyllable: PreSyllable = {
        initial: '',
        vowel: '',
        space: false
    }

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

    // The consonants used when two consonants touch each other
    wyanjanaAksara: { [key: string]: string } = {
        'ng': '꧀ꦔ',
        'ny': '꧀ꦚ',
        'th': '꧀ꦛ',
        'dh': '꧀ꦝ',
        'h': '꧀ꦲ',
        'n': '꧀ꦤ',
        'c': '꧀ꦕ',
        'r': '꧀ꦫ',
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
        'e': 'ꦲꦺ',
        'é': 'ꦲꦼ',
        'o': 'ꦲꦺꦴ'
    };

    // Vowels that cannot possibly be confused for h+vowel, included for planned update where this will be an option.
    explicitAksaraVowels: { [key: string]: string } = {
        'a': 'ꦄ',
        'i': 'ꦆ',
        'u': 'ꦈ',
        'e': 'ꦌ',
        'é': 'Words CAN NOT start with é',
        'o': 'ꦎ'
    }

    // The text to be converted to Aksara, and whether or not to include spaces
    constructor(public readonly text: string, public readonly spaces: boolean = false) {
        this.text = text;
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
            let nextNextChar = (i + 2 < this.text.length) ? this.text[i + 2] : undefined;
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
                i++;
            }
        }

        return tokenizedText;
    }

    getAksara(): string {
        let wyanjanaFlag = false;
        let postVowelFlag = false;

        let i = 0;
        while (i < this.tokens.length) {
            let previousToken = (i - 1 >= 0) ? this.tokens[i - 1] : undefined;
            let token = this.tokens[i];
            let nextToken = (i + 1 < this.text.length) ? this.tokens[i + 1] : undefined;

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
                    this.hanacaraka += this.normalAksaraVowels[token];
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
                } else if (wyanjanaFlag === true) {
                    wyanjanaFlag = false;
                    this.hanacaraka += this.wyanjanaAksara[token];
                }
            } else if (token === ' ') {
                if (this.spaces === true) {
                    this.hanacaraka += ' ';
                }
            }
            i++;
            
        }
        return this.hanacaraka;
    }
}


export { Aksara };