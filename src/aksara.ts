class Aksara {
    hanacaraka: string;

    vowels: Set<string> = new Set(['a', 'i', 'u', 'e', 'é', 'o']);

    consonants: Set<string> = new Set(['h', 'n', 'c', 'r', 'k', 'd', 't', 's', 'w', 'l', 'p', 'j', 'y', 'm', 'g', 'b']);

    pangkon: string = '꧀';

    consonantDiacritics: { [key: string]: string } = {
        'ng': 'ꦁ',
        'm': 'ꦀ',
        'r': 'ꦂ',
        'h': 'ꦃ'
    };

    vowelDiacritics: { [key: string]: string } = {
        'a': '',
        'i': 'ꦶ',
        'u': 'ꦸ',
        'e': 'ꦼ',
        'é': 'ꦺ',
        'o': 'ꦼ'
    };

    initialConsonantAksara: { [key: string]: string } = {
        'ng': 'ꦔ',
        'ny': 'ꦚ',
        'th': 'ꦡ',
        'dh': 'ꦝ',
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

    normalAksaraVowels: { [key: string]: string } = {
        'a': 'ꦲ',
        'i': 'ꦲꦶ',
        'u': 'ꦲꦸ',
        'e': 'ꦲꦺ',
        'é': 'ꦲꦼ',
        'o': 'ꦲꦺꦴ'
    };

    explicitAksaraVowels: { [key: string]: string } = {
        'a': 'ꦄ',
        'i': 'ꦆ',
        'u': 'ꦈ',
        'e': 'ꦌ',
        'é': 'Words CAN NOT start with é',
        'o': 'ꦎ'
    }

    constructor(public readonly text: string, public readonly spaces: boolean = false) {
        this.text = text;
        this.hanacaraka = '';
        this.spaces = spaces;
    }

    getText(): string {
        return this.text;
    }

    isVowel(char: string): boolean {
        return this.vowels.has(char);
    }

    isConsonant(char: string): boolean {
        return this.consonants.has(char);
    }

    // Function to separate syllables in Latin-based Javanese
    separateSyllables(input: string): string[] {
        const syllables: string[] = [];
        let currentSyllable = '';
        let i = 0;

        while (i < input.length) {
            let char = input[i];
            let nextChar = input[i + 1];
            let nextNextChar = input[i + 2];

            if (char === ' ') {
                // Treat space as a separate element
                if (currentSyllable) {
                    syllables.push(currentSyllable);
                    currentSyllable = '';
                }
                if (this.spaces === true) {
                    syllables.push(' ');
                }
                i++;
                continue;
            }

            if (char === 'n' && (nextChar === 'g' || nextChar === 'y')) {
                // Handle "ng" and "ny"
                char += nextChar;
                i++;
            } else if (char === 'd' && nextChar === 'h') {
                // Handle "dh"
                char += nextChar;
                i++;
            } else if (char === 't' && nextChar === 'h') {
                // Handle "th"
                char += nextChar;
                i++;
            }

            currentSyllable += char;

            if (this.isVowel(nextChar)) {
                currentSyllable += nextChar;
                i++; // Skip the next character since it's part of the current syllable
            }

            // If the next character is a consonant or the end of the string, end the current syllable
            if (this.isConsonant(nextChar) || !nextChar || (nextChar && nextNextChar && this.isConsonant(nextNextChar))) {
                syllables.push(currentSyllable);
                currentSyllable = '';
            }

            i++;
        }

        // Add any remaining characters to the last syllable
        if (currentSyllable) {
            syllables.push(currentSyllable);
        }

        return syllables;
    }

    getAksara(): string {
        this.separateSyllables(this.text).forEach((syllable) => {
            let aksara = '';
            let i = 0;

            if (syllable.length === 1) {
                // Handle single character syllables
                let char = syllable[0];
                if (this.isVowel(char)) {
                    aksara += this.normalAksaraVowels[char];
                } else if (char === ' ') {
                    aksara += ' ';
                } else if (this.isConsonant(char)) {
                    aksara += this.initialConsonantAksara[char] + this.pangkon;
                }
            } else {
                while (i < syllable.length) {
                    let char = syllable[i];
                    let nextChar = syllable[i + 1];
                    let nextNextChar = syllable[i + 2];

                    if (char === 'n' && (nextChar === 'g' || nextChar === 'y')) {
                        // Handle "ng" and "ny"
                        char += nextChar;
                        i++;
                    } else if (char === 'd' && nextChar === 'h') {
                        // Handle "dh"
                        char += nextChar;
                        i++;
                    } else if (char === 't' && nextChar === 'h') {
                        // Handle "th"
                        char += nextChar;
                        i++;
                    }

                    if (this.isConsonant(char)) {
                        // Initial consonant
                        if (i === 0) {
                            aksara += this.initialConsonantAksara[char];
                        } else {
                            // Wyanjana
                            aksara += this.wyanjanaAksara[char];
                        }
                    }

                    if (this.isVowel(char)) {
                        aksara += this.vowelDiacritics[char];
                    }

                    i++;
                }
            }

            this.hanacaraka += aksara;
        });

        return this.hanacaraka;
    }

}

export { Aksara };