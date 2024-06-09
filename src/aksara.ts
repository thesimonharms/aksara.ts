type Syllable = {
    initial: string,
    vowel: string,
    final: string,
    space: boolean
}

class Aksara {
    // Eventually becomes returned aksara jawa text
    hanacaraka: string;

    // List of all vowels used in Javanese
    vowels: Set<string> = new Set(['a', 'i', 'u', 'e', 'é', 'o']);

    // List of all consonants used in Javanese
    consonants: Set<string> = new Set(['h', 'n', 'c', 'r', 'k', 'd', 't', 's', 'w', 'l', 'p', 'j', 'y', 'm', 'g', 'b']);

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
        'o': 'ꦼ'
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

    // TODO: Implement this function properly
    nextCharacterFinalConsonant(char: string, nextChar: string, nextNextChar: string, nextNextNextChar: string): boolean {
        return this.isConsonant(nextChar) && !this.isVowel(nextNextChar) && !this.isVowel(nextNextNextChar);
    }

    // Function to separate syllables in Latin script Javanese
    // TODO: Make this function able to handle consonant final syllables.
    separateSyllables(input: string): string[] {
        // Empty array to store syllables
        const syllables: string[] = [];
        // Empty string to store the current syllable
        let currentSyllable = '';
        // Start at the beginning of the string (index 0)
        let i = 0;

        // Iterate through the string
        while (i < input.length) {
            // Set char as the current character
            let char = input[i];
            // Set nextChar as the next character
            let nextChar = input[i + 1];
            // Set nextNextChar as the character after the next character
            let nextNextChar = input[i + 2];

            // Handle spaces
            if (char === ' ') {
                // Treat space as a separate element
                if (currentSyllable) {
                    // Push the current syllable to the syllables array
                    syllables.push(currentSyllable);
                    // Reset the current syllable
                    currentSyllable = '';
                }
                // If spaces are enabled, add a space to the syllables array
                if (this.spaces === true) {
                    syllables.push(' ');
                }
                // Continue iterating and skip the rest of the loop
                i++;
                continue;
            }

            // Detect ng and ny, the nextChar === 'h' is probably not necassary but I don't want to have to fix this later.
            if (char === 'n' && (nextChar === 'g' || nextChar === 'h' || nextChar === 'y')) {
                // Handle "ng" and "ny"
                if (this.isVowel(nextNextChar)) {
                    char += nextChar + nextNextChar;
                } else {
                    char += nextChar;
                }
                // Skip 2 characters since "ng" and "ny" are 2 characters
                i += 2;
            } else if (char === 'd' && nextChar === 'h') {
                // Handle "dh"
                if (this.isVowel(nextNextChar)) {
                    char += nextChar + nextNextChar;
                } else {
                    char += nextChar;
                }
                // Skip 2 characters since "dh" is 2 characters
                i += 2;
            } else if (char === 't' && nextChar === 'h') {
                // Handle "th"
                if (this.isVowel(nextNextChar)) {
                    char += nextChar + nextNextChar;
                } else {
                    char += nextChar;
                }
                // Skip 2 characters since "th" is 2 characters
                i += 2;
            }

            // Add the current character to the current syllable
            currentSyllable += char;

            // If the next character is a vowel, add it to the current syllable
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

    divideSyllables(input: string): Syllable {
        let syllable: Syllable = {
            initial: '',
            vowel: '',
            final: '',
            space: false
        };

        let i = 0;

        while (i < input.length) {
            let char = input[i];
            let nextChar = input[i + 1];

            if (char === ' ') {
                syllable.initial += char;
                syllable.space = true;
            }
            if (this.isVowel(char)) {
                syllable.vowel += char;
            }
            if (i === 0 && this.isConsonant(char)) {
                if (this.isConsonant(nextChar) && nextChar === 'h') {
                    syllable.initial += char + nextChar;
                    i++;
                } else if (char === 'n' && this.isConsonant(nextChar) && (nextChar === 'g' || nextChar === 'y')) {
                    syllable.initial += char + nextChar;
                    i++;
                } else if (this.isConsonant(char)) {
                    syllable.initial += char;
                }
            } else if (this.isConsonant(char)) {
                if (this.isConsonant(nextChar) && nextChar !== 'h') {
                    syllable.final += char;
                } else if (this.isConsonant(nextChar) && nextChar === 'h') {
                    syllable.final += char + nextChar;
                    i++;
                } else if (char === 'n' && this.isConsonant(nextChar) && (nextChar === 'g' || nextChar === 'y')) {
                    syllable.final += char + nextChar;
                    i++;
                }
            }
            i++;
        }
        return syllable;
    }

    getAksara(): string {
        this.separateSyllables(this.text).forEach((syllable) => {
            let syllableObj = this.divideSyllables(syllable);
            let aksara = '';
            if (syllableObj.initial !== '') {
                aksara += this.initialConsonantAksara[syllableObj.initial];
                aksara += this.vowelDiacritics[syllableObj.vowel];
            } else {
                aksara += this.normalAksaraVowels[syllableObj.vowel];
            }
            if (syllableObj.final !== '') {
                if(this.diacriticConsonants.has(syllableObj.final)) {
                    aksara += this.consonantDiacritics[syllableObj.final];
                }
            }
            if (this.spaces === true && syllableObj.space === true) {
                aksara = ' ';
            }
            console.log(aksara);
            this.hanacaraka += aksara;
        });

        return this.hanacaraka;
    }

}

export { Aksara };