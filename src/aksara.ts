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

    // List of all vowels used in Javanese
    vowels: Set<string> = new Set(['a', 'i', 'u', 'e', 'é', 'o']);

    // List of all consonants used in Javanese
    consonants: Set<string> = new Set(['h', 'n', 'c', 'r', 'k', 'd', 't', 's', 'w', 'l', 'p', 'j', 'y', 'm', 'g', 'b']);

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

    isFinalConsonant(nextChar: string, nextNextChar: string = '', nextNextNextChar: string = '', lastChar: string = ''): boolean {
        if (this.isConsonant(nextChar) && this.isVowel(nextNextChar)) {
            return false;
        } else if (this.isConsonant(nextChar) && this.isConsonant(nextNextChar) && this.isVowel(nextNextNextChar)) {
            return true;
        } else if (this.isTwoCharacterConsonant(nextChar, nextNextChar) && this.isConsonant(nextNextNextChar) && this.isVowel(lastChar)) {
            return true;
        } else if (this.isVowel(nextChar)) {
            return false;
        } else {
            return false;
        }
    }

    containsVowel(syllable: string): boolean {
        let i = 0;
        let vowelFound = false;
        while (i < syllable.length) {
            if (this.isVowel(syllable[i])) {
                vowelFound = true;
            }
            i++;
        }
        return vowelFound;
    }

    /**
     * Separates a given input string into individual syllables.
     *
     * @param input - The input string to separate into syllables.
     * @returns An array of strings representing the syllables.
     */
    separateSyllables(input: string): string[] {
        // Initialize an empty array to store the output syllables
        const syllables: string[] = [];
        // Initialize an empty string to store the current syllable
        let currentSyllable = '';

        // Start at the beginning of the input string (index 0)
        let i = 0;
        // Iterate through the input string
        while (i < input.length) {
            // Set char as the current character
            let char = input[i];
            // Set nextChar as the next character
            let nextChar = input[i + 1];
            // Set nextNextChar as the character after the next character
            let nextNextChar = input[i + 2];
            // Set nextNextNextChar as the character after the character after the next character
            let nextNextNextChar = input[i + 3];

            // If the current character is a consonant and the next character is a vowel, or if the next character is a space, end the current syllable
            // Also end the syllable if the next character is a final consonant
            if (this.isConsonant(char) && this.isVowel(nextChar)) {
                syllables.push(char + nextChar);
                // Increment i by 2 to skip the next character
                i += 2;
            } else if (char === ' ' && this.spaces === true) {
                // If the current character is a space, add a space to the syllables array
                syllables.push(' ');
                // Increment i by 1 to skip the next character
                i++;
            } else if (this.isFinalConsonant(char, nextChar, nextNextChar, nextNextNextChar)) {
                // Test for final consonants
                if (this.isTwoCharacterConsonant(char, nextChar) && this.isVowel(nextNextChar)) {
                    // If it is a two consonant pair + vowel, add all 3 to the syllable array
                    syllables.push(char + nextChar + nextNextChar);
                    // Increment i by 3 to skip the next two characters
                    i += 3;
                } else if (this.isTwoCharacterConsonant(char, nextChar) && this.isConsonant(nextNextChar) && this.isVowel(nextNextNextChar)) {
                    // If it is a two consonant pair final consonant, add the consonant pair to the syllable array by itself
                    syllables.push(char + nextChar);
                    // Increment i by 2 to skip the next character
                    i += 2;
                }
            } else if (this.isVowel(char)) {
                // Handle vowels
                syllables.push(char);
                // Skip to the next character
                i++;
            } else {
                // If confused just increment i by 1
                i++;
            }
        }

        // Push the last syllable if it's not empty
        if (currentSyllable !== '') {
            // Probably a useless piece of code but once again, scared to remove it.
            // TODO: Find out if this actually does anything
            syllables.push(currentSyllable);
        }

        // Return syllable array (as string)
        return syllables;
    }

    // Function to divide syllables into initial, vowel, and final, and include space information
    // TODO: Fix final consonants instead of bypassing to getAksara()
    divideSyllables(input: string[]): PreSyllable[] {

        let preSyllableArray: PreSyllable[] = [];

        input.forEach((divSyllable => {
            // Start at the beginning of the syllable (index 0)
            let i = 0;
            // Initialize an empty syllable object
            let syllable: PreSyllable = {
                initial: '',
                vowel: '',
                space: false
            };
            // Iterate through the syllable
            while (i < divSyllable.length) {
                // Set char as the current character
                let char = divSyllable[i];
                // Set nextChar as the next character
                let nextChar = divSyllable[i + 1];

                // Handle spaces
                if (char === ' ') {
                    syllable.initial += char;
                    syllable.space = true;
                }
                // If the character is a vowel, add it to the vowel part of the syllable
                if (this.isVowel(char)) {
                    syllable.vowel += char;
                }
                // If the character is a consonant, add it to the initial or final part of the syllable
                if (i === 0 && this.isConsonant(char)) {
                    if (this.isConsonant(nextChar) && nextChar === 'h') {
                        // Handle dh and th
                        syllable.initial += char + nextChar;
                        i++;
                    } else if (char === 'n' && this.isConsonant(nextChar) && (nextChar === 'g' || nextChar === 'y')) {
                        // Handle ng and ny
                        syllable.initial += char + nextChar;
                        i++;
                    } else if (this.isConsonant(char)) {
                        // Handle single character consonants
                        syllable.initial += char;
                    }
                }
                /* else if (this.isConsonant(char)) {
                    // Currently useless code that doesn't work, because the input fed to it doesn't have final consonants. TODO: WILL FIX.
                    if (this.isConsonant(nextChar) && nextChar !== 'h') {
                        syllable.final += char;
                    } else if (this.isConsonant(nextChar) && nextChar === 'h') {
                        syllable.final += char + nextChar;
                        i++;
                    } else if (char === 'n' && this.isConsonant(nextChar) && (nextChar === 'g' || nextChar === 'y')) {
                        syllable.final += char + nextChar;
                        i++;
                    }
                } */
                i++;
            }
            preSyllableArray.push(syllable);
        }));
        return preSyllableArray;
    }

    analyzePreSyllables(preSyllables: PreSyllable[]): CompleteSyllable[] {
        let syllables: CompleteSyllable[] = [];
        let previousPreSyllable = this.emptyPreSyllable;
        let i = 0;
        while (i < preSyllables.length) {
            if (i !== 0) {
                previousPreSyllable = preSyllables[i - 1];
            }
            let currentPreSyllable = preSyllables[i];
            let nextPreSyllable = (i + 1 < preSyllables.length) ? preSyllables[i + 1] : undefined;
            if (previousPreSyllable !== this.emptyPreSyllable) {
                if (currentPreSyllable.initial !== '' && previousPreSyllable.vowel !== '' && nextPreSyllable && nextPreSyllable.initial !== '') {
                    syllables.push({
                        initial: currentPreSyllable.initial,
                        vowel: currentPreSyllable.vowel,
                        final: '',
                        space: currentPreSyllable.space
                    });
                    i++;
                } else if (currentPreSyllable.initial === '' && nextPreSyllable && nextPreSyllable.vowel === '' && nextPreSyllable !== undefined) {
                    syllables.push({
                        initial: '',
                        vowel: currentPreSyllable.vowel,
                        final: nextPreSyllable.initial,
                        space: nextPreSyllable.space
                    });
                    i+=2;
                } else if (currentPreSyllable.initial === '' && currentPreSyllable.vowel === '' && currentPreSyllable.space === true) {
                    syllables.push({
                        initial: '',
                        vowel: '',
                        final: '',
                        space: currentPreSyllable.space
                    });
                    i++;
                } else {
                    syllables.push({
                        initial: currentPreSyllable.initial,
                        vowel: currentPreSyllable.vowel,
                        final: '',
                        space: currentPreSyllable.space
                    });
                    i++;
                }
            } else {
                if (nextPreSyllable && nextPreSyllable.vowel === '') {
                    syllables.push({
                        initial: currentPreSyllable.initial,
                        vowel: currentPreSyllable.vowel,
                        final: nextPreSyllable.initial,
                        space: currentPreSyllable.space
                    });
                    i+=2;
                } else if (nextPreSyllable && nextPreSyllable.vowel !== '') {
                    syllables.push({
                        initial: currentPreSyllable.initial,
                        vowel: currentPreSyllable.vowel,
                        final: '',
                        space: currentPreSyllable.space
                    });
                    i++;
                } else {
                    syllables.push({
                        initial: currentPreSyllable.initial,
                        vowel: currentPreSyllable.vowel,
                        final: '',
                        space: currentPreSyllable.space
                    });
                    i++;
                }
            }
        }

        return syllables;
    }

    // Function to return aksara from latin (probably will only ever work for Javanese, but who knows?)
    getAksara(): string {
        // This whole function will need to be retoolled when I change separateSyllables to not separate
        // by strict syllables but by the smallest convertable units that can be converted to aksara.
        // Separate the syllables and iterate through them.
        let preSyllables: PreSyllable[] = this.divideSyllables(this.separateSyllables(this.text));
        // Analyze the syllables and return the aksara.
        let syllables = this.analyzePreSyllables(preSyllables);
        /* syllables.forEach((syllableObj) => {
            // Reset the aksara variable
            let aksara = '';
            // If the syllable has an initial consonant, add it to the aksara
            if (syllableObj.initial !== '') {
                // If the initial consonant is a diacritic consonant, use the diacritic aksara,  retool later when final consonants work. TODO: WILL FIX.
                aksara += this.initialConsonantAksara[syllableObj.initial];
                aksara += this.vowelDiacritics[syllableObj.vowel];
            } else {
                // TODO: Implement this for explicit vowels
                aksara += this.normalAksaraVowels[syllableObj.vowel];
            }
            // If the syllable has a final consonant, add it to the aksara
            if (syllableObj.final !== '') {
                if (this.diacriticConsonants.has(syllableObj.final)) {
                    aksara += this.consonantDiacritics[syllableObj.final];
                }
            }
            if (this.spaces === true && syllableObj.space === true) {
                aksara = ' ';
            }
            this.hanacaraka += aksara;
        }); */

        let i = 0;
        while (i < syllables.length) {
            let syllable = syllables[i];
            let nextSyllable = (i + 1 < syllables.length) ? syllables[i + 1] : undefined;
            let aksara = '';

            if (this.spaces === true && syllable.space === true) {
                aksara = ' ';
            }

            if (syllable.initial === '') {
                aksara += this.normalAksaraVowels[syllable.vowel];
            } else {
                aksara += this.initialConsonantAksara[syllable.initial];
                aksara += this.vowelDiacritics[syllable.vowel];
            }

            if (syllable.final !== '') {
                if (nextSyllable && this.diacriticConsonants.has(syllable.final)) {
                    aksara += this.consonantDiacritics[syllable.final];
                } else if (nextSyllable && !this.diacriticConsonants.has(syllable.final)) {
                    aksara += this.wyanjanaAksara[syllable.final];
                } else {
                    aksara += this.initialConsonantAksara[syllable.final] + this.pangkon;
                }
            }
            this.hanacaraka+=aksara;
            i++;
        }

        return this.hanacaraka;
    }

}

export { Aksara };