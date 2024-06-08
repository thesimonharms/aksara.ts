class Aksara {
    hanacaraka: string;

    vowels: Set<string> = new Set(['a', 'i', 'u', 'e', 'é', 'o']);

    consonants: Set<string> = new Set(['h', 'n', 'c', 'r', 'k', 'd', 't', 's', 'w', 'l', 'p', 'j', 'y', 'm', 'g', 'b']);

    latinToAksara: { [key: string]: string } = {
        'ng': 'ꦔ',
        'ny': 'ꦚ',
        'th': 'ꦛ',
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

    getAksara(): string {
        this.hanacaraka = this.text.toLowerCase();
        if (this.spaces === false) this.hanacaraka = this.hanacaraka.replace(/ /g, '');
        this.hanacaraka = this.hanacaraka.replace(/ha/g, 'ꦲ');
        this.hanacaraka = this.hanacaraka.replace(/hi/g, 'ꦲꦶ');
        this.hanacaraka = this.hanacaraka.replace(/hu/g, 'ꦲꦸ');
        this.hanacaraka = this.hanacaraka.replace(/he/g, 'ꦲꦼ');
        this.hanacaraka = this.hanacaraka.replace(/hé/g, 'ꦲꦺ');
        this.hanacaraka = this.hanacaraka.replace(/ho/g, 'ꦲꦼ');
        this.hanacaraka = this.hanacaraka.replace(/na/g, 'ꦤ');
        this.hanacaraka = this.hanacaraka.replace(/ni/g, 'ꦤꦶ');
        this.hanacaraka = this.hanacaraka.replace(/nu/g, 'ꦤꦸ');
        this.hanacaraka = this.hanacaraka.replace(/ne/g, 'ꦤꦺ');
        this.hanacaraka = this.hanacaraka.replace(/né/g, 'ꦤꦺ');
        this.hanacaraka = this.hanacaraka.replace(/no/g, 'ꦤꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ca/g, 'ꦕ');
        this.hanacaraka = this.hanacaraka.replace(/ci/g, 'ꦕꦶ');
        this.hanacaraka = this.hanacaraka.replace(/cu/g, 'ꦕꦸ');
        this.hanacaraka = this.hanacaraka.replace(/ce/g, 'ꦕꦺ');
        this.hanacaraka = this.hanacaraka.replace(/co/g, 'ꦕꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ra/g, 'ꦫ');
        this.hanacaraka = this.hanacaraka.replace(/ri/g, 'ꦫꦶ');
        this.hanacaraka = this.hanacaraka.replace(/ru/g, 'ꦫꦸ');
        this.hanacaraka = this.hanacaraka.replace(/re/g, 'ꦫꦺ');
        this.hanacaraka = this.hanacaraka.replace(/ro/g, 'ꦫꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ka/g, 'ꦏ');
        this.hanacaraka = this.hanacaraka.replace(/ki/g, 'ꦏꦶ');
        this.hanacaraka = this.hanacaraka.replace(/ku/g, 'ꦏꦸ');
        this.hanacaraka = this.hanacaraka.replace(/ke/g, 'ꦏꦺ');
        this.hanacaraka = this.hanacaraka.replace(/ko/g, 'ꦏꦼ');
        this.hanacaraka = this.hanacaraka.replace(/da/g, 'ꦢ');
        this.hanacaraka = this.hanacaraka.replace(/di/g, 'ꦢꦶ');
        this.hanacaraka = this.hanacaraka.replace(/du/g, 'ꦢꦸ');
        this.hanacaraka = this.hanacaraka.replace(/de/g, 'ꦢꦺ');
        this.hanacaraka = this.hanacaraka.replace(/do/g, 'ꦢꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ta/g, 'ꦠ');
        this.hanacaraka = this.hanacaraka.replace(/ti/g, 'ꦠꦶ');
        this.hanacaraka = this.hanacaraka.replace(/tu/g, 'ꦠꦸ');
        this.hanacaraka = this.hanacaraka.replace(/te/g, 'ꦠꦺ');
        this.hanacaraka = this.hanacaraka.replace(/to/g, 'ꦠꦼ');
        this.hanacaraka = this.hanacaraka.replace(/sa/g, 'ꦱ');
        this.hanacaraka = this.hanacaraka.replace(/si/g, 'ꦱꦶ');
        this.hanacaraka = this.hanacaraka.replace(/su/g, 'ꦱꦸ');
        this.hanacaraka = this.hanacaraka.replace(/se/g, 'ꦱꦺ');
        this.hanacaraka = this.hanacaraka.replace(/so/g, 'ꦱꦼ');
        this.hanacaraka = this.hanacaraka.replace(/wa/g, 'ꦮ');
        this.hanacaraka = this.hanacaraka.replace(/wi/g, 'ꦮꦶ');
        this.hanacaraka = this.hanacaraka.replace(/wu/g, 'ꦮꦸ');
        this.hanacaraka = this.hanacaraka.replace(/we/g, 'ꦮꦺ');
        this.hanacaraka = this.hanacaraka.replace(/wo/g, 'ꦮꦼ');
        this.hanacaraka = this.hanacaraka.replace(/la/g, 'ꦭ');
        this.hanacaraka = this.hanacaraka.replace(/li/g, 'ꦭꦶ');
        this.hanacaraka = this.hanacaraka.replace(/lu/g, 'ꦭꦸ');
        this.hanacaraka = this.hanacaraka.replace(/le/g, 'ꦭꦺ');
        this.hanacaraka = this.hanacaraka.replace(/lo/g, 'ꦭꦼ');
        this.hanacaraka = this.hanacaraka.replace(/pa/g, 'ꦥ');
        this.hanacaraka = this.hanacaraka.replace(/pi/g, 'ꦥꦶ');
        this.hanacaraka = this.hanacaraka.replace(/pu/g, 'ꦥꦸ');
        this.hanacaraka = this.hanacaraka.replace(/pe/g, 'ꦥꦺ');
        this.hanacaraka = this.hanacaraka.replace(/po/g, 'ꦥꦼ');
        this.hanacaraka = this.hanacaraka.replace(/dha/g, 'ꦝ');
        this.hanacaraka = this.hanacaraka.replace(/dhi/g, 'ꦝꦶ');
        this.hanacaraka = this.hanacaraka.replace(/dhu/g, 'ꦝꦸ');
        this.hanacaraka = this.hanacaraka.replace(/dhe/g, 'ꦝꦺ');
        this.hanacaraka = this.hanacaraka.replace(/dho/g, 'ꦝꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ja/g, 'ꦗ');
        this.hanacaraka = this.hanacaraka.replace(/ji/g, 'ꦗꦶ');
        this.hanacaraka = this.hanacaraka.replace(/ju/g, 'ꦗꦸ');
        this.hanacaraka = this.hanacaraka.replace(/je/g, 'ꦗꦺ');
        this.hanacaraka = this.hanacaraka.replace(/jo/g, 'ꦗꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ya/g, 'ꦪ');
        this.hanacaraka = this.hanacaraka.replace(/yi/g, 'ꦪꦶ');
        this.hanacaraka = this.hanacaraka.replace(/yu/g, 'ꦪꦸ');
        this.hanacaraka = this.hanacaraka.replace(/ye/g, 'ꦪꦺ');
        this.hanacaraka = this.hanacaraka.replace(/yo/g, 'ꦪꦼ');
        this.hanacaraka = this.hanacaraka.replace(/nya/g, 'ꦚ');
        this.hanacaraka = this.hanacaraka.replace(/nyi/g, 'ꦚꦶ');
        this.hanacaraka = this.hanacaraka.replace(/nyu/g, 'ꦚꦸ');
        this.hanacaraka = this.hanacaraka.replace(/nye/g, 'ꦚꦺ');
        this.hanacaraka = this.hanacaraka.replace(/nyo/g, 'ꦚꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ma/g, 'ꦩ');
        this.hanacaraka = this.hanacaraka.replace(/mi/g, 'ꦩꦶ');
        this.hanacaraka = this.hanacaraka.replace(/mu/g, 'ꦩꦸ');
        this.hanacaraka = this.hanacaraka.replace(/me/g, 'ꦩꦺ');
        this.hanacaraka = this.hanacaraka.replace(/mo/g, 'ꦩꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ga/g, 'ꦒ');
        this.hanacaraka = this.hanacaraka.replace(/gi/g, 'ꦒꦶ');
        this.hanacaraka = this.hanacaraka.replace(/gu/g, 'ꦒꦸ');
        this.hanacaraka = this.hanacaraka.replace(/ge/g, 'ꦒꦺ');
        this.hanacaraka = this.hanacaraka.replace(/go/g, 'ꦒꦼ');
        this.hanacaraka = this.hanacaraka.replace(/ba/g, 'ꦧ');
        this.hanacaraka = this.hanacaraka.replace(/bi/g, 'ꦧꦶ');
        this.hanacaraka = this.hanacaraka.replace(/bu/g, 'ꦧꦸ');
        this.hanacaraka = this.hanacaraka.replace(/be/g, 'ꦧꦺ');
        this.hanacaraka = this.hanacaraka.replace(/bo/g, 'ꦧꦼ');
        this.hanacaraka = this.hanacaraka.replace(/tha/g, 'ꦛ');
        this.hanacaraka = this.hanacaraka.replace(/thi/g, 'ꦛꦶ');
        this.hanacaraka = this.hanacaraka.replace(/thu/g, 'ꦛꦸ');
        this.hanacaraka = this.hanacaraka.replace(/the/g, 'ꦛꦺ');
        this.hanacaraka = this.hanacaraka.replace(/tho/g, 'ꦛꦼ');
        this.hanacaraka = this.hanacaraka.replace(/nga/g, 'ꦔ');
        this.hanacaraka = this.hanacaraka.replace(/ngi/g, 'ꦔꦶ');
        this.hanacaraka = this.hanacaraka.replace(/ngu/g, 'ꦔꦸ');
        this.hanacaraka = this.hanacaraka.replace(/nge/g, 'ꦔꦺ');
        this.hanacaraka = this.hanacaraka.replace(/ngo/g, 'ꦔꦼ');
        this.hanacaraka = this.hanacaraka.replace(/a/g, 'ꦄ');
        this.hanacaraka = this.hanacaraka.replace(/i/g, 'ꦆ');
        this.hanacaraka = this.hanacaraka.replace(/u/g, 'ꦈ');
        this.hanacaraka = this.hanacaraka.replace(/e/g, 'ꦌ');
        this.hanacaraka = this.hanacaraka.replace(/o/g, 'ꦎ');
        return this.hanacaraka;
    }
}

export { Aksara };