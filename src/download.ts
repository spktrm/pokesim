import * as fs from "fs";
import {
    Ability,
    ID,
    Item,
    Learnset,
    ModdedDex,
    Move,
    Species,
    SpeciesName,
    Type,
} from "@pkmn/dex";
import { Generations } from "@pkmn/data";
import { TeamValidator } from "@pkmn/sim";
import { Move as DexMove } from "@pkmn/sim/build/cjs/sim/dex-moves";
import { Species as DexSpecies } from "@pkmn/sim/build/cjs/sim/dex-species";

// Define the base URL and the paths to download
const BASE_URL = "https://raw.githubusercontent.com/pkmn/ps/main/sim/{}.ts";
const PATHS = [
    "sim/battle-actions",
    "sim/battle-queue",
    "sim/battle-stream",
    "sim/battle",
    "sim/dex-abilities",
    "sim/dex-conditions",
    "sim/dex-data",
    "sim/dex-formats",
    "sim/dex-items",
    "sim/dex-moves",
    "sim/dex-species",
    "sim/dex",
    "sim/exported-global-types",
    "sim/field",
    "sim/global-types",
    "sim/global-variables.d",
    "sim/index",
    "sim/pokemon",
    "sim/prng",
    "sim/side",
    "sim/state",
    "sim/team-validator.",
    "sim/teams",
    "lib/index",
    "lib/streams",
    "lib/utils",
    "data/abilities",
    "data/aliases",
    "data/conditions",
    "data/formats-data",
    "data/index",
    "data/items",
    "data/learnsets",
    "data/legality",
    "data/moves",
    "data/natures",
    "data/pokedex",
    "data/pokemongo",
    "data/rulesets",
    "data/scripts",
    "data/tags",
    "data/typechart",
];
const URLS = PATHS.map((path) => BASE_URL.replace("{}", path));

// Helper function to convert a string to an identifier
function toId(string: string): string {
    return string.toLowerCase().replace(/[^a-z0-9]/g, "");
}

// Helper function to reduce an array to unique, sorted identifiers
function reduce(arr: string[]): string[] {
    return Array.from(new Set(arr.map(toId))).sort();
}

function findDuplicates(arr: string[]): string[] {
    return arr.filter((item, index) => {
        return arr.indexOf(item) !== index;
    });
}

// Helper function to create an enumeration from an array
function enumerate(arr: string[]): { [key: string]: number } {
    const enumeration: { [key: string]: number } = {};
    const dupes = findDuplicates(arr);
    arr.forEach((item, index) => {
        enumeration[item] = index;
    });
    return enumeration;
}

// Function to fetch text content from a URL
async function fetchText(url: string): Promise<string> {
    const response = await fetch(url);
    return response.text();
}

// Function to download all files and concatenate their content
async function downloadAll(urls: string[]): Promise<string[]> {
    const requests = urls.map((url) => fetchText(url));
    return Promise.all(requests);
}

// Function to extract unique strings from source text based on a regular expression
function extractPatterns(src: string, pattern: RegExp): string[] {
    return [...src.matchAll(pattern)].map((match) => match[1]);
}

type GenData = {
    species: Species[];
    moves: Move[];
    abilities: Ability[];
    items: Item[];
    typechart: Type[];
    learnsets: {
        species: SpeciesName;
        [k: string]: any;
    }[];
};

async function getGenData(gen: number): Promise<GenData> {
    const format = `gen${gen}` as ID;
    const dex = new ModdedDex(format);
    const species = [...dex.species.all()];
    const allMoves = [...dex.moves.all()];
    const promises: any[] = [];
    const validator = new TeamValidator(format);
    species.map((species) =>
        allMoves.map((move) =>
            promises.push(
                validator.omCheckCanLearn(
                    move as unknown as DexMove,
                    species as unknown as DexSpecies,
                ),
            ),
        ),
    );
    const learnsets = species.map((_, speciesIndex) => {
        const datum: { [k: string]: any } = {};
        allMoves.map((move, moveIndex) => {
            datum[move.id] =
                promises[speciesIndex * species.length + moveIndex] === null;
        });
        return datum;
    });
    const data = {
        species: species,
        moves: [...dex.moves.all()],
        abilities: [...dex.abilities.all()],
        items: [...dex.items.all()],
        typechart: [...dex.types.all()],
        learnsets: species.map((species, index) => ({
            species: species.name,
            learnset: learnsets[index],
        })),
    };
    return data;
}

function mapId<T extends { id: string; [key: string]: any }>(
    arr: T[],
): string[] {
    return arr.map((item) => item.id);
}

const padToken = "<PAD>";
const unkToken = "<UNK>";
const nullToken = "<NULL>";
const extraTokens = [padToken, unkToken];

function formatKey(key: string): string {
    return key.startsWith("<") ? key : key.toLowerCase().replace(/[\W_]+/g, "");
}

function formatData(data: GenData) {
    const moveIds = [
        ...data.moves.map((item) => {
            if (item.id === "return") {
                return `${item.id}102`;
            } else {
                return formatKey(item.name);
            }
        }),
        "return",
    ];
    return {
        species: enumerate([...extraTokens, ...mapId(data.species)]),
        moves: enumerate(["<SWITCH>", "<NONE>", ...extraTokens, ...moveIds]),
        abilities: enumerate([...extraTokens, ...mapId(data.abilities)]),
        items: enumerate([nullToken, ...extraTokens, ...mapId(data.items)]),
    };
}

// The main function that executes the download and processing
async function main(): Promise<void> {
    const sources = await downloadAll(URLS);
    const src = sources.join("\n");

    // Extract patterns for different categories
    const weathersPattern = /['|"]-weather['|"],\s*['|"](.*)['|"],/g;
    const sideConditionsPattern = /sideCondition:\s*['|"](.*)['|"],/g;
    const terrainPattern = /terrain:\s*['|"](.*)['|"],/g;
    const pseudoWeatherPattern = /pseudoWeather:\s['|"](.*?)['|"]/g;

    // Define patterns for volatile status
    const volatileStatusPatterns = [
        /removeVolatile\(['|"](.*?)['|"]\)/g,
        /hasVolatile\(['|"](.*?)['|"]\)/g,
        /volatiles\[['|"](.*?)['|"]\]/g,
        /volatiles\.(.*?)[\[\)| ]/g,
        /volatileStatus:\s*['|"](.*)['|'],/g,
    ];

    // Use a Set to ensure uniqueness
    let volatileStatusSet = new Set<string>();

    // Process each pattern and add the results to the Set
    volatileStatusPatterns.forEach((pattern) => {
        const matches = src.match(pattern);
        if (matches) {
            matches.forEach((match) => {
                const cleanedMatch = match
                    .replace(pattern, "$1") // Extract the captured group
                    .replace(/['|"|\[|\]|\(|\)|,]/g, "") // Clean up any extra characters
                    .trim();
                if (cleanedMatch) {
                    volatileStatusSet.add(cleanedMatch);
                }
            });
        }
    });

    // Convert the Set to an array and reduce it to unique, sorted identifiers
    let volatileStatus = [
        "itemremoved",
        "typechange",
        "wrap",
        "formechange",
        "transform",
        ...reduce(Array.from(volatileStatusSet)),
    ];

    let weathers = extractPatterns(src, weathersPattern);
    weathers = reduce(weathers).map((t) => t.replace("raindance", "rain"));

    let sideConditions = extractPatterns(src, sideConditionsPattern);
    sideConditions = reduce(sideConditions);

    let terrain = extractPatterns(src, terrainPattern);
    terrain = reduce(terrain).map((t) => t.replace("terrain", ""));

    let pseudoweather = extractPatterns(src, pseudoWeatherPattern);
    pseudoweather = reduce(pseudoweather);

    const genData = await getGenData(9);

    // Create the data object
    const data = {
        pseudoWeather: enumerate([nullToken, ...pseudoweather.sort()]),
        volatileStatus: enumerate([nullToken, ...volatileStatus.sort()]),
        weathers: enumerate([nullToken, ...weathers.sort()]),
        terrain: enumerate([nullToken, ...terrain.sort()]),
        sideConditions: enumerate([nullToken, ...sideConditions.sort()]),
        ...formatData(genData),
        statuses: enumerate([
            nullToken,
            "slp",
            "psn",
            "brn",
            "frz",
            "par",
            "tox",
        ]),
        boosts: enumerate([
            "atk",
            "def",
            "spa",
            "spd",
            "spe",
            "accuracy",
            "evasion",
        ]),
        types: enumerate([
            padToken,
            unkToken,
            ...genData.typechart.flatMap((type) =>
                type.isNonstandard === "Future" ? [] : type.id,
            ),
        ]),
        genders: enumerate([unkToken, "M", "F", "N"]),
    };

    const parentDataDir = `src/data/`;

    if (!fs.existsSync(parentDataDir)) {
        fs.mkdirSync(parentDataDir, { recursive: true });
    }

    // Write the data to a JSON file
    fs.writeFileSync(
        `${parentDataDir}/data.json`,
        JSON.stringify(data, null, 2),
    );

    for (const genNo of [1, 2, 3, 4, 5, 6, 7, 8, 9]) {
        const parentDir = `${parentDataDir}/gen${genNo}/`;
        if (!fs.existsSync(parentDir)) {
            fs.mkdirSync(parentDir, { recursive: true });
        }
        const genData = await getGenData(genNo);
        for (const [key, value] of Object.entries(genData)) {
            const ourPath = `${parentDir}/${key}.json`;
            console.log(`writing ${ourPath}`);
            fs.writeFileSync(ourPath, JSON.stringify(value, null, 2));
        }
    }
}

// Execute the main function
main().catch((error) => {
    console.error("An error occurred:", error);
});
