import * as fs from "fs";
import {
    Ability,
    ID,
    Item,
    Learnset,
    ModdedDex,
    Move,
    Species,
    Type,
} from "@pkmn/dex";

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

// Helper function to create an enumeration from an array
function enumerate(arr: string[]): { [key: string]: number } {
    const enumeration: { [key: string]: number } = {};
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

async function getGenData(gen: number) {
    const format = `gen${gen}` as ID;
    const dex = new ModdedDex(format);
    const species = dex.species.all();
    const promises = species.map((species) => dex.learnsets.get(species.id));
    const learnsets = await Promise.all(promises);
    const data = {
        species: [...species],
        moves: [...dex.moves.all()],
        abilities: [...dex.abilities.all()],
        items: [...dex.items.all()],
        typechart: [...dex.types.all()],
        learnsets: [...learnsets],
    };
    return data;
}

type GenData = {
    species: Species[];
    moves: Move[];
    abilities: Ability[];
    items: Item[];
    typechart: Type[];
    learnsets: Learnset[];
};

function mapId<T extends { id: string; [key: string]: any }>(
    arr: T[],
): string[] {
    return arr.map((item) => item.id);
}

function formatData(data: GenData) {
    return {
        species: enumerate(mapId(data.species)),
        moves: enumerate(mapId(data.moves)),
        abilities: enumerate(mapId(data.abilities)),
        items: enumerate(mapId(data.items)),
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
    let volatileStatus = reduce(Array.from(volatileStatusSet));

    let weathers = extractPatterns(src, weathersPattern);
    weathers = reduce(weathers).map((t) => t.replace("raindance", "rain"));

    let sideConditions = extractPatterns(src, sideConditionsPattern);
    sideConditions = reduce(sideConditions);

    let terrain = extractPatterns(src, terrainPattern);
    terrain = reduce(terrain).map((t) => t.replace("terrain", ""));

    let pseudoweather = extractPatterns(src, pseudoWeatherPattern);
    pseudoweather = reduce(pseudoweather);

    // Create the data object
    const data = {
        pseudoWeather: enumerate(pseudoweather),
        volatileStatus: enumerate(volatileStatus),
        weathers: enumerate(weathers),
        terrain: enumerate(terrain),
        sideConditions: enumerate(sideConditions),
        ...formatData(await getGenData(9)),
    };

    // Write the data to a JSON file
    fs.writeFileSync("src/data.json", JSON.stringify(data, null, 2));
}

// Execute the main function
main().catch((error) => {
    console.error("An error occurred:", error);
});
