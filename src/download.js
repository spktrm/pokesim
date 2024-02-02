"use strict";
var __assign =
    (this && this.__assign) ||
    function () {
        __assign =
            Object.assign ||
            function (t) {
                for (var s, i = 1, n = arguments.length; i < n; i++) {
                    s = arguments[i];
                    for (var p in s)
                        if (Object.prototype.hasOwnProperty.call(s, p))
                            t[p] = s[p];
                }
                return t;
            };
        return __assign.apply(this, arguments);
    };
var __awaiter =
    (this && this.__awaiter) ||
    function (thisArg, _arguments, P, generator) {
        function adopt(value) {
            return value instanceof P
                ? value
                : new P(function (resolve) {
                      resolve(value);
                  });
        }
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) {
                try {
                    step(generator.next(value));
                } catch (e) {
                    reject(e);
                }
            }
            function rejected(value) {
                try {
                    step(generator["throw"](value));
                } catch (e) {
                    reject(e);
                }
            }
            function step(result) {
                result.done
                    ? resolve(result.value)
                    : adopt(result.value).then(fulfilled, rejected);
            }
            step(
                (generator = generator.apply(thisArg, _arguments || [])).next(),
            );
        });
    };
var __generator =
    (this && this.__generator) ||
    function (thisArg, body) {
        var _ = {
                label: 0,
                sent: function () {
                    if (t[0] & 1) throw t[1];
                    return t[1];
                },
                trys: [],
                ops: [],
            },
            f,
            y,
            t,
            g;
        return (
            (g = { next: verb(0), throw: verb(1), return: verb(2) }),
            typeof Symbol === "function" &&
                (g[Symbol.iterator] = function () {
                    return this;
                }),
            g
        );
        function verb(n) {
            return function (v) {
                return step([n, v]);
            };
        }
        function step(op) {
            if (f) throw new TypeError("Generator is already executing.");
            while ((g && ((g = 0), op[0] && (_ = 0)), _))
                try {
                    if (
                        ((f = 1),
                        y &&
                            (t =
                                op[0] & 2
                                    ? y["return"]
                                    : op[0]
                                      ? y["throw"] ||
                                        ((t = y["return"]) && t.call(y), 0)
                                      : y.next) &&
                            !(t = t.call(y, op[1])).done)
                    )
                        return t;
                    if (((y = 0), t)) op = [op[0] & 2, t.value];
                    switch (op[0]) {
                        case 0:
                        case 1:
                            t = op;
                            break;
                        case 4:
                            _.label++;
                            return { value: op[1], done: false };
                        case 5:
                            _.label++;
                            y = op[1];
                            op = [0];
                            continue;
                        case 7:
                            op = _.ops.pop();
                            _.trys.pop();
                            continue;
                        default:
                            if (
                                !((t = _.trys),
                                (t = t.length > 0 && t[t.length - 1])) &&
                                (op[0] === 6 || op[0] === 2)
                            ) {
                                _ = 0;
                                continue;
                            }
                            if (
                                op[0] === 3 &&
                                (!t || (op[1] > t[0] && op[1] < t[3]))
                            ) {
                                _.label = op[1];
                                break;
                            }
                            if (op[0] === 6 && _.label < t[1]) {
                                _.label = t[1];
                                t = op;
                                break;
                            }
                            if (t && _.label < t[2]) {
                                _.label = t[2];
                                _.ops.push(op);
                                break;
                            }
                            if (t[2]) _.ops.pop();
                            _.trys.pop();
                            continue;
                    }
                    op = body.call(thisArg, _);
                } catch (e) {
                    op = [6, e];
                    y = 0;
                } finally {
                    f = t = 0;
                }
            if (op[0] & 5) throw op[1];
            return { value: op[0] ? op[1] : void 0, done: true };
        }
    };
var __spreadArray =
    (this && this.__spreadArray) ||
    function (to, from, pack) {
        if (pack || arguments.length === 2)
            for (var i = 0, l = from.length, ar; i < l; i++) {
                if (ar || !(i in from)) {
                    if (!ar) ar = Array.prototype.slice.call(from, 0, i);
                    ar[i] = from[i];
                }
            }
        return to.concat(ar || Array.prototype.slice.call(from));
    };
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var dex_1 = require("@pkmn/dex");
// Define the base URL and the paths to download
var BASE_URL = "https://raw.githubusercontent.com/pkmn/ps/main/sim/{}.ts";
var PATHS = [
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
var URLS = PATHS.map(function (path) {
    return BASE_URL.replace("{}", path);
});
// Helper function to convert a string to an identifier
function toId(string) {
    return string.toLowerCase().replace(/[^a-z0-9]/g, "");
}
// Helper function to reduce an array to unique, sorted identifiers
function reduce(arr) {
    return Array.from(new Set(arr.map(toId))).sort();
}
function findDuplicates(arr) {
    return arr.filter(function (item, index) {
        return arr.indexOf(item) !== index;
    });
}
// Helper function to create an enumeration from an array
function enumerate(arr) {
    var enumeration = {};
    var dupes = findDuplicates(arr);
    arr.forEach(function (item, index) {
        enumeration[item] = index;
    });
    return enumeration;
}
// Function to fetch text content from a URL
function fetchText(url) {
    return __awaiter(this, void 0, void 0, function () {
        var response;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    return [4 /*yield*/, fetch(url)];
                case 1:
                    response = _a.sent();
                    return [2 /*return*/, response.text()];
            }
        });
    });
}
// Function to download all files and concatenate their content
function downloadAll(urls) {
    return __awaiter(this, void 0, void 0, function () {
        var requests;
        return __generator(this, function (_a) {
            requests = urls.map(function (url) {
                return fetchText(url);
            });
            return [2 /*return*/, Promise.all(requests)];
        });
    });
}
// Function to extract unique strings from source text based on a regular expression
function extractPatterns(src, pattern) {
    return __spreadArray([], src.matchAll(pattern), true).map(function (match) {
        return match[1];
    });
}
function getGenData(gen) {
    return __awaiter(this, void 0, void 0, function () {
        var format, dex, species, promises, learnsets, data;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    format = "gen".concat(gen);
                    dex = new dex_1.ModdedDex(format);
                    species = dex.species.all();
                    promises = species.map(function (species) {
                        return dex.learnsets.get(species.id);
                    });
                    return [4 /*yield*/, Promise.all(promises)];
                case 1:
                    learnsets = _a.sent();
                    data = {
                        species: __spreadArray([], species, true),
                        moves: __spreadArray([], dex.moves.all(), true),
                        abilities: __spreadArray([], dex.abilities.all(), true),
                        items: __spreadArray([], dex.items.all(), true),
                        typechart: __spreadArray([], dex.types.all(), true),
                        learnsets: __spreadArray([], learnsets, true),
                    };
                    return [2 /*return*/, data];
            }
        });
    });
}
function mapId(arr) {
    var hashmap = new Set();
    arr.map(function (item) {
        return hashmap.add(item.id);
    });
    return Array.from(hashmap);
}
var extraTokens = ["<UNK>", "<PAD>"];
function formatData(data) {
    var moveIds = __spreadArray(
        __spreadArray(
            [],
            data.moves.map(function (item) {
                if (item.id === "hiddenpower") {
                    var moveType = item.type.toLowerCase();
                    if (moveType === "normal") {
                        return item.id;
                    } else {
                        return "".concat(item.id).concat(moveType);
                    }
                } else if (item.id === "return") {
                    return "".concat(item.id, "102");
                } else {
                    return item.id;
                }
            }),
            true,
        ),
        ["return"],
        false,
    );
    return {
        species: enumerate(
            __spreadArray(
                __spreadArray(["switch-in"], extraTokens, true),
                mapId(data.species),
                true,
            ),
        ),
        moves: enumerate(extraTokens.concat(moveIds)),
        abilities: enumerate(
            __spreadArray(
                __spreadArray([], extraTokens, true),
                mapId(data.abilities),
                true,
            ),
        ),
        items: enumerate(
            __spreadArray(
                __spreadArray([], extraTokens, true),
                mapId(data.items),
                true,
            ),
        ),
    };
}
// The main function that executes the download and processing
function main() {
    return __awaiter(this, void 0, void 0, function () {
        var sources,
            src,
            weathersPattern,
            sideConditionsPattern,
            terrainPattern,
            pseudoWeatherPattern,
            volatileStatusPatterns,
            volatileStatusSet,
            volatileStatus,
            weathers,
            sideConditions,
            terrain,
            pseudoweather,
            data,
            _a,
            _b;
        return __generator(this, function (_c) {
            switch (_c.label) {
                case 0:
                    return [4 /*yield*/, downloadAll(URLS)];
                case 1:
                    sources = _c.sent();
                    src = sources.join("\n");
                    weathersPattern = /['|"]-weather['|"],\s*['|"](.*)['|"],/g;
                    sideConditionsPattern = /sideCondition:\s*['|"](.*)['|"],/g;
                    terrainPattern = /terrain:\s*['|"](.*)['|"],/g;
                    pseudoWeatherPattern = /pseudoWeather:\s['|"](.*?)['|"]/g;
                    volatileStatusPatterns = [
                        /removeVolatile\(['|"](.*?)['|"]\)/g,
                        /hasVolatile\(['|"](.*?)['|"]\)/g,
                        /volatiles\[['|"](.*?)['|"]\]/g,
                        /volatiles\.(.*?)[\[\)| ]/g,
                        /volatileStatus:\s*['|"](.*)['|'],/g,
                    ];
                    volatileStatusSet = new Set();
                    // Process each pattern and add the results to the Set
                    volatileStatusPatterns.forEach(function (pattern) {
                        var matches = src.match(pattern);
                        if (matches) {
                            matches.forEach(function (match) {
                                var cleanedMatch = match
                                    .replace(pattern, "$1") // Extract the captured group
                                    .replace(/['|"|\[|\]|\(|\)|,]/g, "") // Clean up any extra characters
                                    .trim();
                                if (cleanedMatch) {
                                    volatileStatusSet.add(cleanedMatch);
                                }
                            });
                        }
                    });
                    volatileStatus = reduce(Array.from(volatileStatusSet));
                    weathers = extractPatterns(src, weathersPattern);
                    weathers = reduce(weathers).map(function (t) {
                        return t.replace("raindance", "rain");
                    });
                    sideConditions = extractPatterns(
                        src,
                        sideConditionsPattern,
                    );
                    sideConditions = reduce(sideConditions);
                    terrain = extractPatterns(src, terrainPattern);
                    terrain = reduce(terrain).map(function (t) {
                        return t.replace("terrain", "");
                    });
                    pseudoweather = extractPatterns(src, pseudoWeatherPattern);
                    pseudoweather = reduce(pseudoweather);
                    _a = [
                        {
                            pseudoWeather: enumerate(pseudoweather),
                            volatileStatus: enumerate(volatileStatus),
                            weathers: enumerate(weathers),
                            terrain: enumerate(terrain),
                            sideConditions: enumerate(sideConditions),
                        },
                    ];
                    _b = formatData;
                    return [4 /*yield*/, getGenData(9)];
                case 2:
                    data = __assign.apply(
                        void 0,
                        _a.concat([_b.apply(void 0, [_c.sent()])]),
                    );
                    // Write the data to a JSON file
                    fs.writeFileSync(
                        "src/data.json",
                        JSON.stringify(data, null, 2),
                    );
                    return [2 /*return*/];
            }
        });
    });
}
// Execute the main function
main().catch(function (error) {
    console.error("An error occurred:", error);
});
