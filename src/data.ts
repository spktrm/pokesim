import * as fs from "fs";

export const formatId = "gen3randombattle";

import * as dex from "@pkmn/dex";
import { Generations } from "@pkmn/data";

const generations = new Generations(dex.Dex);
const formatDex = generations.dex.mod(formatId.slice(0, 4) as dex.GenID);

const maxPP = Object.fromEntries(
    (formatDex.moves as any)
        .all()
        .map((move: { id: any; pp: number }) => [move.id, (move.pp * 8) / 5]),
);

const data = fs.readFileSync("./src/data/data.json");
const {
    sideConditions: sideConditionsMapping,
    weathers: weatherMapping,
    pseudoWeather: pseudoWeatherMapping,
    terrain: terrainMapping,
    volatileStatus: volatileStatusMapping,
    species: pokemonMapping,
    items: itemMapping,
    abilities: abilityMapping,
    moves: moveMapping,
    statuses: statusMapping,
    boosts: boostsMapping,
    types: typeMapping,
    genders: genderMapping,
} = JSON.parse(data.toString());

const pseudoWeatherVectorSize = 3 * Object.values(pseudoWeatherMapping).length;
const weatherVectorSize = 3;
const terrainVectorSize = 3;

const contextVectorSize =
    2 *
        (Object.values(volatileStatusMapping).length +
            Object.keys(sideConditionsMapping).length +
            Object.keys(boostsMapping).length) +
    (pseudoWeatherVectorSize + weatherVectorSize + terrainVectorSize);

for (const type of Object.keys(typeMapping)) {
    maxPP[`hiddenpower${type}`] = maxPP["hiddenpower"];
}
maxPP["return102"] = maxPP["return"];

export {
    pokemonMapping,
    abilityMapping,
    moveMapping,
    itemMapping,
    sideConditionsMapping,
    terrainMapping,
    weatherMapping,
    volatileStatusMapping,
    pseudoWeatherMapping,
    statusMapping,
    boostsMapping,
    maxPP,
    genderMapping,
    contextVectorSize,
    pseudoWeatherVectorSize,
    weatherVectorSize,
    terrainVectorSize,
    typeMapping,
};
