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

const typeMapping = Object.fromEntries(
    ((formatDex.types as any).all() as dex.Type[])
        .sort((a, b) => a.id.localeCompare(b.id))
        .filter((type) => type.isNonstandard !== "Future")
        .map(({ id, damageTaken }, index) => [
            id,
            {
                index,
                damageTaken,
            },
        ]),
);

const data = fs.readFileSync("./src/data.json");
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
} = JSON.parse(data.toString());

const statusMapping: { [k: string]: number } = {
    slp: 1,
    psn: 2,
    brn: 3,
    frz: 4,
    par: 5,
    tox: 6,
};

const boostsMapping = {
    atk: 0,
    def: 1,
    spa: 2,
    spd: 3,
    spe: 4,
    accuracy: 5,
    evasion: 6,
};

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
};
