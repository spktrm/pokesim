import * as fs from "fs";

export const formatId = "gen3ou";

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
    slp: 0,
    psn: 1,
    brn: 2,
    frz: 3,
    par: 4,
    tox: 5,
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
};
