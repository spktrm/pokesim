import { Pokemon, Side } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import {
    abilityMapping,
    boostsMapping,
    itemMapping,
    moveMapping,
    pokemonMapping,
    pseudoWeatherMapping,
    sideConditionsMapping,
    statusMapping,
    terrainMapping,
    volatileStatusMapping,
    weatherMapping,
} from "./data";
import { SideConditions } from "./types";
import { BoostID } from "@pkmn/dex";
import { BattlesHandler } from "./helpers";

let stateSize: number | undefined = undefined;

function formatKey(key: string): string {
    // Convert to lowercase and remove spaces and non-alphanumeric characters
    return key.toLowerCase().replace(/[\W_]+/g, "");
}

function getMappingValue(
    pokemon: AnyObject,
    mapping: AnyObject,
    key: string
): number {
    let suffix: string = "";
    if (key === "asone") {
        if (pokemon.baseSpeciesForme === "Calyrex-Shadow") {
            suffix = "spectrier";
        } else if (pokemon.baseSpeciesForme === "Calyrex-Ice") {
            suffix = "glastrier";
        }
        key = key + suffix;
    }
    return mapping[key];
}

function logKeyError(key: string, mapping: AnyObject) {
    if (key !== undefined && key !== "") {
        console.error(`${key} not in ${JSON.stringify(mapping).slice(0, 30)}`);
    }
}

function getMappingValueWrapper(
    pokemon: AnyObject,
    mapping: AnyObject,
    key: string
): number {
    const stringKey = key ?? "";
    if (stringKey === "") {
        logKeyError(key, mapping);
        return -1;
    }
    const mappedValue = getMappingValue(pokemon, mapping, key);
    if (mappedValue === undefined) {
        logKeyError(key, mapping);
        return -1;
    } else {
        return mappedValue;
    }
}

function getPublicPokemon(
    pokemon: AnyObject,
    active: boolean,
    buckets: number = 1024
): Int8Array {
    let moveTokens = [];
    for (let i = 0; i < 4; i++) {
        moveTokens.push(
            getMappingValueWrapper(pokemon, moveMapping, pokemon.moves[i])
        );
    }
    const formatedPokemonName = formatKey(pokemon.name);
    const speciesToken = getMappingValueWrapper(
        pokemon,
        pokemonMapping,
        formatedPokemonName
    );
    const itemToken = getMappingValueWrapper(
        pokemon,
        itemMapping,
        pokemon.item
    );
    const abilityToken = getMappingValueWrapper(
        pokemon,
        abilityMapping,
        pokemon.ability
    );
    const hpToken =
        pokemon.maxhp !== undefined
            ? Math.floor(buckets * (pokemon.hp / pokemon.maxhp))
            : buckets;
    const pokemonArray = [
        speciesToken,
        itemToken,
        abilityToken,
        hpToken,
        active ? 1 : 0,
        pokemon.fainted ? 1 : 0,
        statusMapping[pokemon.status] ?? -1,
        ...moveTokens,
    ];
    return new Int8Array(new Int16Array(pokemonArray).buffer);
}

function getPrivatePokemon(
    pokemon: AnyObject,
    buckets: number = 1024
): Int8Array {
    let moveTokens = [];
    for (let i = 0; i < 4; i++) {
        moveTokens.push(
            getMappingValueWrapper(pokemon, moveMapping, pokemon.moves[i])
        );
    }
    const formatedPokemonName = formatKey(pokemon.name);
    const speciesToken = getMappingValueWrapper(
        pokemon,
        pokemonMapping,
        formatedPokemonName
    );
    const itemToken = getMappingValueWrapper(
        pokemon,
        itemMapping,
        pokemon.item
    );
    const abilityToken = getMappingValueWrapper(
        pokemon,
        abilityMapping,
        pokemon.ability
    );
    const hpToken =
        pokemon.maxhp !== undefined
            ? Math.floor(buckets * (pokemon.hp / pokemon.maxhp))
            : buckets;
    const pokemonArray = [
        speciesToken,
        itemToken,
        abilityToken,
        hpToken,
        pokemon.active ? 1 : 0,
        pokemon.fainted ? 1 : 0,
        statusMapping[pokemon.status] ?? -1,
        ...moveTokens,
    ];
    return new Int8Array(new Int16Array(pokemonArray).buffer);
}

const fillPokemonObj = { name: "", moves: [] };
const fillPokemonArray = getPublicPokemon(fillPokemonObj, false);
const boostsEntries = Object.entries(boostsMapping);

export class Int8State {
    handler: BattlesHandler;
    playerIndex: number;
    workerIndex: number;
    done: number;
    reward: number;
    constructor(
        handler: BattlesHandler,
        playerIndex: number,
        workerIndex: number,
        done: number,
        reward: number
    ) {
        this.handler = handler;
        this.playerIndex = playerIndex;
        this.workerIndex = workerIndex;
        this.done = done;
        this.reward = reward;
    }

    actionToVector(actionLine: string): Int8Array {
        const room = this.handler.getMyBattle();
        if (actionLine === undefined) {
            return new Int8Array(new Int16Array([-1, -1, -1]).buffer);
        }
        const splitString = actionLine.split("|");
        const actionType = splitString[1];
        const user = splitString[2];
        const action: string =
            actionType === "move" ? formatKey(splitString[3]) : actionType;
        const p1keys = room.p1.team.map((x) => x.ident.toString()).slice(0, 6);
        const p2keys = room.p2.team.map((x) => x.ident.toString()).slice(0, 6);
        const keys = [...p1keys, ...Array(6 - p1keys.length)].concat([
            ...p2keys,
            ...Array(6 - p1keys.length),
        ]);
        const userIndex = keys.indexOf(user);
        const actionIndex = moveMapping[action] ?? -2;
        const actionVector = [userIndex >= 6 ? 1 : 0, userIndex, actionIndex];
        return new Int8Array(new Int16Array(actionVector).buffer);
    }

    getMyBoosts(): Int8Array {
        const side = this.getMyPublicSide();
        return this.getBoosts(side.active);
    }
    getOppBoosts(): Int8Array {
        const side = this.getOppSide();
        return this.getBoosts(side.active);
    }

    getBoosts(actives: (Pokemon | null)[]): Int8Array {
        const boostsVector = new Int8Array(
            actives.length * boostsEntries.length
        );
        boostsVector.fill(0);

        for (const [activeIndex, activePokemon] of actives.entries()) {
            if (activePokemon !== null) {
                for (const [boost, value] of Object.entries(
                    activePokemon.boosts
                )) {
                    boostsVector[
                        activeIndex + boostsMapping[boost as BoostID]
                    ] = value;
                }
            }
        }
        return boostsVector;
    }

    getField(): Int8Array {
        const field = this.handler.getMyBattle().field;
        const fieldVector = new Int8Array(9 + 6);
        fieldVector.fill(-1);
        for (const [index, [name, pseudoWeather]] of Object.entries(
            field.pseudoWeather
        ).entries()) {
            fieldVector[index] = pseudoWeatherMapping[name];
            fieldVector[index + 1] = pseudoWeather.minDuration;
            fieldVector[index + 2] = pseudoWeather.maxDuration;
        }
        fieldVector[9] = weatherMapping[field.weatherState.id] ?? -1;
        fieldVector[10] = field.weatherState.minDuration;
        fieldVector[11] = field.weatherState.maxDuration;
        fieldVector[12] = terrainMapping[field.terrainState.id] ?? -1;
        fieldVector[13] = field.terrainState.minDuration;
        fieldVector[14] = field.terrainState.maxDuration;
        return fieldVector;
    }

    getMyVolatileStatus(): Int8Array {
        const side = this.getMyPublicSide();
        return this.getVolatileStatus(side.active);
    }
    getOppVolatileStatus(): Int8Array {
        const side = this.getOppSide();
        return this.getVolatileStatus(side.active);
    }

    getVolatileStatus(actives: (Pokemon | null)[]): Int8Array {
        const volatileStatusVector = new Int8Array(actives.length * 20);
        volatileStatusVector.fill(-1);
        for (const [activeIndex, activePokemon] of actives.entries()) {
            if (activePokemon !== null) {
                for (const [volatileIndex, volatileStatus] of Object.values(
                    activePokemon.volatiles
                ).entries()) {
                    volatileStatusVector[activeIndex + volatileIndex] =
                        volatileStatusMapping[volatileStatus.id];
                    volatileStatusVector[activeIndex + volatileIndex + 1] =
                        volatileStatus.level ?? 0;
                }
            }
        }
        return volatileStatusVector;
    }

    getMySideConditions(): Int8Array {
        const side = this.getMyPublicSide();
        return this.getSideConditions(side.sideConditions);
    }
    getOppSideConditions(): Int8Array {
        const side = this.getOppSide();
        return this.getSideConditions(side.sideConditions);
    }

    getSideConditions(sideConditions: SideConditions): Int8Array {
        const sideConditionVector = new Int8Array(
            Object.keys(sideConditionsMapping).length
        );
        for (const [name, sideCondition] of Object.entries(sideConditions)) {
            sideConditionVector[sideConditionsMapping[name]] =
                sideCondition.level;
        }
        return sideConditionVector;
    }

    getMyPrivateSide(): AnyObject {
        return (this.handler.getMyBattle().request ?? {}) as AnyObject;
    }

    getMyPublicSide(): Side {
        return this.handler.getMyBattle().sides[this.playerIndex];
    }

    getOppSide(): Side {
        return this.handler.getMyBattle().sides[1 - this.playerIndex];
    }

    getMyPrivateTeam(): Int8Array {
        const side = this.getMyPublicSide();
        const request = this.getMyPrivateSide();
        return this.getPrivateTeam(request, side.team);
    }

    getMyPublicTeam(): Int8Array {
        const side = this.getMyPublicSide();
        return this.getPublicTeam(side.team, side.active);
    }

    getOppTeam(): Int8Array {
        const side = this.getOppSide();
        return this.getPublicTeam(side.team, side.active);
    }

    getPublicTeam(team: Pokemon[], actives: (Pokemon | null)[]): Int8Array {
        const teamArray = new Int8Array(fillPokemonArray.length * 6);
        const activeIdents = [];

        for (let i = 0; i < actives.length; i++) {
            const ident = (actives[i] ?? {}).ident ?? "";
            if (ident !== undefined) {
                activeIdents.push(ident);
            }
        }
        for (let i = 0; i < 6; i++) {
            if (team[i] === undefined) {
                teamArray.set(fillPokemonArray, i * fillPokemonArray.length);
            } else {
                teamArray.set(
                    getPublicPokemon(
                        team[i],
                        activeIdents.includes(team[i].ident)
                    ),
                    i * fillPokemonArray.length
                );
            }
        }
        return teamArray;
    }

    getPrivateTeam(request: AnyObject, team: Pokemon[]): Int8Array {
        const requestSide = (request?.side ?? { pokemon: [] }).pokemon;
        const teamArray = new Int8Array(fillPokemonArray.length * 6);
        for (let i = 0; i < 6; i++) {
            const amalgam = {
                ...fillPokemonObj,
                ...team[i],
                ...(requestSide[i] ?? {}),
            };
            teamArray.set(
                getPrivatePokemon(amalgam),
                i * fillPokemonArray.length
            );
        }
        return teamArray;
    }

    getLegalMask(): Int8Array {
        const request = this.handler.getMyBattle().request as AnyObject;
        const mask = new Int8Array(10);
        if (request === undefined || this.done) {
            mask.fill(1);
            // mask[4] = 0;
        } else {
            mask.fill(0);

            if (request.wait) {
            } else if (request.forceSwitch) {
                const pokemon = request.side.pokemon;
                const forceSwitchLength = request.forceSwitch.length;
                const isReviving = !!pokemon[0].reviving;

                for (let j = 1; j <= 6; j++) {
                    const currentPokemon = pokemon[j - 1];
                    if (
                        currentPokemon &&
                        j > forceSwitchLength &&
                        (isReviving ? 1 : 0) ^
                            (currentPokemon.condition.endsWith(" fnt") ? 0 : 1)
                    ) {
                        mask[j + 3] = 1;
                    }
                }
            } else if (request.active) {
                const pokemon = request.side.pokemon;
                const active = request.active[0];
                const possibleMoves = active.moves ?? [];
                const canSwitch = [];

                for (let j = 1; j <= possibleMoves.length; j++) {
                    const currentMove = possibleMoves[j - 1];
                    if (!currentMove.disabled) {
                        mask[j - 1] = 1;
                    }
                }

                for (let j = 1; j <= 6; j++) {
                    const currentPokemon = pokemon[j - 1];
                    if (
                        currentPokemon &&
                        !currentPokemon.active &&
                        !currentPokemon.condition.endsWith(" fnt")
                    ) {
                        canSwitch.push(j);
                    }
                }

                const switches =
                    active.trapped || active.maybeTrapped ? [] : canSwitch;

                for (let i = 0; i < switches.length; i++) {
                    const slot = switches[i];
                    mask[slot + 3] = 1;
                }
            }
        }

        return mask;
    }

    getState(): Int8Array {
        const turn = Math.min(127, this.handler.getMyBattle().turn - 1 ?? 0);
        const actionLines = this.handler.getTurnLines(turn);
        const data = [
            new Int8Array([
                this.workerIndex,
                this.playerIndex,
                this.done,
                this.reward,
                turn,
            ]),
            this.getMyPrivateTeam(),
            this.getMyPublicTeam(),
            this.getOppTeam(),
            this.getMySideConditions(),
            this.getOppSideConditions(),
            this.getMyVolatileStatus(),
            this.getOppVolatileStatus(),
            this.getMyBoosts(),
            this.getOppBoosts(),
            this.getField(),
            this.actionToVector(actionLines[0]),
            this.actionToVector(actionLines[1]),
            this.actionToVector(actionLines[2]),
            this.actionToVector(actionLines[3]),
            this.getLegalMask(),
        ];
        if (stateSize === undefined) {
            stateSize = data.reduce(
                (accumulator, currentValue) =>
                    accumulator + currentValue.length,
                0
            );
        }
        const state = new Int8Array(stateSize);
        let offset = 0;
        for (const datum of data) {
            state.set(datum, offset);
            offset += datum.length;
        }
        this.handler.getMyBattle().request = undefined;
        return state;
    }
}
