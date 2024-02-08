import { Battle, Pokemon, Side } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import {
    abilityMapping,
    boostsMapping,
    itemMapping,
    moveMapping,
    pokemonMapping,
    pseudoWeatherMapping,
    pseudoWeatherVectorSize,
    sideConditionsMapping,
    statusMapping,
    terrainMapping,
    terrainVectorSize,
    volatileStatusMapping,
    weatherMapping,
    weatherVectorSize,
} from "./data";
import { SideConditions } from "./types";
import { BoostID } from "@pkmn/dex";
import { BattlesHandler, historyVectorSize } from "./helpers";

let stateSize: number | undefined = undefined;

function formatKey(key: string): string {
    return key.startsWith("<") ? key : key.toLowerCase().replace(/[\W_]+/g, "");
}

function getMappingValue(
    pokemon: AnyObject,
    mapping: AnyObject,
    key: string,
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

function getMappingValueWrapper(
    pokemon: AnyObject,
    mapping: AnyObject,
    key: string,
): number {
    const mappedValue = getMappingValue(pokemon, mapping, key);
    if (mappedValue === undefined) {
        throw new Error(
            `${key} not in ${JSON.stringify(mapping).slice(0, 20)}`,
        );
    }

    return mappedValue;
}

export function getPublicPokemon(
    battle: Battle | undefined,
    pokemon: AnyObject,
    active: boolean,
    sideToken: number,
    buckets: number = 1024,
): Int8Array {
    let moveTokens = [];
    let movePpLeft = [];
    let movePpMax = Array(4);
    movePpMax.fill(0);

    const moveSlots = pokemon.moveSlots ?? [];
    const moves = pokemon.moves ?? [];

    for (let i = 0; i < 4; i++) {
        moveTokens.push(
            getMappingValueWrapper(pokemon, moveMapping, moves[i] ?? "<UNK>"),
        );
        const ppUsed = (moveSlots[i] ?? {})?.ppUsed ?? 0;
        movePpLeft.push(ppUsed);
        if (battle !== undefined) {
            movePpMax[i] = ~~battle.gen.dex.moves.get(moves[i]).pp * (8 / 5);
        }
    }

    const lastMove =
        (pokemon.lastMove === "" && !!pokemon.newlySwitched) ||
        pokemon.lastMove === "switch-in"
            ? "<SWITCH>"
            : pokemon.lastMove === "" || pokemon.lastMove === undefined
              ? "<NONE>"
              : pokemon.lastMove;
    const lastMoveToken = getMappingValueWrapper(
        pokemon,
        moveMapping,
        lastMove,
    );

    const formatedPokemonName =
        pokemon.name === undefined ? "<UNK>" : formatKey(pokemon.name);
    const speciesToken = getMappingValueWrapper(
        pokemon,
        pokemonMapping,
        formatedPokemonName,
    );

    const item =
        pokemon.item === "" || pokemon.item === undefined
            ? "<UNK>"
            : pokemon.item;
    const itemToken = getMappingValueWrapper(pokemon, itemMapping, item);

    const ability =
        pokemon.ability === "" || pokemon.ability === undefined
            ? "<UNK>"
            : pokemon.ability;
    const abilityToken = getMappingValueWrapper(
        pokemon,
        abilityMapping,
        ability,
    );

    const hpToken =
        pokemon.maxhp !== undefined && pokemon.maxhp > 0
            ? Math.floor(buckets * (pokemon.hp / pokemon.maxhp))
            : buckets;

    const pokemonArray = [
        speciesToken,
        itemToken,
        abilityToken,
        hpToken,
        active ? 1 : 0,
        pokemon.fainted ? 1 : 0,
        statusMapping[pokemon.status],
        lastMoveToken,
        1,
        sideToken,
        pokemon?.statusState?.sleepTurns ?? 0,
        pokemon?.statusState?.toxicTurns ?? 0,
        ...movePpLeft,
        ...movePpMax,
        ...moveTokens,
    ];
    return new Int8Array(new Int16Array(pokemonArray).buffer);
}

export function getPrivatePokemon(
    battle: Battle | undefined,
    pokemon: AnyObject,
    buckets: number = 1024,
): Int8Array {
    let moveTokens = [];
    let movePpLeft = [];
    let movePpMax = Array(4);
    movePpMax.fill(0);

    const moveSlots = pokemon.moveSlots ?? [];
    const moves = pokemon.moves ?? [];

    for (let i = 0; i < 4; i++) {
        moveTokens.push(
            getMappingValueWrapper(pokemon, moveMapping, moves[i] ?? "<PAD>"),
        );
        const ppUsed = (moveSlots[i] ?? {})?.ppUsed ?? 0;
        movePpLeft.push(ppUsed);
        if (battle !== undefined) {
            movePpMax[i] = ~~battle.gen.dex.moves.get(moves[i]).pp * (8 / 5);
        }
    }

    const lastMove =
        (pokemon.lastMove === "" && !!pokemon.newlySwitched) ||
        pokemon.lastMove === "switch-in"
            ? "<SWITCH>"
            : pokemon.lastMove === "" || pokemon.lastMove === undefined
              ? "<NONE>"
              : pokemon.lastMove;
    const lastMoveToken = getMappingValueWrapper(
        pokemon,
        moveMapping,
        lastMove,
    );

    const formatedPokemonName =
        pokemon.name === undefined ? "<UNK>" : formatKey(pokemon.name);
    const speciesToken = getMappingValueWrapper(
        pokemon,
        pokemonMapping,
        formatedPokemonName,
    );

    const item =
        pokemon.item === "" || pokemon.item === undefined
            ? "<UNK>"
            : pokemon.item;
    const itemToken = getMappingValueWrapper(pokemon, itemMapping, item);

    const ability =
        pokemon.ability === "" || pokemon.ability === undefined
            ? "<UNK>"
            : pokemon.ability;
    const abilityToken = getMappingValueWrapper(
        pokemon,
        abilityMapping,
        ability,
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
        statusMapping[pokemon.status],
        lastMoveToken,
        0,
        1,
        pokemon?.statusState?.sleepTurns ?? 0,
        pokemon?.statusState?.toxicTurns ?? 0,
        ...movePpLeft,
        ...movePpMax,
        ...moveTokens,
    ];
    return new Int8Array(new Int16Array(pokemonArray).buffer);
}

const paddedPokemonObj = {
    name: "<PAD>",
    item: "<PAD>",
    ability: "<PAD>",
    lastMove: "<PAD>",
    status: "<NULL>",
    moves: ["<PAD>", "<PAD>", "<PAD>", "<PAD>"],
};
const paddedPokemonArray = getPublicPokemon(
    undefined,
    paddedPokemonObj,
    false,
    1,
);

const unknownPokemonObj = {
    name: "<UNK>",
    item: "<UNK>",
    ability: "<UNK>",
    lastMove: "<UNK>",
    status: "<NULL>",
    moves: ["<UNK>", "<UNK>", "<UNK>", "<UNK>"],
};
const unknownPokemonArray0 = getPublicPokemon(
    undefined,
    unknownPokemonObj,
    false,
    0,
);
const unknownPokemonArray1 = getPublicPokemon(
    undefined,
    unknownPokemonObj,
    false,
    1,
);

const boostsEntries = Object.entries(boostsMapping);

const pseudoWeatherOffset = 0;

const weatherOffset = pseudoWeatherOffset + pseudoWeatherVectorSize;

const terrainOffset = weatherOffset + weatherVectorSize;

const fieldVectorLength =
    pseudoWeatherVectorSize + weatherVectorSize + terrainVectorSize;

export class Int8State {
    handler: BattlesHandler;
    playerIndex: number;
    workerIndex: number;
    done: number;
    reward: number;
    constructor(args: {
        handler: BattlesHandler;
        playerIndex: number;
        workerIndex: number;
        done: number;
        reward: number;
    }) {
        const { handler, playerIndex, workerIndex, done, reward } = args;
        this.handler = handler;
        this.playerIndex = playerIndex;
        this.workerIndex = workerIndex;
        this.done = done;
        this.reward = reward;
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
            actives.length * boostsEntries.length,
        );
        boostsVector.fill(0);

        for (const [activeIndex, activePokemon] of actives.entries()) {
            if (activePokemon !== null) {
                for (const [boost, value] of Object.entries(
                    activePokemon.boosts,
                )) {
                    boostsVector[
                        activeIndex + boostsMapping[boost as BoostID]
                    ] = value;
                }
            }
        }
        return boostsVector;
    }

    getFieldVectors(): Int8Array[] {
        const field = this.handler.getMyBattle().field;

        const pseudoWeatherVector = new Int8Array(pseudoWeatherVectorSize);
        pseudoWeatherVector.fill(0);

        const weatherVector = new Int8Array(weatherVectorSize);
        weatherVector.fill(0);

        const terrainVector = new Int8Array(terrainVectorSize);
        terrainVector.fill(0);

        if (Object.keys(field.pseudoWeather).length > 0) {
            for (const [name, datum] of Object.entries(field.pseudoWeather)) {
                const vectorIndex = pseudoWeatherMapping[name];
                pseudoWeatherVector[vectorIndex] = 1;
                pseudoWeatherVector[vectorIndex + 1] = datum.minDuration;
                pseudoWeatherVector[vectorIndex + 2] = datum.maxDuration;
            }
        } else {
            const vectorIndex = pseudoWeatherMapping["<NULL>"];
            pseudoWeatherVector[vectorIndex] = 1;
            // pseudoWeatherVector[vectorIndex + 1] = 0;
            // pseudoWeatherVector[vectorIndex + 2] = 0;
        }

        if (field.weather !== undefined) {
            const vectorIndex = 3 * weatherMapping[field.weatherState.id];
            weatherVector[vectorIndex] = 1;
            weatherVector[vectorIndex + 1] = field.weatherState.minDuration;
            weatherVector[vectorIndex + 2] = field.weatherState.maxDuration;
        } else {
            const vectorIndex = weatherMapping["<NULL>"];
            weatherVector[vectorIndex] = 1;
            // weatherVector[vectorIndex + 1] = 0;
            // weatherVector[vectorIndex + 2] = 0;
        }

        if (field.terrain !== undefined) {
            const vectorIndex = 3 * terrainMapping[field.terrainState.id];
            terrainVector[vectorIndex] = 1;
            terrainVector[vectorIndex + 1] = field.terrainState.minDuration;
            terrainVector[vectorIndex + 2] = field.terrainState.maxDuration;
        } else {
            const vectorIndex = terrainMapping["<NULL>"];
            terrainVector[vectorIndex] = 1;
            // terrainVector[vectorIndex + 1] = 0;
            // terrainVector[vectorIndex + 2] = 0;
        }

        return [pseudoWeatherVector, weatherVector, terrainVector];
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
        const volatileStatusVector = new Int8Array(
            actives.length * Object.values(volatileStatusMapping).length,
        );
        volatileStatusVector.fill(0);
        for (const [activeIndex, activePokemon] of actives.entries()) {
            if (activePokemon !== null) {
                if (Object.keys(activePokemon.volatiles).length > 0) {
                    for (const volatileStatus of Object.values(
                        activePokemon.volatiles,
                    )) {
                        const vectorIndex =
                            volatileStatusMapping[volatileStatus.id];
                        if (vectorIndex === undefined) {
                            throw new Error(`${volatileStatus.id} not found`);
                        }
                        volatileStatusVector[activeIndex + vectorIndex] =
                            volatileStatus.level ?? 0;
                    }
                } else {
                    const vectorIndex = volatileStatusMapping["<NULL>"];
                    volatileStatusVector[activeIndex + vectorIndex] = 1;
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
            Object.keys(sideConditionsMapping).length,
        );
        sideConditionVector.fill(0);
        if (Object.keys(sideConditions).length > 0) {
            for (const [name, sideCondition] of Object.entries(
                sideConditions,
            )) {
                const vectorIndex = sideConditionsMapping[name];
                if (vectorIndex === undefined) {
                    throw new Error(`${name} not found`);
                }
                sideConditionVector[vectorIndex] = sideCondition.level;
            }
        } else {
            const vectorIndex = sideConditionsMapping["<NULL>"];
            sideConditionVector[vectorIndex] = 1;
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
        const request = this.getMyPrivateSide();
        return this.getPrivateTeam(request);
    }

    getMyPublicTeam(): Int8Array {
        const side = this.getMyPublicSide();
        return this.getPublicTeam(side);
    }

    getOppTeam(): Int8Array {
        const side = this.getOppSide();
        return this.getPublicTeam(side);
    }

    getPublicTeam(side: Side): Int8Array {
        const request = this.getMyPrivateSide();
        const isMe = request.side?.name === side.name ? 1 : 0;
        const team = side.team;
        const actives = side.active;

        const teamArray = new Int8Array(paddedPokemonArray.length * 6);
        const activeIdents = [];

        for (let i = 0; i < actives.length; i++) {
            const ident = (actives[i] ?? {}).ident ?? "";
            if (ident !== undefined) {
                activeIdents.push(ident);
            }
        }
        for (let i = 0; i < 6; i++) {
            if (team[i] === undefined) {
                if (i < side.totalPokemon) {
                    if (isMe) {
                        teamArray.set(
                            unknownPokemonArray1,
                            i * paddedPokemonArray.length,
                        );
                    } else {
                        teamArray.set(
                            unknownPokemonArray0,
                            i * paddedPokemonArray.length,
                        );
                    }
                } else {
                    teamArray.set(
                        paddedPokemonArray,
                        i * paddedPokemonArray.length,
                    );
                }
            } else {
                const ident = team[i].ident;
                teamArray.set(
                    getPublicPokemon(
                        this.handler.getMyBattle(),
                        team[i],
                        activeIdents.includes(ident),
                        isMe,
                    ),
                    i * paddedPokemonArray.length,
                );
            }
        }
        return teamArray;
    }

    getPrivateTeam(request: AnyObject): Int8Array {
        const requestSide = (request?.side ?? { pokemon: [] }).pokemon;
        const teamArray = new Int8Array(paddedPokemonArray.length * 6);

        const privateId =
            (this.handler.getMyBattle().request?.side ?? {})?.id ?? "";
        const privateIndex = parseInt(privateId.slice(1)) - 1;

        for (let i = 0; i < 6; i++) {
            const entity = requestSide[i];
            const ident = (entity ?? {}).ident ?? "";
            const amalgam = {
                ...paddedPokemonObj,
                ...(entity === undefined ? {} : this.handler.getPokemon(ident)),
            };
            teamArray.set(
                getPrivatePokemon(this.handler.getMyBattle(), amalgam),
                i * paddedPokemonArray.length,
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

                for (let j = 0; j < 6; j++) {
                    const currentPokemon = pokemon[j];
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

                for (const switchIndex of switches) {
                    mask[switchIndex + 4] = 1;
                }
            } else if (request.teamPreview) {
                const pokemon = request.side.pokemon;
                const canSwitch = [];

                for (let j = 0; j < 6; j++) {
                    const currentPokemon = pokemon[j];
                    if (currentPokemon) {
                        canSwitch.push(j);
                    }
                }

                for (const switchIndex of canSwitch) {
                    mask[switchIndex + 4] = 1;
                }
            }
        }

        return mask;
    }

    updatetHistoryVectors(): void {
        // let offset = 0;

        for (const [key, moveStore] of Object.entries(
            this.handler.damageInfos,
        )) {
            if (moveStore.hasVector) {
                continue;
            }
            const {
                // user,
                // target,
                // userSide,
                // targetSide,
                order,
                isCritical,
                damage,
                effectiveness,
                missed,
                moveName,
                targetFainted,
                moveCounter,
                switchCounter,
            } = moveStore.context;
            // const isMe = this.playerIndex === userSide ? 1 : 0;
            // const userProps = this.handler.getPokemon(userSide, user);
            // const targetProps = this.handler.getPokemon(targetSide, target);

            const action =
                moveName === "switch-in"
                    ? "<SWITCH>"
                    : formatKey(moveName ?? "") ?? "";

            const moveArray = new Int8Array(
                new Int16Array([
                    1,
                    isCritical ? 1 : 0,
                    effectiveness,
                    missed ? 1 : 0,
                    targetFainted ? 1 : 0,
                    Math.floor(2047 * (1 + Math.max(-1, Math.min(1, damage)))),
                    moveCounter,
                    switchCounter,
                    moveMapping[action],
                    order,
                ]).buffer,
            );
            // const history: Array<Int8Array> = [
            //     contextVector,
            //     userSide === this.playerIndex
            //         ? getPublicPokemon(userProps, true, isMe)
            //         : getPrivatePokemon(userProps),
            //     targetSide === this.playerIndex
            //         ? getPublicPokemon(targetProps, true, isMe)
            //         : getPrivatePokemon(targetProps),
            //     moveArray,
            // ];
            // offset = 0;
            // for (const datum of history) {
            //     this.handler.damageInfos[key].vector.set(datum, offset);
            //     offset += datum.length;
            // }
            const offset = historyVectorSize - moveArray.length;
            this.handler.damageInfos[key].vector.set(moveArray, offset);
            moveStore.hasVector = true;
        }
    }

    getContextVectors(): Int8Array[] {
        return [
            this.getMySideConditions(),
            this.getOppSideConditions(),
            this.getMyVolatileStatus(),
            this.getOppVolatileStatus(),
            this.getMyBoosts(),
            this.getOppBoosts(),
            ...this.getFieldVectors(),
        ];
    }

    getState(): Int8Array {
        const turn = this.handler.getBattleTurn();
        this.handler.getAggregatedTurnLines();

        const heuristicAction = this.done
            ? -1
            : this.handler.getHeuristicActionIndex(
                  this.playerIndex,
                  this.workerIndex,
              );
        const legalMask = this.getLegalMask();

        const contextVectors = this.getContextVectors();

        this.updatetHistoryVectors();

        const damageInfos = Object.values(this.handler.damageInfos)
            .sort((a, b) => a.context.order - b.context.order)
            .slice(-20);
        const historyVectors =
            damageInfos.length > 0
                ? damageInfos.map(({ vector }) => vector)
                : [];
        const historyPadding = new Int8Array(
            historyVectorSize * (20 - historyVectors.length),
        );
        historyPadding.fill(0);

        const data = [
            new Int8Array([
                this.workerIndex,
                this.playerIndex,
                this.done,
                this.reward,
                turn,
                heuristicAction,
            ]),
            this.getMyPrivateTeam(),
            this.getMyPublicTeam(),
            this.getOppTeam(),
            ...contextVectors,
            ...historyVectors,
            historyPadding,
            legalMask,
        ];
        if (heuristicAction >= 0) {
            for (const [actionIndex, legalMaskValue] of legalMask.entries()) {
                if (actionIndex === heuristicAction && legalMaskValue === 0) {
                    console.error("bad action");
                }
            }
        }
        if (stateSize === undefined) {
            stateSize = data.reduce(
                (accumulator, currentValue) =>
                    accumulator + currentValue.length,
                0,
            );
        }
        const state = new Int8Array(stateSize);
        // state.set(new Int8Array(Int32Array.from([stateSize - 4]).buffer));
        let offset = 0;
        for (const datum of data) {
            state.set(datum, offset);
            offset += datum.length;
        }
        this.handler.getMyBattle().request = undefined;
        return state;
    }
}
