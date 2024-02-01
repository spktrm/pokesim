import { Battle, Pokemon, Side } from "@pkmn/client";
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
import { BattlesHandler, historyVectorSize } from "./helpers";

let stateSize: number | undefined = undefined;

function formatKey(key: string | undefined): string | undefined {
    // Convert to lowercase and remove spaces and non-alphanumeric characters
    if (key === undefined) {
        return undefined;
    } else {
        return key.toLowerCase().replace(/[\W_]+/g, "");
    }
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

const paddingToken = -1;
const unknownToken = -2;
const switchToken = -3;

function getMappingValueWrapper(
    pokemon: AnyObject,
    mapping: AnyObject,
    key: string | undefined
): number {
    if (key === "") {
        return unknownToken;
    } else if (key === undefined) {
        return paddingToken;
    }
    const mappedValue = getMappingValue(pokemon, mapping, key);
    if (mappedValue === undefined) {
        logKeyError(key, mapping);
        return paddingToken;
    } else {
        return mappedValue;
    }
}

export function getPublicPokemon(
    battle: Battle | undefined,
    pokemon: AnyObject,
    active: boolean,
    sideToken: number,
    buckets: number = 1024
): Int8Array {
    let moveTokens = [];
    let movePpLeft = [];
    let movePpMax = Array(4);
    movePpMax.fill(0);

    const moveSlots = pokemon.moveSlots ?? [];
    const moves = pokemon.moves ?? [];

    for (let i = 0; i < 4; i++) {
        moveTokens.push(
            getMappingValueWrapper(pokemon, moveMapping, moves[i] ?? "")
        );
        const ppUsed = (moveSlots[i] ?? {})?.ppUsed ?? 0;
        movePpLeft.push(ppUsed);
        if (battle !== undefined) {
            movePpMax[i] = battle.gen.dex.moves.get(moves[i]).pp * (8 / 5);
        }
    }
    const lastMoveToken =
        (pokemon.lastMove === "" && !!pokemon.newlySwitched) ||
        pokemon.lastMove === "switch-in"
            ? switchToken
            : getMappingValueWrapper(pokemon, moveMapping, pokemon.lastMove);

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
        statusMapping[pokemon.status] ?? paddingToken,
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
    buckets: number = 1024
): Int8Array {
    let moveTokens = [];
    let movePpLeft = [];
    let movePpMax = Array(4);
    movePpMax.fill(0);

    const moveSlots = pokemon.moveSlots ?? [];
    const moves = pokemon.moves ?? [];

    for (let i = 0; i < 4; i++) {
        moveTokens.push(
            getMappingValueWrapper(pokemon, moveMapping, moves[i] ?? "")
        );
        const ppUsed = (moveSlots[i] ?? {})?.ppUsed ?? 0;
        movePpLeft.push(ppUsed);
        if (battle !== undefined) {
            movePpMax[i] = battle.gen.dex.moves.get(moves[i]).pp * (8 / 5);
        }
    }

    const lastMoveToken =
        (pokemon.lastMove === "" && !!pokemon.newlySwitched) ||
        pokemon.lastMove === "switch-in"
            ? switchToken
            : getMappingValueWrapper(pokemon, moveMapping, pokemon.lastMove);
    const formatedPokemonName = formatKey(pokemon.name);
    const speciesToken = getMappingValueWrapper(
        pokemon,
        pokemonMapping,
        formatedPokemonName
    );
    const itemToken = getMappingValueWrapper(
        pokemon,
        itemMapping,
        pokemon.item === "" ? undefined : pokemon.item
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
        statusMapping[pokemon.status] ?? paddingToken,
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

const paddedPokemonObj = { moves: [] };
const paddedPokemonArray = getPublicPokemon(
    undefined,
    paddedPokemonObj,
    false,
    1
);

const unknownPokemonObj = { name: "", moves: [] };
const unknownPokemonArray0 = getPublicPokemon(
    undefined,
    unknownPokemonObj,
    false,
    0
);
const unknownPokemonArray1 = getPublicPokemon(
    undefined,
    unknownPokemonObj,
    false,
    1
);

const boostsEntries = Object.entries(boostsMapping);

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
                            i * paddedPokemonArray.length
                        );
                    } else {
                        teamArray.set(
                            unknownPokemonArray0,
                            i * paddedPokemonArray.length
                        );
                    }
                } else {
                    teamArray.set(
                        paddedPokemonArray,
                        i * paddedPokemonArray.length
                    );
                }
            } else {
                const ident = team[i].ident;
                teamArray.set(
                    getPublicPokemon(
                        this.handler.getMyBattle(),
                        team[i],
                        activeIdents.includes(ident),
                        isMe
                    ),
                    i * paddedPokemonArray.length
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
                ...(entity === undefined
                    ? {}
                    : this.handler.getPokemon(privateIndex, ident)),
            };
            teamArray.set(
                getPrivatePokemon(this.handler.getMyBattle(), amalgam),
                i * paddedPokemonArray.length
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

    updatetHistoryVectors(contextVector: Int8Array): void {
        // let offset = 0;

        for (const [key, moveStore] of Object.entries(
            this.handler.damageInfos
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
            const moveArray = new Int8Array(
                new Int16Array([
                    isCritical ? 1 : 0,
                    effectiveness,
                    missed ? 1 : 0,
                    targetFainted ? 1 : 0,
                    Math.floor(2047 * (1 + Math.max(-1, Math.min(1, damage)))),
                    moveCounter,
                    switchCounter,
                    moveName === "switch-in"
                        ? switchToken
                        : moveMapping[formatKey(moveName) ?? ""],
                    order,
                ]).buffer
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

    getContextVector(): Int8Array {
        const context = [
            this.getMySideConditions(),
            this.getOppSideConditions(),
            this.getMyVolatileStatus(),
            this.getOppVolatileStatus(),
            this.getMyBoosts(),
            this.getOppBoosts(),
            this.getField(),
        ];
        const contextVector = new Int8Array(99);
        let contextOffset = 0;
        for (const datum of context) {
            contextVector.set(datum, contextOffset);
            contextOffset += datum.length;
        }
        return contextVector;
    }

    getState(): Int8Array {
        const turn = this.handler.getBattleTurn();
        this.handler.getAggregatedTurnLines();

        const heuristicAction = this.done
            ? -1
            : this.handler.getHeuristicActionIndex(
                  this.playerIndex,
                  this.workerIndex
              );
        const legalMask = this.getLegalMask();

        const contextVector = this.getContextVector();

        this.updatetHistoryVectors(contextVector);

        const damageInfos = Object.values(this.handler.damageInfos)
            .sort((a, b) => a.context.order - b.context.order)
            .slice(-20);
        const historyVectors =
            damageInfos.length > 0
                ? damageInfos.map(({ vector }) => vector)
                : [];
        const historyPadding = new Int8Array(
            historyVectorSize * (20 - historyVectors.length)
        );
        historyPadding.fill(-1);

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
            contextVector,
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
