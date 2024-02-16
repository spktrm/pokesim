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
import {
    BattlesHandler,
    historyVectorSize,
    paddedPokemonArray,
} from "./helpers";

let stateSize: number | undefined = undefined;

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
        return this.getPublicTeam(side, true);
    }

    getOppTeam(): Int8Array {
        const side = this.getOppSide();
        return this.getPublicTeam(side, false);
    }

    getPublicTeam(side: Side, is_me: boolean): Int8Array {
        const team = side.team;
        const teamArray = new Int8Array(paddedPokemonArray.length * 6);

        for (let i = 0; i < side.totalPokemon; i++) {
            const ident = team[i]?.ident;
            const vectorPokemon = this.handler.getVectorPokemon({
                ident,
                is_me,
                is_public: true,
            });
            const vector = vectorPokemon.getVector();
            if (vector.length !== paddedPokemonArray.length) {
                throw new Error(JSON.stringify(vectorPokemon));
            }
            teamArray.set(vector, i * paddedPokemonArray.length);
        }
        for (let i = side.totalPokemon; i < 6; i++) {
            teamArray.set(paddedPokemonArray, i * paddedPokemonArray.length);
        }
        return teamArray;
    }

    getPrivateTeam(request: AnyObject): Int8Array {
        const requestSide = (request?.side ?? { pokemon: [] }).pokemon;
        const teamArray = new Int8Array(paddedPokemonArray.length * 6);

        for (let i = 0; i < 6; i++) {
            const ident = requestSide[i]?.ident ?? "";
            const vectorPokemon = this.handler.getVectorPokemon({
                ident,
                is_me: true,
                is_public: false,
            });
            const vector = vectorPokemon.getVector();
            if (vector.length !== paddedPokemonArray.length) {
                throw new Error(JSON.stringify(vectorPokemon));
            }
            teamArray.set(vector, i * paddedPokemonArray.length);
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

    // updatetHistoryVectors(): void {
    //     // let offset = 0;

    //     for (const [key, moveStore] of Object.entries(
    //         this.handler.damageInfos
    //     )) {
    //         if (moveStore.hasVector) {
    //             continue;
    //         }
    //         const {
    //             // user,
    //             // target,
    //             // userSide,
    //             // targetSide,
    //             order,
    //             isCritical,
    //             damage,
    //             effectiveness,
    //             missed,
    //             moveName,
    //             targetFainted,
    //             moveCounter,
    //             switchCounter,
    //         } = moveStore.context;
    //         // const isMe = this.playerIndex === userSide ? 1 : 0;
    //         // const userProps = this.handler.getPokemon(userSide, user);
    //         // const targetProps = this.handler.getPokemon(targetSide, target);

    //         const action =
    //             moveName === "switch-in"
    //                 ? "<SWITCH>"
    //                 : formatKey(moveName ?? "") ?? "";

    //         const moveArray = new Int8Array(
    //             new Int16Array([
    //                 1,
    //                 isCritical ? 1 : 0,
    //                 effectiveness,
    //                 missed ? 1 : 0,
    //                 targetFainted ? 1 : 0,
    //                 Math.floor(2047 * (1 + Math.max(-1, Math.min(1, damage)))),
    //                 moveCounter,
    //                 switchCounter,
    //                 moveMapping[action],
    //                 order,
    //             ]).buffer
    //         );
    //         // const history: Array<Int8Array> = [
    //         //     contextVector,
    //         //     userSide === this.playerIndex
    //         //         ? getPublicPokemon(userProps, true, isMe)
    //         //         : getPrivatePokemon(userProps),
    //         //     targetSide === this.playerIndex
    //         //         ? getPublicPokemon(targetProps, true, isMe)
    //         //         : getPrivatePokemon(targetProps),
    //         //     moveArray,
    //         // ];
    //         // offset = 0;
    //         // for (const datum of history) {
    //         //     this.handler.damageInfos[key].vector.set(datum, offset);
    //         //     offset += datum.length;
    //         // }
    //         const offset = historyVectorSize - moveArray.length;
    //         this.handler.damageInfos[key].vector.set(moveArray, offset);
    //         moveStore.hasVector = true;
    //     }
    // }

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
        // this.handler.getAggregatedTurnLines();

        const heuristicAction = this.done
            ? -1
            : this.handler.getHeuristicActionIndex(
                  this.playerIndex,
                  this.workerIndex,
              );
        const legalMask = this.getLegalMask();

        const contextVectors = this.getContextVectors();

        // this.updatetHistoryVectors();

        // const damageInfos = Object.values(this.handler.damageInfos)
        //     .sort((a, b) => a.context.order - b.context.order)
        //     .slice(-20);
        // const historyVectors =
        //     damageInfos.length > 0
        //         ? damageInfos.map(({ vector }) => vector)
        //         : [];
        // const historyPadding = new Int8Array(
        //     historyVectorSize * (20 - historyVectors.length)
        // );
        // historyPadding.fill(0);

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
            // ...historyVectors,
            // historyPadding,
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
        let offset = 0;
        for (const datum of data) {
            state.set(datum, offset);
            offset += datum.length;
        }
        this.handler.getMyBattle().request = undefined;
        return state;
    }
}
