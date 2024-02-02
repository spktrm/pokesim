import { Pokemon, Side } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import {
    pseudoWeatherMapping,
    statusMapping,
    terrainMapping,
    weatherMapping,
} from "./data";
import { SideConditions } from "./types";
import { BattlesHandler, PokemonObject, historyVectorSize } from "./helpers";
import { LegalMask, State } from "./generated/state/v1/state_pb";
import { Game } from "./generated/state/v1/game_pb";
import { Team } from "./generated/state/v1/pokemon_pb";
import { Pokemon as PokemonProto } from "./generated/state/v1/pokemon_pb";
import {
    SideConditions as SideConditionsProto,
    VolatileStatuses as VolatileStatusesProto,
    Boosts as BoostsProto,
    Pseudoweathers as PseudoweatherProto,
    Weather as WeatherProto,
    Terrains as TerrainProto,
    Field as FieldProto,
} from "./generated/state/v1/context_pb";
import {
    Species as SpeciesEnum,
    Items as ItemsEnum,
    Abilities as AbilityEnum,
    Moves as MovesEnum,
    SpeciesMap,
    ItemsMap,
    AbilitiesMap,
    MovesMap,
} from "./generated/state/v1/enum_pb";
import { Context } from "./generated/state/v1/context_pb";

function formatKey(key: string | undefined): string | undefined {
    // Convert to lowercase and remove spaces and non-alphanumeric characters
    if (key === undefined) {
        return undefined;
    } else {
        return key
            .toLowerCase()
            .replace(/[\W_]+/g, "")
            .toUpperCase();
    }
}

const stateSize = new State().serializeBinary().length;

export function callMethodDynamically(
    obj: any,
    methodName: string,
    args: any[],
) {
    if (typeof obj[methodName] === "function") {
        obj[methodName].apply(obj, args);
    } else {
        console.error(`Method ${methodName} not found`);
    }
}

function capitalizeFirstLetter(string: string): string {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function appendSize(buffer: Uint8Array): Buffer {
    const newBuffer = new Uint8Array(4 + buffer.length);
    newBuffer.set(buffer, 4);
    const length = new Uint8Array(Uint32Array.from([buffer.length]).buffer);
    newBuffer.set(length, 0);
    return Buffer.from(newBuffer);
}

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

    getMyBoosts(): BoostsProto {
        const side = this.getMyPublicSide();
        return this.getBoosts(side.active);
    }
    getOppBoosts(): BoostsProto {
        const side = this.getOppSide();
        return this.getBoosts(side.active);
    }

    getBoosts(actives: (Pokemon | null)[]): BoostsProto {
        const proto = new BoostsProto();
        for (const [activeIndex, activePokemon] of actives.entries()) {
            if (activePokemon !== null) {
                for (const [boost, value] of Object.entries(
                    activePokemon.boosts,
                )) {
                    callMethodDynamically(proto, capitalizeFirstLetter(boost), [
                        value,
                    ]);
                }
            }
        }
        return proto;
    }

    getField(): FieldProto {
        const field = this.handler.getMyBattle().field;

        const pseudoweatherProto = new PseudoweatherProto();
        const weatherProto = new WeatherProto();
        const terrainProto = new TerrainProto();

        for (const [name, pseudoWeather] of Object.entries(
            field.pseudoWeather,
        )) {
            const { minDuration, maxDuration } = pseudoWeather;

            callMethodDynamically(
                pseudoweatherProto,
                `set${capitalizeFirstLetter(name)}`,
                [pseudoWeatherMapping[name]],
            );
            callMethodDynamically(
                pseudoweatherProto,
                `set${capitalizeFirstLetter(name)}MinDur`,
                [minDuration],
            );
            callMethodDynamically(
                pseudoweatherProto,
                `set${capitalizeFirstLetter(name)}MaxDur`,
                [maxDuration],
            );
        }

        weatherProto.setWeather(weatherMapping[field.weatherState.id]);
        weatherProto.setMaxDur(field.weatherState.maxDuration);
        weatherProto.setMinDur(field.weatherState.minDuration);

        terrainProto.setTerrain(terrainMapping[field.terrainState.id] ?? -1);
        terrainProto.setMaxDur(field.terrainState.minDuration);
        terrainProto.setMinDur(field.terrainState.maxDuration);

        const proto = new FieldProto();
        proto.setPseudoweather(pseudoweatherProto);
        proto.setWeather(weatherProto);
        proto.setTerrain(terrainProto);

        return proto;
    }

    getMyVolatileStatus(): VolatileStatusesProto {
        const side = this.getMyPublicSide();
        return this.getVolatileStatus(side.active);
    }
    getOppVolatileStatus(): VolatileStatusesProto {
        const side = this.getOppSide();
        return this.getVolatileStatus(side.active);
    }

    getVolatileStatus(actives: (Pokemon | null)[]): VolatileStatusesProto {
        const proto = new VolatileStatusesProto();
        for (const [activeIndex, activePokemon] of actives.entries()) {
            if (activePokemon !== null) {
                for (const volatileStatus of Object.values(
                    activePokemon.volatiles,
                )) {
                    const level = volatileStatus.level ?? 0;

                    callMethodDynamically(
                        proto,
                        `set${capitalizeFirstLetter(volatileStatus.id)}`,
                        [level],
                    );
                }
            }
        }
        return proto;
    }

    getMySideConditions(): SideConditionsProto {
        const side = this.getMyPublicSide();
        return this.getSideConditions(side.sideConditions);
    }
    getOppSideConditions(): SideConditionsProto {
        const side = this.getOppSide();
        return this.getSideConditions(side.sideConditions);
    }

    getSideConditions(sideConditions: SideConditions): SideConditionsProto {
        const proto = new SideConditionsProto();
        for (const [name, sideCondition] of Object.entries(sideConditions)) {
            callMethodDynamically(
                proto,
                `set${capitalizeFirstLetter(sideCondition.name)}`,
                [sideCondition.level],
            );
        }
        return proto;
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

    getMyPrivateTeam(): Team {
        const request = this.getMyPrivateSide();
        return this.getPrivateTeam(request);
    }

    getMyPublicTeam(): Team {
        const side = this.getMyPublicSide();
        return this.getPublicTeam(side);
    }

    getOppTeam(): Team {
        const side = this.getOppSide();
        return this.getPublicTeam(side);
    }

    getPublicTeam(side: Side): Team {
        const request = this.getMyPrivateSide();
        const isMe = request.side?.name === side.name ? true : false;
        const team = side.team;
        const actives = side.active;
        const activeIdents = [];

        const teamProto = new Team();

        for (let i = 0; i < actives.length; i++) {
            const ident = (actives[i] ?? {}).ident ?? "";
            if (ident !== undefined) {
                activeIdents.push(ident);
            }
        }

        for (let teamIndex = 0; teamIndex < 6; teamIndex++) {
            if (team[teamIndex] === undefined) {
                if (teamIndex < side.totalPokemon) {
                    if (isMe) {
                        const unk = new PokemonProto();
                        unk.setSpecies(SpeciesEnum._SPECIES_UNK_);
                        unk.setSide(true);
                        unk.setPublic(true);
                        callMethodDynamically(
                            teamProto,
                            `setPokemon${teamIndex + 1}`,
                            [unk],
                        );
                    } else {
                        const unk = new PokemonProto();
                        unk.setSpecies(SpeciesEnum._SPECIES_UNK_);
                        unk.setSide(false);
                        unk.setPublic(true);
                        callMethodDynamically(
                            teamProto,
                            `setPokemon${teamIndex + 1}`,
                            [unk],
                        );
                    }
                } else {
                    const pad = new PokemonProto();
                    pad.setSpecies(SpeciesEnum._SPECIES_PAD_);
                    callMethodDynamically(
                        teamProto,
                        `setPokemon${teamIndex + 1}`,
                        [pad],
                    );
                }
            } else {
                const ident = team[teamIndex].ident;
                const entry = this.handler.getPokemon(
                    0,
                    ident.slice(0, 2) + ident.slice(3),
                );
                callMethodDynamically(teamProto, `setPokemon${teamIndex + 1}`, [
                    Int8State.getPokemon({
                        ...entry,
                        public: true,
                        side: isMe,
                    }),
                ]);
            }
        }
        return teamProto;
    }

    static getPokemon(pokemon: PokemonObject): PokemonProto {
        const pokemonProto = new PokemonProto();

        const speciesKey = `_SPECIES_${formatKey(pokemon.name)}`;
        const itemKey = `_ITEMS_${(pokemon.item ?? "").toUpperCase()}`;
        const abilityKey = `_ABILITIES_${(
            pokemon.ability ?? ""
        ).toUpperCase()}`;

        const lastMoveKey = (
            (pokemon.lastMove === "" && !!pokemon.newlySwitched) ||
            pokemon.lastMove === "switch-in"
                ? `_moves_switch_in`
                : `_moves_${
                      (pokemon.lastMove ?? "") === ""
                          ? "PAD_"
                          : pokemon.lastMove
                  }`
        ).toUpperCase();

        pokemonProto.setSpecies(SpeciesEnum[speciesKey as keyof SpeciesMap]);
        pokemonProto.setItem(ItemsEnum[itemKey as keyof ItemsMap]);
        pokemonProto.setAbility(AbilityEnum[abilityKey as keyof AbilitiesMap]);
        pokemonProto.setHp(
            pokemon.maxhp !== undefined ? pokemon.hp / pokemon.maxhp : 1,
        );
        pokemonProto.setActive(pokemon.active);
        pokemonProto.setFainted(pokemon.fainted);
        pokemonProto.setStatus(statusMapping[pokemon.status] ?? 0);
        pokemonProto.setLastMove(MovesEnum[lastMoveKey as keyof MovesMap]);

        const moves = pokemon.moves;

        // const battle = this.handler.getMyBattle();
        for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
            // const { id, ppused } = moves[moveIndex] ?? {};
            const id = (moves[moveIndex] ?? "").toUpperCase();
            const moveKey = id === undefined ? `_MOVES_UNK_` : `_MOVES_${id}`;
            const moveToken = MovesEnum[moveKey as keyof MovesMap];
            callMethodDynamically(pokemonProto, `setMove${moveIndex + 1}`, [
                moveToken,
            ]);
            // ppLeftSetterFunc(ppused);
            // ppMaxSetterFunc(
            //     battle.gen.dex.moves.get(moves[moveIndex]).pp * (8 / 5)
            // );
        }

        pokemonProto.setPublic(pokemon.public);
        pokemonProto.setSide(pokemon.side);
        pokemonProto.setSleepTurns(pokemon.sleep_turns);
        pokemonProto.setToxicTurns(pokemon.toxic_turns);

        return pokemonProto;
    }

    getPrivateTeam(request: AnyObject): Team {
        const requestSide = (request?.side ?? { pokemon: [] }).pokemon;

        const privateId =
            (this.handler.getMyBattle().request?.side ?? {})?.id ?? "";
        const privateIndex = parseInt(privateId.slice(1)) - 1;

        const team = new Team();

        for (let teamIndex = 0; teamIndex < 6; teamIndex++) {
            const entity = requestSide[teamIndex];
            if (entity === undefined) {
                continue;
            }
            const ident = (entity ?? {}).ident ?? "";
            const entry = this.handler.getPokemon(privateIndex, ident);
            callMethodDynamically(team, `setPokemon${teamIndex + 1}`, [
                Int8State.getPokemon({ ...entry, public: false, side: true }),
            ]);
        }
        return team;
    }

    getLegalMask(): LegalMask {
        const request = this.handler.getMyBattle().request as AnyObject;
        const mask = new LegalMask();
        if (request === undefined || this.done) {
            mask.setMove1(true);
            mask.setMove2(true);
            mask.setMove3(true);
            mask.setMove4(true);
            mask.setSwitch1(true);
            mask.setSwitch2(true);
            mask.setSwitch3(true);
            mask.setSwitch4(true);
            mask.setSwitch5(true);
            mask.setSwitch6(true);
        } else {
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
                        callMethodDynamically(mask, `setMove${j}`, [true]);
                    }
                }
            } else if (request.active) {
                const pokemon = request.side.pokemon;
                const active = request.active[0];
                const possibleMoves = active.moves ?? [];
                const canSwitch = [];

                for (let funcIndex = 0; funcIndex < 4; funcIndex++) {
                    const currentMove = possibleMoves[funcIndex];
                    if (!currentMove.disabled) {
                        callMethodDynamically(mask, `setMove${funcIndex + 1}`, [
                            true,
                        ]);
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
                    callMethodDynamically(mask, `setSwitch${switchIndex + 1}`, [
                        true,
                    ]);
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
                    callMethodDynamically(mask, `setSwitch${switchIndex + 1}`, [
                        true,
                    ]);
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
    //         const moveArray = new Int8Array(
    //             new Int16Array([
    //                 isCritical ? 1 : 0,
    //                 effectiveness,
    //                 missed ? 1 : 0,
    //                 targetFainted ? 1 : 0,
    //                 Math.floor(2047 * (1 + Math.max(-1, Math.min(1, damage)))),
    //                 moveCounter,
    //                 switchCounter,
    //                 moveName === "switch-in"
    //                     ? switchToken
    //                     : moveMapping[formatKey(moveName) ?? ""],
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

    getContextVector(): Context {
        const context = new Context();
        context.setSideConditions1(this.getMySideConditions());
        context.setSideConditions2(this.getOppSideConditions());
        context.setVolatileStatus1(this.getMyVolatileStatus());
        context.setVolatileStatus1(this.getOppVolatileStatus());
        context.setBoosts1(this.getMyBoosts());
        context.setBoosts2(this.getOppBoosts());
        context.setField(this.getField());
        return context;
    }

    getState(): Buffer {
        const turn = this.handler.getBattleTurn();
        this.handler.getAggregatedTurnLines();

        const heuristicAction = this.done
            ? -1
            : this.handler.getHeuristicActionIndex(
                  this.playerIndex,
                  this.workerIndex,
              );
        const legalMask = this.getLegalMask();

        const contextProto = this.getContextVector();

        // this.updatetHistoryVectors();`

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
        // historyPadding.fill(-1);

        const game = new Game();
        game.setWorkerIndex(this.workerIndex);
        game.setPlayerIndex(this.playerIndex === 0 ? false : true);
        game.setDone(this.done === 0 ? false : true);
        game.setReward(this.reward);
        game.setTurn(turn);
        game.setHeuristicAction(heuristicAction);

        const state = new State();
        state.setGame(game);
        state.setPrivateTeam(this.getMyPrivateTeam());
        state.setPublicTeam1(this.getMyPublicTeam());
        state.setPublicTeam2(this.getOppTeam());
        state.setContext(contextProto);
        state.setLegalMask(legalMask);

        if (heuristicAction >= 0) {
            for (const [actionIndex, getterFunc] of [
                legalMask.getMove1,
                legalMask.getMove2,
                legalMask.getMove3,
                legalMask.getMove4,
                legalMask.getSwitch1,
                legalMask.getSwitch2,
                legalMask.getSwitch3,
                legalMask.getSwitch4,
                legalMask.getSwitch5,
                legalMask.getSwitch6,
            ].entries()) {
                const legalMaskValue = getterFunc();
                if (
                    actionIndex === heuristicAction &&
                    legalMaskValue === false
                ) {
                    console.error("bad action");
                }
            }
        }

        this.handler.getMyBattle().request = undefined;

        return appendSize(state.serializeBinary());
    }
}
