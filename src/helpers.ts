import {
    Pokemon as ClientPokemon,
    Side,
    Battle as ClientBattle,
} from "@pkmn/client";
import { Protocol } from "@pkmn/protocol";
import { AnyObject, Pokemon as ServerPokemon } from "@pkmn/sim";
import { Int8State } from "./state";
import {
    arange,
    getRandomAction,
    numArange,
    weightedRandomSample,
} from "./random";
import { BoostID } from "@pkmn/data";
import { createHash } from "node:crypto";
import {
    abilityMapping,
    boostsMapping,
    contextVectorSize,
    itemMapping,
    maxPP,
    moveMapping,
    pokemonMapping,
    sideConditionsMapping,
    statusMapping,
    typeMapping,
    volatileStatusMapping,
} from "./data";

const zeroAsciiCode = "0".charCodeAt(0);
const d_AsciiCode = "d".charCodeAt(0);

const EffectivenessLookup = {
    "0": 1,
    "1": 2,
    "2": 0.5,
    "3": 0,
};

function typeStringToFloat(typeString: string): number {
    return EffectivenessLookup[typeString as keyof typeof EffectivenessLookup];
}

const EntryHazards = ["spikes", "stealhrock", "stickyweb", "toxicspikes"];
const AntiHazardMoves = ["rapidspin", "defog"];
const SpeedTierCoefficient = 0.1;
const HPFractionCoefficient = 0.4;
const SwitchOutMatchupThreshold = -2;

function getProperMoveName(moveName: string): string {
    if (moveName.startsWith("return")) {
        return "return";
    }
    return moveName;
}

function processMulthitValue(
    multhitValue: number | number[] | undefined
): number {
    if (multhitValue === undefined) {
        return 1;
    } else if (typeof multhitValue === "number") {
        return multhitValue;
    } else {
        return multhitValue.reduce((a, b) => a + b) / multhitValue.length;
    }
}

function indexOfMax(arr: number[]): number {
    if (arr.length === 0) {
        return -1; // Return -1 if the array is empty
    }

    let max = arr[0];
    let maxIndex = 0;

    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

class SimpleHeuristicPlayer {
    handler: BattlesHandler;
    playerIndex: number;

    constructor(handler: BattlesHandler, playerIndex: number) {
        this.handler = handler;
        this.playerIndex = playerIndex;
    }

    _getType(battle: ClientBattle, atkType: string) {
        const type = battle.gen.types.get(atkType);
        return type;
    }

    _getSpecies(battle: ClientBattle, name: string): any {
        return battle.gen.species.get(name);
    }

    _getMove(battle: ClientBattle, name: string): any {
        return battle.gen.moves.get(getProperMoveName(name));
    }

    _getMoveScore(
        battle: ClientBattle,
        active: ClientPokemon,
        opponent: ClientPokemon,
        name: string,
        physicalRatio: number,
        specialRatio: number
    ): number {
        const moveData = this._getMove(battle, name);
        const speciesData = this._getSpecies(battle, active.name);
        const STAB = speciesData.types.includes(moveData.type) ? 1.5 : 1;
        const attackRatio =
            moveData.category === "Physical" ? physicalRatio : specialRatio;
        const accuracy = moveData.accuracy === 1 ? 100 : moveData.accuracy;
        const numHits = processMulthitValue(moveData.multihit);
        const damageMultiplier = this._damageMultiplier(
            battle,
            opponent,
            moveData.type
        );
        return (
            moveData.basePower *
            STAB *
            attackRatio *
            (accuracy / 100) ** numHits *
            numHits *
            damageMultiplier
        );
    }

    _damageMultiplier(
        battle: ClientBattle,
        entity: ClientPokemon,
        type: string
    ): number {
        const multipliers = entity.types.map((entityType: string) => {
            const typeData = this._getType(battle, entityType) as any;
            const typeValue = typeData.damageTaken[type] ?? 0;
            return typeStringToFloat(typeValue.toString());
        });
        return multipliers.reduce((a: number, b: number) => a * b);
    }

    _estimateMatchup(
        mon: ClientPokemon | null,
        opponent: ClientPokemon | null
    ): number {
        if (mon === null) {
            return -1;
        }
        if (opponent === null) {
            return 1;
        }

        const battle = this.handler.getMyBattle();
        const monSpecies = this._getSpecies(battle, mon.name);
        const opponentSpecies = this._getSpecies(battle, opponent.name);
        let score = Math.max(
            ...opponentSpecies.types.map((opponentType: string) =>
                this._damageMultiplier(battle, monSpecies, opponentType)
            )
        );
        score -= Math.max(
            ...monSpecies.types.map((monType: string) =>
                this._damageMultiplier(battle, opponentSpecies, monType)
            )
        );

        if (monSpecies.baseStats.spe > opponentSpecies.baseStats.spe) {
            score += SpeedTierCoefficient;
        } else {
            score -= SpeedTierCoefficient;
        }

        score += (mon.hp / mon.maxhp) * HPFractionCoefficient;
        score -= (opponent.hp / opponent.maxhp) * HPFractionCoefficient;

        return score;
    }

    _statEstimation(
        mon: ClientPokemon,
        stat: "atk" | "def" | "spa" | "spd" | "spe"
    ) {
        let boost: number;

        const statValue = mon.boosts[stat] ?? 0;
        const baseStats = mon.species.baseStats;

        if (statValue > 1) {
            boost = (2 + statValue) / 2;
        } else {
            boost = 2 / (2 - statValue);
        }
        return (2 * baseStats[stat] + 31 + 5) * boost;
    }

    _shouldSwitchOut(
        switches: ClientPokemon[],
        active: ClientPokemon | null,
        opponent: ClientPokemon | null
    ): boolean {
        if (opponent === null) {
            return false;
        }
        if (active === null) {
            return true;
        }

        const isBadMatchup = switches.filter(
            (potenialSwitch) =>
                this._estimateMatchup(potenialSwitch, opponent) > 0
        );

        const boosts: AnyObject = active.boosts;

        if (isBadMatchup) {
            if ((boosts.def ?? 0 <= -3) || (boosts.spd ?? 0 <= -3)) {
                return true;
            }
            if (
                (boosts.atk ?? 0) <= -3 &&
                (boosts.atk ?? 0) >= (boosts.spa ?? 0)
            ) {
                return true;
            }
            if (
                (boosts.spa ?? 0) <= -3 &&
                (boosts.atk ?? 0) <= (boosts.spa ?? 0)
            ) {
                return true;
            }
            if (
                this._estimateMatchup(active, opponent) <
                SwitchOutMatchupThreshold
            ) {
                return true;
            }
        }
        return false;
    }

    getSwitchIndex(
        switches: ClientPokemon[],
        opponent: ClientPokemon | null
    ): number {
        const switchScores = switches.map((switchValue) =>
            switchValue.fainted ?? false
                ? -10000
                : this._estimateMatchup(switchValue, opponent)
        );
        const switchArgmax = indexOfMax(switchScores);
        return 4 + 1 + switchArgmax;
    }

    getActionIndex(): number {
        const battle = this.handler.getMyBattle();
        const stateHandler = new Int8State({
            handler: this.handler,
            playerIndex: this.playerIndex,
            workerIndex: 0,
            done: 0,
            reward: 0,
        });
        const legalMask = stateHandler.getLegalMask();
        const availableMoves = legalMask.slice(0, 4).reduce((a, b) => a + b);
        const availableSwitches = legalMask.slice(4).reduce((a, b) => a + b);

        const request = (battle.request ?? {}) as AnyObject;
        const requestSide = request.side ?? { pokemon: [] };
        const sideIndex = this.playerIndex;
        const active = battle.sides[this.playerIndex].active[0];

        const switches: any[] = requestSide.pokemon
            .filter((x: AnyObject) => !x.active)
            .map((x: AnyObject) => x.ident)
            .map((ident: string) => this.handler.getPokemon(ident));

        const mySide = battle.sides[this.playerIndex];
        const oppSide = battle.sides[1 - this.playerIndex];
        const opponent = oppSide.active[0];

        if (active === null || (request.forceSwitch ?? false)) {
            return this.getSwitchIndex(switches, opponent);
        }
        const requestActive = request.active[0];

        const privateActive = this.handler.getPokemon(active.originalIdent);

        const calcRatio = active !== null && opponent !== null;

        const physicalRatio = calcRatio
            ? this._statEstimation(active, "atk") /
              this._statEstimation(opponent, "def")
            : 0;
        const specialRatio = calcRatio
            ? this._statEstimation(active, "spa") /
              this._statEstimation(opponent, "spd")
            : 0;

        if (
            availableMoves &&
            (!this._shouldSwitchOut(switches, active, opponent) ||
                !availableSwitches)
        ) {
            const nRemainingMons =
                mySide.totalPokemon -
                mySide.team.reduce((a, x) => a + (x.fainted ? 1 : 0), 0);
            const nOppRemainingMons =
                oppSide.totalPokemon -
                oppSide.team.reduce((a, x) => a + (x.fainted ? 1 : 0), 0);

            if (active !== null) {
                const oppSideConditionKeys = Object.keys(
                    oppSide.sideConditions
                );
                for (const [moveIndex, move] of requestActive.moves.entries()) {
                    const isDisabled = move.disabled ?? false;
                    if (isDisabled) {
                        continue;
                    }
                    if (
                        // entry hazards
                        nOppRemainingMons >= 3 &&
                        EntryHazards.includes(move.id) &&
                        !oppSideConditionKeys.includes(move.id)
                    ) {
                        return moveIndex;
                    } else if (
                        // hazard removal
                        oppSideConditionKeys.length > 0 &&
                        AntiHazardMoves.includes(move.id) &&
                        nRemainingMons >= 2
                    ) {
                        return moveIndex;
                    }
                }

                // setup moves
                const hpRatio = privateActive.hp / privateActive.max_hp;
                const opponentMatchup = this._estimateMatchup(active, opponent);
                if (hpRatio >= 0.99 && opponentMatchup > 0) {
                    for (const [
                        moveIndex,
                        move,
                    ] of requestActive.moves.entries()) {
                        const isDisabled = move.disabled ?? false;
                        if (isDisabled) {
                            continue;
                        }
                        const moveData = this._getMove(
                            battle,
                            getProperMoveName(move.id)
                        );
                        if (moveData.boosts !== undefined) {
                            const boostValues = Object.values(
                                moveData.boosts
                            ) as Array<number>;
                            const boostSum = boostValues.reduce(
                                (a, b) => a + b
                            );
                            const moveTarget = moveData.target;
                            const currentBoosts = Object.entries(
                                moveData.boosts
                            )
                                .filter(([_, value]) => (value as number) > 0)
                                .map(
                                    ([key, _]) =>
                                        active.boosts[key as BoostID] ?? 0
                                );

                            if (
                                boostSum >= 2 &&
                                moveTarget === "self" &&
                                Math.min(...currentBoosts) < 6
                            ) {
                                return moveIndex;
                            }
                        }
                    }
                }

                if (opponent !== null) {
                    const moveScores = requestActive.moves.map(
                        (move: { id: string; [k: string]: any }) =>
                            move.disabled ?? false
                                ? -10000
                                : this._getMoveScore(
                                      battle,
                                      active,
                                      opponent,
                                      move.id,
                                      physicalRatio,
                                      specialRatio
                                  )
                    );
                    const moveArgmax = indexOfMax(moveScores);
                    return moveArgmax;
                }
            }
        }

        if (availableSwitches) {
            return this.getSwitchIndex(switches, opponent);
        }

        const [randIndex] = weightedRandomSample(
            numArange,
            new Array(...legalMask),
            1
        );
        return randIndex;
    }
}

class MaxdmgPlayer {
    handler: BattlesHandler;
    playerIndex: number;
    legalMask: Int8Array;

    constructor(
        handler: BattlesHandler,
        playerIndex: number,
        legalMask: Int8Array
    ) {
        this.handler = handler;
        this.playerIndex = playerIndex;
        this.legalMask = legalMask;
    }

    getAction(): string {
        const battle = this.handler.getMyBattle();
        const request = battle.request as AnyObject;
        const active = (request?.active ?? [])[0] as AnyObject;
        const canMove = this.legalMask.slice(0, 4).reduce((a, b) => a + b) > 0;

        let maxMoveIndex: number = -1;
        if (active !== undefined && canMove) {
            const moves: string[] = active.moves.map(
                (value: { id: any }) => value.id
            );
            const basePowers = moves.map((moveId, moveIndex) => {
                const valid = this.legalMask[moveIndex];
                const returnLength = "return".length;
                if (
                    moveId.startsWith("return") &&
                    moveId.length !== returnLength
                ) {
                    return (
                        valid * parseInt(moveId.slice(returnLength)) +
                        (1 - valid) * -1
                    );
                }
                return (
                    valid * battle.gen.dex.moves.get(moveId).basePower +
                    (1 - valid) * -1
                );
            });
            maxMoveIndex = basePowers.reduce(
                (maxIndex, currentValue, currentIndex, array) =>
                    currentValue > array[maxIndex] ? currentIndex : maxIndex,
                0
            );
        }

        return maxMoveIndex === -1 ? "default" : `move ${maxMoveIndex + 1}`;
    }
}

class moveInfo {
    order: number = 0;
    user: string = "";
    target: string = "";
    userSide: number = 0;
    targetSide: number = 0;
    moveName: string | undefined;
    damage: number = 0;
    effectiveness: number = 1;
    isCritical: boolean = false;
    targetFainted: boolean = false;
    missed: boolean = false;
    moveCounter: number = 0;
    switchCounter: number = 0;

    isReady() {
        return this.moveName !== undefined;
    }

    hash() {
        let salt = "";
        for (const addition of [
            this.order,
            this.user,
            this.target,
            this.userSide,
            this.targetSide,
            this.moveName,
            this.damage,
            this.effectiveness,
            this.isCritical,
            this.targetFainted,
            this.missed,
            this.moveCounter,
            this.switchCounter,
        ]) {
            salt += `|${addition}`;
        }
        return createHash("sha256").update(salt).digest("hex");
    }
}

export const historyVectorSize = contextVectorSize + 2 * 48 + 20;

export const padString = "<PAD>";
export const unkString = "<UNK>";
export const nullString = "<NULL>";
export const noneString = "<NONE>";

function formatKey(key: string): string {
    return key.startsWith("<") ? key : key.toLowerCase().replace(/[\W_]+/g, "");
}

export class VectorPokemon {
    species: string;
    item: string;
    ability: string;
    hp: number;
    max_hp: number;
    active: number;
    fainted: number;
    status: string;
    last_move: string;
    is_public: number;
    side: number;
    sleep_turns: number;
    toxic_turns: number;
    types: string[];
    moves: string[];
    pp_left: number[];
    pp_max: number[];

    constructor() {
        this.species = unkString;
        this.item = unkString;
        this.ability = unkString;
        this.hp = 100;
        this.max_hp = 100;
        this.active = 0;
        this.fainted = 0;
        this.status = nullString;
        this.last_move = noneString;
        this.is_public = 0;
        this.side = 1;
        this.sleep_turns = 0;
        this.toxic_turns = 0;
        this.moves = [unkString, unkString, unkString, unkString];
        this.pp_left = [0, 0, 0, 0];
        this.pp_max = [0, 0, 0, 0];
        this.types = [unkString, unkString];
    }

    getHpToken() {
        return Math.floor((1023 * this.hp) / this.max_hp);
    }

    updatePublicEntity(pokemon: ClientPokemon) {
        this.species = formatKey(pokemon.name);
        this.item = pokemon.item === "" ? unkString : pokemon.item;
        this.ability = pokemon.ability === "" ? unkString : pokemon.ability;
        this.hp = pokemon.hp;
        this.max_hp = pokemon.maxhp;
        this.fainted = pokemon.hp === 0 ? 1 : 0;

        for (const [typeIndex, type] of pokemon.types.entries()) {
            this.types[typeIndex] = formatKey(type);
        }
        if (pokemon.types.length <= 1) {
            this.types[1] = padString;
        }

        const { sleepTurns, toxicTurns } = pokemon.statusState;
        this.sleep_turns = sleepTurns;
        this.toxic_turns = toxicTurns;

        for (const [moveIndex, move] of pokemon.moveSlots.slice(-4).entries()) {
            const { id, ppUsed } = move;
            const formattedMove = formatKey(id);
            this.moves[moveIndex] = formattedMove;
            this.pp_left[moveIndex] = ppUsed;
            this.pp_max[moveIndex] = maxPP[formattedMove];
        }

        let last_move_string: string = "";
        if (
            (pokemon.lastMove === "" && !!pokemon.newlySwitched) ||
            pokemon.lastMove === "switch-in"
        ) {
            last_move_string = "<SWITCH>";
        } else {
            if (pokemon.lastMove === "" || pokemon.lastMove === undefined) {
                last_move_string = "<NONE>";
            } else {
                last_move_string = formatKey(pokemon.lastMove);
            }
        }
        this.last_move = last_move_string;

        return;
    }

    updatePrivateEntity(pokemon: Protocol.Request.Pokemon) {
        this.species = formatKey(pokemon.name);
        this.item = pokemon.item === "" ? nullString : pokemon.item;
        this.ability = pokemon.ability;
        this.hp = pokemon.hp;
        this.max_hp = pokemon.maxhp;
        this.active = pokemon.active ? 1 : 0;
        this.fainted = pokemon.hp === 0 ? 1 : 0;

        const ppMapping: { [k: string]: number[] } = {};
        for (const [moveIndex, move] of this.moves.entries()) {
            ppMapping[move] = [
                this.pp_left[moveIndex] ?? 0,
                this.pp_max[moveIndex] ?? 0,
            ];
        }
        this.moves = pokemon.moves;
        for (let i = this.moves.length; i < 4; i++) {
            this.moves.push("<NONE>");
        }

        for (const [moveIndex, move] of pokemon.moves.entries()) {
            const formattedMove = formatKey(move);
            const [ppUsed, ppMax] = ppMapping[formattedMove] ?? [];
            this.pp_left[moveIndex] = ppUsed ?? 0;
            this.pp_max[moveIndex] = ppMax ?? maxPP[formattedMove] ?? 0;
        }

        return;
    }

    getVector(): Int8Array {
        const moveTokens: number[] = [];
        for (let i = 0; i < 4; i++) {
            moveTokens.push(moveMapping[this.moves[i]]);
        }
        const arr = [
            pokemonMapping[this.species],
            itemMapping[this.item],
            abilityMapping[this.ability],
            this.getHpToken(),
            this.active,
            this.fainted,
            statusMapping[this.status],
            moveMapping[this.last_move],
            this.is_public,
            this.side,
            this.sleep_turns,
            this.toxic_turns,
            typeMapping[this.types[0]],
            typeMapping[this.types[1]],
            ...this.pp_left,
            ...this.pp_max,
            ...moveTokens,
        ];
        for (const token of arr) {
            if (token === undefined) {
                throw new Error(JSON.stringify(this));
            }
        }
        return new Int8Array(new Int16Array(arr).buffer);
    }
}

const paddedPokemonObj = new VectorPokemon();
paddedPokemonObj.species = padString;
export const paddedPokemonArray = paddedPokemonObj.getVector();

const unknownPokemonObj = new VectorPokemon();

unknownPokemonObj.side = 0;
export const unknownPokemonArray0 = unknownPokemonObj.getVector();

unknownPokemonObj.side = 1;
export const unknownPokemonArray1 = unknownPokemonObj.getVector();

export class BattlesHandler {
    playerIndex: number;
    battles: ClientBattle[];
    turn: number;
    turns: AnyObject;
    damage: AnyObject;
    entityVectors: AnyObject;
    damageInfos: {
        [k: string]: {
            context: moveInfo;
            vector: Int8Array;
            hasVector: boolean;
        };
    };

    constructor(playerIndex: number, battles: ClientBattle[]) {
        this.playerIndex = playerIndex;
        this.battles = battles;
        this.turn = 0;
        this.turns = {};
        this.damage = {};
        this.entityVectors = {};
        this.damageInfos = {};
    }

    reset() {
        this.turn = 0;
        this.turns = {};
        this.damage = {};
        this.entityVectors = {};
        this.damageInfos = {};
    }

    getNumFainted(playerIndex: number): number {
        const side = this.battles[0].sides[playerIndex];
        let numFainted = 0;
        for (let sidePokemon of side.team) {
            if (sidePokemon.fainted) {
                numFainted += 1;
            }
        }
        return numFainted;
    }

    getPokemon(ident: string): AnyObject {
        if (ident.startsWith("p1a") || ident.startsWith("p2a")) {
            ident = ident.slice(0, 2) + ident.slice(3);
        }
        const battle = this.getMyBattle();
        const request = battle.request;
        const requestSidePokemon = request?.side?.pokemon ?? [];
        const privatePokemon =
            requestSidePokemon.filter((x) => x?.ident === ident)[0] ?? {};
        for (const side of battle.sides) {
            const team = side.team;
            for (const member of team) {
                if (member.originalIdent === ident) {
                    return {
                        ...member,
                        ...privatePokemon,
                    };
                }
            }
        }
        return {
            ...privatePokemon,
        };
    }

    getVectorPokemon(args: {
        ident: string | undefined;
        is_me: boolean;
        is_public: boolean;
    }): VectorPokemon {
        const { ident, is_me, is_public } = args;
        let _ident = ident;

        const placeholder = new VectorPokemon();
        const battle = this.getMyBattle();
        const request = battle.request;
        placeholder.side = is_me ? 1 : 0;

        if (_ident !== undefined) {
            if (_ident.startsWith("p1a") || _ident.startsWith("p2a")) {
                _ident = _ident.slice(0, 2) + _ident.slice(3);
            }
            const requestSidePokemon = request?.side?.pokemon ?? [];
            const privatePokemon = requestSidePokemon.filter(
                (x) => x?.ident === _ident
            )[0];

            let shouldBreak = false;

            for (const side of battle.sides) {
                const team = side.team;
                const activeIdents = side.active.map(
                    (value) => value?.originalIdent
                );
                for (const member of team) {
                    if (member.originalIdent === _ident) {
                        placeholder.updatePublicEntity(member);
                        placeholder.active = activeIdents.includes(
                            member.originalIdent
                        )
                            ? 1
                            : 0;

                        shouldBreak = true;
                        break;
                    }
                }
                if (shouldBreak) {
                    break;
                }
            }
            if (is_me && !is_public && _ident !== "") {
                placeholder.updatePrivateEntity(privatePokemon);
            }
        }
        placeholder.is_public = is_public ? 1 : 0;
        return placeholder;
    }

    getMyBattle() {
        return this.battles[0];
    }

    getBattleTurn() {
        return Math.min(127, this.getMyBattle().turn - 1);
    }

    getRequest(): AnyObject {
        const battle = this.getMyBattle() ?? {};
        return battle.request as AnyObject;
    }

    getOppBattle() {
        return this.battles[1];
    }

    getTurnLines(turn: number): string[] {
        return this.turns[turn] ?? [];
    }

    getAggregatedTurnLines() {
        const turn = Object.keys(this.turns).length - 1;
        const actionLines = this.getTurnLines(turn);
        let currentMoveInfo: {
            context: moveInfo;
            vector: Int8Array;
            hasVector: boolean;
        } = {
            context: new moveInfo(),
            vector: new Int8Array(historyVectorSize),
            hasVector: false,
        };

        let moveCounter = 0;
        let switchCounter = 0;

        for (const [lineIndex, line] of actionLines.entries()) {
            if (line === "|") {
                if (currentMoveInfo.context.isReady()) {
                    this.damageInfos[currentMoveInfo.context.hash()] =
                        currentMoveInfo;
                    currentMoveInfo = {
                        context: new moveInfo(),
                        vector: new Int8Array(historyVectorSize),
                        hasVector: false,
                    };
                }
            }
            if (line.startsWith("|move")) {
                if (currentMoveInfo.context.isReady()) {
                    this.damageInfos[currentMoveInfo.context.hash()] =
                        currentMoveInfo;
                    currentMoveInfo = {
                        context: new moveInfo(),
                        vector: new Int8Array(historyVectorSize),
                        hasVector: false,
                    };
                }
                const [_, __, user, moveName, target] = line.split("|");
                currentMoveInfo.context.userSide =
                    parseInt(user.slice(1, 2)) - 1;
                currentMoveInfo.context.user = user.slice(0, 2) + user.slice(3);
                currentMoveInfo.context.moveName = moveName;
                const targetCorrect = target === "" ? user : target ?? user;
                currentMoveInfo.context.targetSide =
                    parseInt(targetCorrect.slice(1, 2)) - 1;
                currentMoveInfo.context.target =
                    targetCorrect.slice(0, 2) + targetCorrect.slice(3);

                let offset = 0;
                const history = this.entityVectors[turn][lineIndex + 1];
                for (const datum of history) {
                    currentMoveInfo.vector.set(datum, offset);
                    offset += datum.length;
                }

                currentMoveInfo.context.moveCounter = moveCounter;
                currentMoveInfo.context.switchCounter = switchCounter;
                currentMoveInfo.context.order = turn;

                moveCounter = moveCounter + 1;
            } else if (line.startsWith("|switch")) {
                if (currentMoveInfo.context.isReady()) {
                    this.damageInfos[currentMoveInfo.context.hash()] =
                        currentMoveInfo;
                    currentMoveInfo = {
                        context: new moveInfo(),
                        vector: new Int8Array(historyVectorSize),
                        hasVector: false,
                    };
                }
                const [_, __, target, ...___] = line.split("|");

                const targetSideIndex = parseInt(target.slice(1, 2)) - 1;
                currentMoveInfo.context.targetSide = targetSideIndex;

                currentMoveInfo.context.moveName = "switch-in";
                currentMoveInfo.context.target =
                    target.slice(0, 2) + target.slice(3);

                currentMoveInfo.context.userSide =
                    currentMoveInfo.context.targetSide;

                const user =
                    this.getMyBattle().sides[targetSideIndex].active[0]
                        ?.originalIdent ?? currentMoveInfo.context.target;
                currentMoveInfo.context.user = user;

                let offset = 0;
                const history = this.entityVectors[turn][lineIndex + 1];
                for (const datum of history) {
                    currentMoveInfo.vector.set(datum, offset);
                    offset += datum.length;
                }

                currentMoveInfo.context.moveCounter = moveCounter;
                currentMoveInfo.context.switchCounter = switchCounter;
                currentMoveInfo.context.order = turn;

                switchCounter = switchCounter + 1;
            } else if (
                line.startsWith("|-damage") ||
                line.startsWith("|-heal")
            ) {
                currentMoveInfo.context.damage +=
                    this.damage[turn][lineIndex + 1];
            } else if (line.startsWith("|-crit")) {
                currentMoveInfo.context.isCritical = true;
            } else if (line.startsWith("|-supereffective")) {
                currentMoveInfo.context.effectiveness = 3;
            } else if (line.startsWith("|-resisted")) {
                currentMoveInfo.context.effectiveness = 2;
            } else if (line.startsWith("|-immune")) {
                currentMoveInfo.context.effectiveness = 0;
            } else if (line.startsWith("|-fail")) {
                currentMoveInfo.context.effectiveness = 4;
            } else if (line.startsWith("|faint")) {
                currentMoveInfo.context.targetFainted = true;
            } else if (line.startsWith("|-miss")) {
                currentMoveInfo.context.missed = true;
            }
        }
        if (currentMoveInfo.context.isReady()) {
            this.damageInfos[currentMoveInfo.context.hash()] = currentMoveInfo;
        }
        return this.damageInfos;
    }

    // appendTurnLine(
    //     playerIndex: number,
    //     workerIndex: number,
    //     line: string
    // ): void {
    //     if (this.turns[this.turn] === undefined) {
    //         this.turns[this.turn] = [];
    //         this.damage[this.turn] = {};
    //         this.entityVectors[this.turn] = {};
    //     }
    //     this.turns[this.turn].push(line);
    //     const battle = this.getMyBattle();
    //     if (line.startsWith("|move")) {
    //         let [_, __, user, ___, target] = line.split("|");
    //         const userSide = parseInt(user.slice(1, 2)) - 1;
    //         const targetCorrect = target === "" ? user : target ?? user;
    //         const targetSide = parseInt(targetCorrect.slice(1, 2)) - 1;

    //         const userIsMe = this.playerIndex === userSide ? 1 : 0;
    //         const targetIsMe = this.playerIndex === targetSide ? 1 : 0;

    //         const userProps = this.getPokemon(user);
    //         const targetProps = this.getPokemon(target);
    //         const userVector = userIsMe
    //             ? getPrivatePokemon(this.getMyBattle(), userProps)
    //             : getPublicPokemon(
    //                   this.getMyBattle(),
    //                   userProps,
    //                   true,
    //                   userIsMe
    //               );
    //         const targetVector = targetIsMe
    //             ? getPrivatePokemon(this.getMyBattle(), targetProps)
    //             : getPublicPokemon(
    //                   this.getMyBattle(),
    //                   targetProps,
    //                   true,
    //                   targetIsMe
    //               );
    //         const stateHandler = new Int8State({
    //             handler: this,
    //             playerIndex: playerIndex,
    //             workerIndex: workerIndex ?? 0,
    //             done: 0,
    //             reward: 0,
    //         });
    //         this.entityVectors[this.turn][this.turns[this.turn].length] = [
    //             ...stateHandler.getContextVectors(),
    //             userVector,
    //             targetVector,
    //         ];
    //     } else if (line.startsWith("|switch")) {
    //         let [_, __, target, ...___] = line.split("|");
    //         const targetSide = parseInt(target.slice(1, 2)) - 1;
    //         const user =
    //             this.getMyBattle().sides[targetSide].active[0]?.originalIdent ??
    //             target;
    //         const userSide = targetSide;

    //         const userIsMe = this.playerIndex === userSide ? 1 : 0;
    //         const targetIsMe = this.playerIndex === targetSide ? 1 : 0;

    //         const userProps = this.getPokemon(user);
    //         const targetProps = this.getPokemon(target);
    //         const userVector = userIsMe
    //             ? getPrivatePokemon(this.getMyBattle(), userProps)
    //             : getPublicPokemon(
    //                   this.getMyBattle(),
    //                   userProps,
    //                   true,
    //                   userIsMe
    //               );
    //         const targetVector = targetIsMe
    //             ? getPrivatePokemon(this.getMyBattle(), targetProps)
    //             : getPublicPokemon(
    //                   this.getMyBattle(),
    //                   targetProps,
    //                   true,
    //                   targetIsMe
    //               );
    //         const stateHandler = new Int8State({
    //             handler: this,
    //             playerIndex: playerIndex,
    //             workerIndex: workerIndex ?? 0,
    //             done: 0,
    //             reward: 0,
    //         });
    //         this.entityVectors[this.turn][this.turns[this.turn].length] = [
    //             ...stateHandler.getContextVectors(),
    //             userVector,
    //             targetVector,
    //         ];
    //     } else if (line.startsWith("|-damage") || line.startsWith("|-heal")) {
    //         let prevHp: number;
    //         const [_, __, user, healthRatio] = line.split("|");
    //         const originalIdent = user.slice(0, 2) + user.slice(3);
    //         const [numerator, denominator] = healthRatio.split("/");
    //         const currentHp = healthRatio.startsWith("0")
    //             ? 0
    //             : parseInt(numerator) / parseInt(denominator);
    //         for (const publicPokemon of [
    //             ...battle.p1.team,
    //             ...battle.p2.team,
    //         ]) {
    //             if (originalIdent === publicPokemon.originalIdent) {
    //                 prevHp = publicPokemon.hp / publicPokemon.maxhp;
    //                 const damage = currentHp - prevHp;
    //                 this.damage[this.turn][this.turns[this.turn].length] =
    //                     damage;
    //             }
    //         }
    //     }
    // }

    getEntityProperties(callables: { [k: string]: (entity: any) => any }) {
        const entities = [
            ...this.battles[0].p1.team,
            ...this.battles[0].p2.team,
        ];
        return Object.fromEntries(
            entities.map((entity) => [
                entity.originalIdent,
                Object.fromEntries(
                    Object.entries(callables).map(([feature, callable]) => [
                        feature,
                        callable(entity),
                    ])
                ),
            ])
        );
    }

    getState(args: {
        done: number;
        playerIndex: number;
        workerIndex?: number;
        reward?: number;
    }) {
        const { done, playerIndex, workerIndex, reward } = args;
        const stateHandler = new Int8State({
            handler: this,
            playerIndex: playerIndex,
            workerIndex: workerIndex ?? 0,
            done: done,
            reward: reward ?? 0,
        });
        const state = stateHandler.getState();
        const stateBuffer = Buffer.from(state.buffer);
        this.turn += 1;
        return stateBuffer;
    }

    getRandomActionString(playerIndex: number, workerIndex?: number): string {
        const stateHandler = new Int8State({
            handler: this,
            playerIndex: playerIndex,
            workerIndex: workerIndex ?? 0,
            done: 0,
            reward: 0,
        });
        const legalMask = stateHandler.getLegalMask();
        return getRandomAction(legalMask);
    }

    getMaxdmgActionString(playerIndex: number, workerIndex?: number): string {
        const stateHandler = new Int8State({
            handler: this,
            playerIndex: playerIndex,
            workerIndex: workerIndex ?? 0,
            done: 0,
            reward: 0,
        });
        const legalMask = stateHandler.getLegalMask();
        const player = new MaxdmgPlayer(this, playerIndex, legalMask);
        return player.getAction();
    }

    getHeuristicActionIndex(playerIndex: number, workerIndex?: number): number {
        const simpleHeuristic = new SimpleHeuristicPlayer(this, playerIndex);
        return simpleHeuristic.getActionIndex();
    }

    getHeuristicActionString(
        playerIndex: number,
        workerIndex?: number
    ): string {
        const simpleHeuristic = new SimpleHeuristicPlayer(this, playerIndex);
        const actionIndex = simpleHeuristic.getActionIndex();
        if (actionIndex < 4) {
            return `move ${actionIndex + 1}`;
        } else {
            return `switch ${actionIndex + 1 - 4}`;
        }
    }
}

export function actionCharToString(actionChar: string): string {
    const actionIndex = actionChar.charCodeAt(0);
    const actionIndexMinusOffset = actionIndex - zeroAsciiCode;
    // assuming its the string "0-9"

    if (0 <= actionIndexMinusOffset && actionIndexMinusOffset <= 3) {
        return `move ${actionIndexMinusOffset + 1}`;
    } else if (3 < actionIndexMinusOffset && actionIndexMinusOffset <= 9) {
        return `switch ${actionIndexMinusOffset - 3}`;
    } else if (actionIndex === d_AsciiCode) {
        return `default`;
    } else {
        return `default`;
    }
}

export function isAction(line: string): boolean {
    const splitString = line.split("|");
    const actionType = splitString[1];
    switch (actionType) {
        case "move":
            return true;
        case "switch":
            return true;
        default:
            return false;
    }
}

export function isActionRequired(battle: ClientBattle, chunk: string): boolean {
    const request = (battle.request ?? {}) as AnyObject;
    if (request === undefined) {
        return false;
    }
    if (request.teamPreview) {
        return true;
    }
    if (!!request.wait) {
        return false;
    }
    if (chunk.includes("|turn")) {
        return true;
    }
    if (!chunk.includes("|request")) {
        return !!request.forceSwitch;
    }
    return false;
}

export function formatTeamPreviewAction(
    action: string,
    totalPokemon: number
): string {
    const actionIndex = parseInt(action.split(" ")[1]);
    const remainder = arange(1, totalPokemon + 1)
        .filter((value) => value != actionIndex)
        .map((value) => `${value}`)
        .join("");
    return `team ${actionIndex}${remainder}`;
}

export function getIsTeamPreview(request: AnyObject): boolean {
    return request.teamPreview;
}

export interface InternalState {
    workerIndex: number;
}

export class AsyncQueue {
    private queue: string[];
    private resolveWaitingDequeue?: (value: string) => void;

    constructor() {
        this.queue = [];
    }

    enqueue(item: string): void {
        if (this.resolveWaitingDequeue) {
            this.resolveWaitingDequeue(item);
            this.resolveWaitingDequeue = undefined;
        } else {
            this.queue.push(item);
        }
    }

    async dequeue(): Promise<string> {
        if (this.queue.length > 0) {
            return this.queue.shift()!;
        }

        return new Promise<string>((resolve) => {
            this.resolveWaitingDequeue = resolve;
        });
    }
}

export const delay = (t: number | undefined, val: any = 0) =>
    new Promise((resolve) => setTimeout(resolve, t, val));
