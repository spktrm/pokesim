import { Pokemon, Side, Battle as clientBattle } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import { Int8State } from "./state";
import {
    arange,
    getRandomAction,
    numArange,
    weightedRandomSample,
} from "./random";
import { BoostID } from "@pkmn/data";
import { createHash } from "node:crypto";

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

    _getType(battle: clientBattle, atkType: string) {
        const type = battle.gen.types.get(atkType);
        return type;
    }

    _getSpecies(battle: clientBattle, name: string): any {
        return battle.gen.species.get(name);
    }

    _getMove(battle: clientBattle, name: string): any {
        return battle.gen.moves.get(getProperMoveName(name));
    }

    _getMoveScore(
        battle: clientBattle,
        active: Pokemon,
        opponent: Pokemon,
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
        battle: clientBattle,
        entity: Pokemon,
        type: string
    ): number {
        const multipliers = entity.types.map((entityType: string) => {
            const typeData = this._getType(battle, entityType) as any;
            const typeValue = typeData.damageTaken[type] ?? 0;
            return typeStringToFloat(typeValue.toString());
        });
        return multipliers.reduce((a: number, b: number) => a * b);
    }

    _estimateMatchup(mon: Pokemon | null, opponent: Pokemon | null): number {
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

    _statEstimation(mon: Pokemon, stat: "atk" | "def" | "spa" | "spd" | "spe") {
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
        switches: Pokemon[],
        active: Pokemon | null,
        opponent: Pokemon | null
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

    getSwitchIndex(switches: Pokemon[], opponent: Pokemon | null): number {
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
            .map((ident: string) => this.handler.getPokemon(sideIndex, ident));

        const mySide = battle.sides[this.playerIndex];
        const oppSide = battle.sides[1 - this.playerIndex];
        const opponent = oppSide.active[0];

        if (active === null || (request.forceSwitch ?? false)) {
            return this.getSwitchIndex(switches, opponent);
        }
        const requestActive = request.active[0];

        const privateActive = this.handler.getPokemon(
            sideIndex,
            active.originalIdent
        );

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
                const hpRatio = privateActive.hp / privateActive.maxhp;
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

    isReady() {
        return this.moveName !== undefined;
    }

    hash() {
        return createHash("sha256")
            .update(
                `${this.user}|${this.target}|${this.moveName}|${this.damage}|${this.effectiveness}|${this.isCritical}|${this.targetFainted}`
            )
            .digest("hex");
    }

    serialize(): { [k: string]: any } {
        return {
            user: this.user,
            target: this.target,
            moveName: this.moveName,
            damage: this.damage,
            effectiveness: this.effectiveness,
            isCritical: this.isCritical,
        };
    }
}

export const historyVectorSize = 177;
export class BattlesHandler {
    battles: clientBattle[];
    turn: number;
    turns: AnyObject;
    damage: AnyObject;
    damageInfos: {
        [k: string]: {
            context: moveInfo;
            vector: Int8Array;
            hasVector: boolean;
        };
    };

    constructor(battles: clientBattle[]) {
        this.battles = battles;
        this.turn = 0;
        this.turns = {};
        this.damage = {};
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

    getPokemon(sideIndex: number, ident: string) {
        const battle = this.getMyBattle();
        const side = battle.getSide(`p${sideIndex + 1}`);
        const team = side.team;
        const request = battle.request;
        const requestSidePokemon = request?.side?.pokemon ?? [];
        const publicPokemon =
            team.filter((x) => x.originalIdent === ident)[0] ?? {};
        const privatePokemon =
            requestSidePokemon.filter((x) => x?.ident === ident)[0] ?? {};
        const moves = privatePokemon.moves;
        return {
            ...publicPokemon,
            ...privatePokemon,
            moves,
        };
    }

    getMyBattle() {
        return this.battles[0];
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
        let currentMoveInfo: moveInfo = new moveInfo();
        for (const [lineIndex, line] of actionLines.entries()) {
            if (!line.startsWith("|-")) {
                if (currentMoveInfo.isReady()) {
                    this.damageInfos[currentMoveInfo.hash()] = {
                        context: currentMoveInfo,
                        vector: new Int8Array(historyVectorSize),
                        hasVector: false,
                    };
                    currentMoveInfo = new moveInfo();
                }
            }

            if (line.startsWith("|move")) {
                const [_, __, user, moveName, target] = line.split("|");
                currentMoveInfo.userSide = parseInt(user.slice(1, 2)) - 1;
                currentMoveInfo.user = user.slice(0, 2) + user.slice(3);
                currentMoveInfo.moveName = moveName;
                const targetCorrect = target ?? user;
                currentMoveInfo.targetSide =
                    parseInt(targetCorrect.slice(1, 2)) - 1;
                currentMoveInfo.target =
                    targetCorrect.slice(0, 2) + targetCorrect.slice(3);
            } else if (
                line.startsWith("|-damage") ||
                line.startsWith("|-heal")
            ) {
                currentMoveInfo.damage += this.damage[turn][lineIndex + 1];
            } else if (line.startsWith("|-crit")) {
                currentMoveInfo.isCritical = true;
            } else if (line.startsWith("|-supereffective")) {
                currentMoveInfo.effectiveness = 3;
            } else if (line.startsWith("|-resisted")) {
                currentMoveInfo.effectiveness = 2;
            } else if (line.startsWith("|-immune")) {
                currentMoveInfo.effectiveness = 0;
            } else if (line.startsWith("|-fail")) {
                currentMoveInfo.effectiveness = 4;
            } else if (line.startsWith("|faint")) {
                currentMoveInfo.targetFainted = true;
            } else if (line.startsWith("|-miss")) {
                currentMoveInfo.missed = true;
            }
        }
        return this.damageInfos;
    }

    appendTurnLine(line: string): void {
        if (this.turns[this.turn] === undefined) {
            this.turns[this.turn] = [];
            this.damage[this.turn] = {};
        }
        this.turns[this.turn].push(line);
        const battle = this.getMyBattle();
        if (line.startsWith("|-damage") || line.startsWith("|-heal")) {
            let prevHp: number;
            const [_, __, user, healthRatio] = line.split("|");
            const originalIdent = user.slice(0, 2) + user.slice(3);
            const [numerator, denominator] = healthRatio.split("/");
            const currentHp = healthRatio.startsWith("0")
                ? 0
                : parseInt(numerator) / parseInt(denominator);
            for (const publicPokemon of [
                ...battle.p1.team,
                ...battle.p2.team,
            ]) {
                if (originalIdent === publicPokemon.originalIdent) {
                    prevHp = publicPokemon.hp / publicPokemon.maxhp;
                    const damage = currentHp - prevHp;
                    this.damage[this.turn][this.turns[this.turn].length] =
                        damage;
                }
            }
        }
    }

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

    getRandomAction(playerIndex: number, workerIndex?: number): string {
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

    getMaxdmgAction(playerIndex: number, workerIndex?: number): string {
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

    getHeuristicAction(playerIndex: number, workerIndex?: number): number {
        const simpleHeuristic = new SimpleHeuristicPlayer(this, playerIndex);
        return simpleHeuristic.getActionIndex();
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

export function isActionRequired(battle: clientBattle, chunk: string): boolean {
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
