import { Pokemon, Side, Battle as clientBattle } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import { Int8State } from "./state";
import { arange, getRandomAction } from "./random";
import { BoostID } from "@pkmn/types";
import { Type } from "@pkmn/dex";

const zeroAsciiCode = "0".charCodeAt(0);
const d_AsciiCode = "d".charCodeAt(0);

type identType = { active: (string | undefined)[]; team: string[] };

// class SimpleHeuristicPlayer {
//     handler: BattlesHandler;
//     playerIndex: number;

//     constructor(handler: BattlesHandler, playerIndex: number) {
//         this.handler = handler;
//         this.playerIndex = playerIndex;
//     }

//     _getTypeMatchup(battle: clientBattle, atkType: string, type: string) {
//         return battle.gen.types.get(atkType); //!.damageTaken![type];
//     }

//     _estimateMatchup(attacker: Pokemon, defender: Pokemon): number {
//         const battle = this.handler.getMyBattle();

//         let score = Math.max(
//             ...defender.types.map((type) =>
//                 attacker.types
//                     .map((atkType) =>
//                         this._getTypeMatchup(battle, atkType, type)
//                     )
//                     .reduce((a, b) => a * b, 1)
//             )
//         );
//         score -= Math.max(
//             ...attacker.species.types.map((type) => {
//                 if (type === undefined) {
//                     return 1;
//                 } else {
//                     return defender.species.types
//                         .map(
//                             (atkType) =>
//                                 (
//                                     battle.gen.types.get(
//                                         atkType
//                                     ) as unknown as Type
//                                 ).damageTaken[type]
//                         )
//                         .reduce((a, b) => a * b);
//                 }
//             })
//         );

//         return score;
//     }

//     _statEstimation(mon: Pokemon, stat: "atk" | "def" | "spa" | "spd" | "spe") {
//         let boost: number;

//         const statValue = mon.boosts[stat] ?? 0;
//         const baseStats = mon.species.baseStats;

//         if (statValue > 1) {
//             boost = (2 + statValue) / 2;
//         } else {
//             boost = 2 / (2 - statValue);
//         }
//         return (2 * baseStats[stat] + 31 + 5) * boost;
//     }

//     _shouldSwitchOut(
//         switches: Pokemon[],
//         active: Pokemon,
//         opponent: Pokemon
//     ): boolean {
//         const isBadMatchup = switches.filter((value) => {
//             this._estimateMatchup(value, opponent) > 0;
//         });
//         if (isBadMatchup) {
//             const activeBoosts = active.boosts ?? {};
//             if (
//                 (activeBoosts.def ?? 0 <= -3) ||
//                 (activeBoosts.spd ?? 0 <= -3)
//             ) {
//                 return true;
//             }
//             if (
//                 (activeBoosts.atk ?? 0) <= -3 &&
//                 (activeBoosts.atk ?? 0) >= (activeBoosts.spa ?? 0)
//             ) {
//                 return true;
//             }
//             if (
//                 (activeBoosts.spa ?? 0) <= -3 &&
//                 (activeBoosts.atk ?? 0) <= (activeBoosts.spa ?? 0)
//             ) {
//                 return true;
//             }
//             if (this._estimateMatchup(active, opponent) < -2) {
//                 return true;
//             }
//         }
//         return false;
//     }

//     getAction(): string {
//         const battle = this.handler.getMyBattle();
//         const stateHandler = new Int8State(
//             this.handler,
//             this.playerIndex,
//             0,
//             0,
//             0
//         );
//         const legalMask = stateHandler.getLegalMask();
//         const availableMoves = legalMask.slice(0, 4).reduce((a, b) => a + b);
//         const availableSwitches = legalMask.slice(4).reduce((a, b) => a + b);

//         const active = battle.sides[this.playerIndex].active[0];
//         const switches = battle.sides[this.playerIndex].team.filter((x) =>
//             x.isActive()
//         );
//         const opponent = battle.sides[1 - this.playerIndex].active[0];

//         const calcRatio = active !== null && opponent !== null;

//         const physicalRatio = calcRatio
//             ? this._statEstimation(active, "atk") /
//               this._statEstimation(opponent, "def")
//             : 0;
//         const specialRatio = calcRatio
//             ? this._statEstimation(active, "spa") /
//               this._statEstimation(opponent, "spd")
//             : 0;

//         if (
//             availableMoves &&
//             (!this._shouldSwitchOut(
//                 switches,
//                 active as Pokemon,
//                 opponent as Pokemon
//             ) ||
//                 !availableSwitches)
//         ) {
//         }

//         return "default";
//     }
// }

class MaxdmgPlayer {
    handler: BattlesHandler;
    playerIndex: number;
    legalMask: Int8Array;

    constructor(
        handler: BattlesHandler,
        playerIndex: number,
        legalMask: Int8Array,
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
                (value: { id: any }) => value.id,
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
                0,
            );
        }

        return maxMoveIndex === -1 ? "default" : `move ${maxMoveIndex + 1}`;
    }
}

export class BattlesHandler {
    battles: clientBattle[];
    turns: AnyObject;
    prevIdents: identType[];

    constructor(battles: clientBattle[]) {
        this.battles = battles;
        this.turns = {};
        this.prevIdents = [
            {
                active: [],
                team: [],
            },
            {
                active: [],
                team: [],
            },
        ];
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

    getMyBattle() {
        return this.battles[0];
    }

    getOppBattle() {
        return this.battles[1];
    }

    getTurnLines(turn: number): string[] {
        return this.turns[turn] ?? [];
    }

    appendTurnLine(turn: number, line: string): void {
        if (this.turns[turn] === undefined) {
            this.turns[turn] = [];
        }
        this.turns[turn].push(line);
    }

    getState(
        done: number,
        playerIndex: number,
        workerIndex?: number,
        reward?: number,
    ) {
        const stateHandler = new Int8State(
            this,
            playerIndex,
            workerIndex ?? 0,
            done,
            reward ?? 0,
        );
        const state = stateHandler.getState();
        const sides: Side[] = [
            stateHandler.getMyPublicSide(),
            stateHandler.getOppSide(),
        ];
        this.prevIdents = sides.map((side) => {
            return {
                active: side.active.map((x) => x?.ident.toString()),
                team: side.team.map((x) => x.ident.toString()),
            };
        });
        const stateBuffer = Buffer.from(state.buffer);
        return stateBuffer;
    }

    getRandomAction(playerIndex: number, workerIndex?: number): string {
        const stateHandler = new Int8State(
            this,
            playerIndex,
            workerIndex ?? 0,
            0,
            0,
        );
        const legalMask = stateHandler.getLegalMask();
        return getRandomAction(legalMask);
    }

    getMaxdmgAction(playerIndex: number, workerIndex?: number): string {
        const stateHandler = new Int8State(
            this,
            playerIndex,
            workerIndex ?? 0,
            0,
            0,
        );
        const legalMask = stateHandler.getLegalMask();
        const player = new MaxdmgPlayer(this, playerIndex, legalMask);
        return player.getAction();
    }

    getHeuristicAction(playerIndex: number, workerIndex?: number): string {
        return "default";
        // const simpleHeuristic = new SimpleHeuristicPlayer(this, playerIndex);
        // return simpleHeuristic.getAction();
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
    totalPokemon: number,
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
