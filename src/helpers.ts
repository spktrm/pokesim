import { Battle as clientBattle } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import { Int8State } from "./state";

const zeroAsciiCode = "0".charCodeAt(0);
const d_AsciiCode = "d".charCodeAt(0);

export class BattlesHandler {
    battles: clientBattle[];
    turns: AnyObject;
    constructor(battles: clientBattle[]) {
        this.battles = battles;
        this.turns = {};
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

export function getState(
    handler: BattlesHandler,
    done: number,
    playerIndex: number,
    workerIndex?: number,
    reward?: number
) {
    const stateHandler = new Int8State(
        handler,
        playerIndex,
        workerIndex ?? 0,
        done,
        reward ?? 0
    );
    const state = stateHandler.getState();
    const stateBuffer = Buffer.from(state.buffer);
    return stateBuffer;
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
