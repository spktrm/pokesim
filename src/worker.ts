import { parentPort, workerData } from "node:worker_threads";
import { AnyObject, BattleStreams, Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { Battle as clientBattle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { BattleStreamsType } from "./types";
import { formatid } from "./data";
import { getRandomAction } from "./random";
import { actionCharToString } from "./helpers";
import { Int8State } from "./state";

Teams.setGeneratorFactory(TeamGenerators);

const workerIndex: number = workerData.workerIndex;
const generations = new Generations(Dex);

let state: Uint8Array;
let streams: BattleStreamsType;

const delay = (t: number | undefined, val: any = 0) =>
    new Promise((resolve) => setTimeout(resolve, t, val));

class AsyncQueue {
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

class QueueManager {
    p1Queue: AsyncQueue;
    p2Queue: AsyncQueue;
    queues: AsyncQueue[];
    constructor() {
        this.p1Queue = new AsyncQueue();
        this.p2Queue = new AsyncQueue();
        this.queues = [this.p1Queue, this.p2Queue];
    }
}

const queueManager = new QueueManager();

function isActionRequired(battle: clientBattle, chunk: string): boolean {
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

export class BattlesHandler {
    battles: clientBattle[];
    turns: AnyObject;
    constructor(battles: clientBattle[]) {
        this.battles = battles;
        this.turns = {};
    }
}

function getState(
    handler: BattlesHandler,
    done: number,
    playerIndex: number,
    reward: number = 0
) {
    const stateHandler = new Int8State(
        handler,
        playerIndex,
        workerIndex,
        done,
        reward
    );
    const state = stateHandler.getState();
    const stateBuffer = Buffer.from(state.buffer);
    return stateBuffer;
}

function isAction(line: string): boolean {
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

const defaultWorkerIndex = 18;
const randomWorkerIndex = 19;

function isEvalPlayer(workerIndex: number, playerIndex: number): boolean {
    if (workerIndex >= 18 && playerIndex === 1) {
        return true;
    } else {
        return false;
    }
}

async function runPlayer(
    stream: ObjectReadWriteStream<string>,
    playerIndex: number,
    p1battle: clientBattle,
    p2battle: clientBattle
) {
    const handler = new BattlesHandler([p1battle, p2battle]);
    const turn = p1battle.turn ?? 0;
    const isEval = isEvalPlayer(workerIndex, playerIndex);

    const log = [];

    let action: string = "";
    let winner: string = "";

    for await (const chunk of stream) {
        // Alternatively: for (const {args, kwArgs} of Protocol.parse(chunk))
        for (const line of chunk.split("\n")) {
            if (line.startsWith("|error")) {
                console.error(line);
            }
            p1battle.add(line);
            if (line.startsWith("|win")) {
                winner = line.split("|")[2];
            }
            log.push(line);
            if (isAction(line)) {
                if (handler.turns[turn] === undefined) {
                    handler.turns[turn] = [];
                }
                handler.turns[turn].push(line);
            }
        }
        if (chunk.includes("|request")) {
            p1battle.update(); // optional, only relevant if stream contains |request|
        }

        if (isActionRequired(p1battle, chunk)) {
            if (!isEval) {
                state = getState(handler, 0, playerIndex);
                parentPort?.postMessage(state, [state.buffer]);
                const actionChar = await queueManager.queues[
                    playerIndex
                ].dequeue();
                action = actionCharToString(actionChar);
            } else {
                if (workerIndex === defaultWorkerIndex) {
                    action = "default";
                } else if (workerIndex === randomWorkerIndex) {
                    const stateHandler = new Int8State(
                        handler,
                        playerIndex,
                        workerIndex,
                        0,
                        0
                    );
                    const legalMask = stateHandler.getLegalMask();
                    action = getRandomAction(legalMask);
                } else {
                    action = "default";
                }
            }
            stream.write(action);
        }
    }
    const reward = winner === p1battle.sides[playerIndex].name ? 1 : -1;
    state = getState(handler, 1, playerIndex, reward);
    parentPort?.postMessage(state, [state.buffer]);
}

async function runGame() {
    const stream = new BattleStreams.BattleStream();
    streams = BattleStreams.getPlayerStreams(stream);
    const spec = { formatid };

    const p1battle = new clientBattle(generations);
    const p2battle = new clientBattle(generations);

    const players = Promise.all([
        runPlayer(streams.p1, 0, p1battle, p2battle),
        runPlayer(streams.p2, 1, p2battle, p1battle),
    ]);

    const p1spec = {
        name: `Bot${workerIndex}1`,
        team: Teams.pack(Teams.generate(formatid)),
    };
    const p2spec = {
        name: `Bot${workerIndex}2`,
        team: Teams.pack(Teams.generate(formatid)),
    };

    void streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);

    return await players;
}

const startQueue = new AsyncQueue();

(async () => {
    while (true) {
        await runGame();
        await startQueue.dequeue();
    }
})();

parentPort?.on("message", (message) => {
    if (message === "s") {
        startQueue.enqueue(message);
    } else {
        const [playerIndex, actionChar] = message;
        queueManager.queues[playerIndex].enqueue(actionChar);
    }
});
