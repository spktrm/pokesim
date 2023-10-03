import { parentPort, workerData } from "node:worker_threads";

import { AnyObject, BattleStreams, Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { BattleStreamsType } from "./types";
import { Int8State } from "./state";
import { formatid } from "./data";

Teams.setGeneratorFactory(TeamGenerators);

let state: Uint8Array;
const workerIndex: number = workerData.workerIndex;
const generations = new Generations(Dex);

let streams: BattleStreamsType;

function timeout(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function isActionRequired(battle: Battle, chunk: string): boolean {
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
    battles: Battle[];
    turns: AnyObject;
    constructor(battles: Battle[]) {
        this.battles = battles;
        this.turns = {};
    }
}

function getState(
    handler: BattlesHandler,
    done: number,
    playerIndex: number,
    reward: number = 0,
) {
    const stateHandler = new Int8State(
        handler,
        playerIndex,
        workerIndex,
        done,
        reward,
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

async function runPlayer(
    globalStream: ObjectReadWriteStream<string>,
    stream: ObjectReadWriteStream<string>,
    playerIndex: number,
    p1battle: Battle,
    p2battle: Battle,
) {
    const log = [];
    const handler = new BattlesHandler([p1battle, p2battle]);
    const turn = p1battle.turn ?? 0;
    const battle = globalStream;

    let prevChunk: string = "";
    let prevRequest: AnyObject = {};
    let winner: string = "";

    for await (const chunk of stream) {
        // Alternatively: for (const {args, kwArgs} of Protocol.parse(chunk))
        for (const line of chunk.split("\n")) {
            if (line.startsWith("|error")) {
                console.error(line);
            }
            p1battle.add(line);
            log.push(line);
            if (line.startsWith("|win")) {
                winner = line.split("|")[2];
            }
            if (isAction(line)) {
                if (handler.turns[turn] === undefined) {
                    handler.turns[turn] = [];
                }
                handler.turns[turn].push(line);
            }
        }
        p1battle.update(); // optional, only relevant if stream contains |request|

        if (isActionRequired(p1battle, chunk)) {
            prevChunk = chunk;
            prevRequest = p1battle.request ?? {};
            state = getState(handler, 0, playerIndex);
            parentPort?.postMessage(state, [state.buffer]);
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

    const p1battle = new Battle(generations);
    const p2battle = new Battle(generations);

    const players = Promise.all([
        runPlayer(stream, streams.p1, 0, p1battle, p2battle),
        runPlayer(stream, streams.p2, 1, p2battle, p1battle),
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

function actionIndexToString(actionIndex: number) {
    if (0 <= actionIndex && actionIndex <= 3) {
        return `move ${actionIndex + 1}`;
    } else if (3 < actionIndex && actionIndex <= 10) {
        return `switch ${actionIndex - 4 + 1}`;
    } else {
        return `default`;
    }
}

type playerIdType = "p1" | "p2";

parentPort?.on("message", (message) => {
    const [playerIndex, actionIndex] = message;
    const playerId = `p${playerIndex + 1}` as playerIdType;
    const actionString = actionIndexToString(parseInt(actionIndex));
    const stream = streams[playerId];
    stream.write(actionString);
});

(async () => {
    while (true) {
        await runGame();
    }
})();
