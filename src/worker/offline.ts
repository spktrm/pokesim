import { parentPort, workerData } from "node:worker_threads";
import { BattleStreams, Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { BattleStreamsType } from "../types";
import { formatId } from "../data";
import {
    AsyncQueue,
    BattlesHandler,
    actionCharToString,
    formatTeamPreviewAction,
    delay,
    isAction,
    isActionRequired,
    getIsTeamPreview,
} from "../helpers";

Teams.setGeneratorFactory(TeamGenerators);

const workerIndex: number = workerData.workerIndex;
const generations = new Generations(Dex);

let state: Uint8Array;
let streams: BattleStreamsType;

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

const defaultWorkerIndex = workerData.config.default_worker_index as number;
const randomWorkerIndex = workerData.config.random_worker_index as number;
const prevWorkerIndex = workerData.config.prev_worker_index as number;
const heuristicWorkerIndex = workerData.config.heuristic_worker_index as number;

function isEvalPlayer(workerIndex: number, playerIndex: number): boolean {
    if (playerIndex === 1) {
        switch (workerIndex) {
            case defaultWorkerIndex:
                return true;
            case randomWorkerIndex:
                return true;
            default:
                return false;
        }
    } else {
        return false;
    }
}

async function runPlayer(
    stream: ObjectReadWriteStream<string>,
    playerIndex: number,
    p1battle: Battle,
    p2battle: Battle
) {
    // const handler = new BattlesHandler([p1battle, p2battle]);
    const handler = new BattlesHandler([p1battle]);
    const isEval = isEvalPlayer(workerIndex, playerIndex);

    const log = [];

    let isTeamPreview = false;
    let action = "";
    let winner = "";
    let reward = 0;

    // if (parseInt(formatId[3]) >= 5 && !formatId.includes("random")) {
    //     state = getState(handler, 0, playerIndex, workerIndex); //, reward);
    //     parentPort?.postMessage(state, [state.buffer]);
    //     const actionChar = await queueManager.queues[playerIndex].dequeue();
    //     action = actionCharToString(actionChar);
    //     stream.write(action);
    //     p1battle.request = undefined;
    // }

    for await (const chunk of stream) {
        const turn = p1battle.turn ?? 0;
        if (turn > 200) {
            break;
        }

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
            handler.appendTurnLine(line);
        }
        if (chunk.includes("|request")) {
            p1battle.update(); // optional, only relevant if stream contains |request|
        }

        if (isActionRequired(p1battle, chunk)) {
            isTeamPreview = getIsTeamPreview(p1battle.request ?? {});

            if (!isEval) {
                state = handler.getState({ done: 0, playerIndex, workerIndex }); //, reward);
                parentPort?.postMessage(state, [state.buffer]);
                const actionChar = await queueManager.queues[
                    playerIndex
                ].dequeue();
                action = actionCharToString(actionChar);
            } else {
                if (workerIndex === defaultWorkerIndex) {
                    action = "default";
                } else if (workerIndex === randomWorkerIndex) {
                    action = handler.getRandomAction(playerIndex, workerIndex);
                    // } else if (workerIndex === heuristicWorkerIndex) {
                    //     action = handler.getHeuristicAction(
                    //         playerIndex,
                    //         workerIndex
                    //     );
                } else {
                    action = "default";
                }
            }

            if (isTeamPreview && action != "default") {
                action = formatTeamPreviewAction(
                    action,
                    p1battle.sides[playerIndex].totalPokemon
                );
            }

            stream.write(action);
            p1battle.request = undefined;
        }
    }
    const hp_count = p1battle.sides.map((side) =>
        side.team.map((x) => x.hp / x.maxhp).reduce((a, b) => a + b)
    );
    reward = hp_count[playerIndex] > hp_count[1 - playerIndex] ? 1 : -1;
    // reward = winner === p1battle.sides[playerIndex].name ? 1 : -1;
    state = handler.getState({ done: 1, playerIndex, workerIndex, reward });
    parentPort?.postMessage(state, [state.buffer]);
    p1battle.request = undefined;
}

const exampleTeam = [
    "charmander|||blaze|ember,,,||85,,85,85,85,85|N|,0,,,,||5|,,,,,Grass",
    "bulbasaur|||torrent|bubble,,,||85,85,85,85,85,85|N|||5|,,,,,Fighting",
    "squirtle|||overgrow|vinewhip,,,||85,85,85,85,85,85|N|||5|,,,,,Steel",
];

const shuffle = (arr: any[]) =>
    arr
        .map((value) => ({ value, sort: Math.random() }))
        .sort((a, b) => a.sort - b.sort)
        .map(({ value }) => value);

async function runGame() {
    const stream = new BattleStreams.BattleStream();
    streams = BattleStreams.getPlayerStreams(stream);
    const spec = { formatid: formatId };

    const p1battle = new Battle(new Generations(Dex));
    const p2battle = new Battle(new Generations(Dex));

    const players = Promise.all([
        runPlayer(streams.p1, 0, p1battle, p2battle),
        runPlayer(streams.p2, 1, p2battle, p1battle),
    ]);

    const p1spec = {
        name: `Bot${workerIndex}1`,
        team: Teams.pack(Teams.generate(formatId)),
        // team: shuffle(exampleTeam).join("]"),
    };
    const p2spec = {
        name: `Bot${workerIndex}2`,
        team: Teams.pack(Teams.generate(formatId)),
        // team: shuffle(exampleTeam).join("]"),
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
        if (
            workerIndex === defaultWorkerIndex ||
            workerIndex === randomWorkerIndex
        ) {
            await delay(50);
        }
    }
})();

parentPort?.on("message", (message: string | any[]) => {
    if (message === "s") {
        startQueue.enqueue(message);
    } else {
        const [playerIndex, actionChar] = message;
        queueManager.queues[playerIndex].enqueue(actionChar);
    }
});
