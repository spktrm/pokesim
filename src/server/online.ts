import path from "node:path";
import { Worker } from "node:worker_threads";

import * as net from "net";
import * as fs from "fs";
import * as yaml from "js-yaml";

import { InternalState } from "../helpers";
import { argv } from "node:process";

type Config = { [k: string]: any };

const debug = argv[2] === "debug";

function loadConfig(path: string): Config {
    let config = yaml.load(fs.readFileSync(path, "utf-8")) as Config;
    const numWorkers = config.num_workers as number;
    for (const [key, value] of Object.entries(config)) {
        if (key.endsWith("worker_index")) {
            if (value <= 0) {
                config[key] = numWorkers + value;
            }
        }
    }
    return { ...config };
}

const config = loadConfig(path.resolve("config.yml"));
console.log(config);

const numWorkers = config.num_workers as number;
const socketPath = config.socket_path as string;
const serverUpdateFreq = config.server_update_freq as number;
console.log(`Max Workers: ${numWorkers}`);

const workers: Worker[] = [];

const throughputs: number[] = new Array(numWorkers).fill(0);
const prevThroughputs: number[] = [...throughputs];

// Delete the socket if it already exists
if (fs.existsSync(socketPath)) {
    fs.unlinkSync(socketPath);
}

function processInput(input: string) {
    const [workIndex, playerIndex, actionChar] = input.split("|");
    const worker = workers[parseInt(workIndex)];
    const message = [parseInt(playerIndex), actionChar];
    worker.postMessage(message);
}

const donesCache: { [k: number]: number } = {};
for (let i = 0; i < numWorkers; i++) {
    donesCache[i] = 0;
}

const emptyWriteObject = {
    write: (message: Buffer) => {
        return;
    },
};

function createWorker(
    workerIndex: number,
    clientSocket: net.Socket | typeof emptyWriteObject = emptyWriteObject,
) {
    const worker = new Worker(path.resolve(__dirname, "../worker/online.js"), {
        workerData: { workerIndex, config: { ...config }, debug },
    });
    worker.on("message", (buffer: Buffer) => {
        const workerIndex = buffer[0];
        // const playerIndex = buffer[1];
        const done = buffer[2];

        throughputs[workerIndex] += 1;
        donesCache[workerIndex] += done;

        switch (donesCache[workerIndex]) {
            case 2:
                const worker = workers[workerIndex];
                worker.postMessage("s");
                donesCache[workerIndex] = 0;
                break;
        }

        const write = clientSocket.write(buffer);
        if (!write) {
            console.error(`Worker ${workerIndex} queued in memory`);
        }
    });
    worker.on("error", (error: Error) => {
        console.error(error);
    });
    workers.push(worker);
}

const maxWorkerIndexLength = 4;
const maxStepsLength = 15;
const maxThroughputLength = 6;
const updateFreq = 1000 / serverUpdateFreq;
const minEvalIndex = Math.min(
    config.default_worker_index,
    config.random_worker_index,
    config.heuristic_worker_index,
);

if (!debug) {
    setInterval(() => {
        let log = "";

        let totalThroughputs = 0;
        let totalSteps = 0;

        for (let index = 0, len = throughputs.length; index < len; index++) {
            const value = throughputs[index];
            const throughput = value - prevThroughputs[index];
            prevThroughputs[index] = value;

            log +=
                "Worker " +
                index.toString().padEnd(maxWorkerIndexLength) +
                " Steps: " +
                value.toString().padEnd(maxStepsLength) +
                " Throughput/sec: " +
                (throughput * serverUpdateFreq)
                    .toFixed(0)
                    .padEnd(maxThroughputLength) +
                "\n";

            if (index < minEvalIndex) {
                totalThroughputs += throughput;
                totalSteps += value;
            }
        }

        log +=
            "\nTotal - Steps: " +
            totalSteps
                .toString()
                .padEnd(maxStepsLength + maxWorkerIndexLength + 8) +
            " Throughput/Sec: " +
            (totalThroughputs * serverUpdateFreq).toFixed(2);

        console.clear();
        console.log(log);
    }, updateFreq);
}

let numConnections = 0;
const socketStates = new Map<net.Socket, InternalState>();
const decoder = new TextDecoder("utf-8");

const server = net.createServer((socket) => {
    console.log(`Client${numConnections} connected`);
    createWorker(numConnections, socket);

    const state: InternalState = {
        workerIndex: numConnections,
    };
    numConnections += 1;

    socketStates.set(socket, state);

    socket.on("data", (data) => {
        const socketState = socketStates.get(socket);
        if (socketState) {
            const workerIndex = socketState.workerIndex;
            const decodedData = decoder.decode(data);
            for (let cmd of decodedData.split("\n")) {
                if (cmd) {
                    processInput(`${workerIndex}|` + cmd);
                }
            }
        }
    });

    socket.on("end", () => {
        const socketState = socketStates.get(socket);
        if (socketState) {
            const workerIndex = socketState.workerIndex;
            console.log("Client disconnected");
            delete workers[workerIndex];
        }
    });
});

server.listen(socketPath, () => {
    console.log("Server listening on", socketPath);
});

server.on("error", (err) => {
    console.error(err);
});
