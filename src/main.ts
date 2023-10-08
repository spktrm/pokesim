import path from "node:path";
import { Worker } from "node:worker_threads";

import * as net from "net";
import * as fs from "fs";
import * as yaml from "js-yaml";
import { numArange, weightedRandomSample } from "./random";

const maxWorkers = Math.max(parseInt(process.argv[2] ?? 1), 1);
console.log(`Max Workers: ${maxWorkers}`);

type Config = { [k: string]: any };
const config = yaml.load(
    fs.readFileSync(path.resolve("config.yml"), "utf-8")
) as Config;
console.log(config);

const socketPath = config.socket_path as string;
const serverUpdateFreq = config.server_update_freq as number;

const workers: Worker[] = [];

const debug = false;

let throughput = 0;

// Delete the socket if it already exists
if (fs.existsSync(socketPath)) {
    fs.unlinkSync(socketPath);
}

function processInput(input: string) {
    const [workIndex, playerIndex, actionChar] = input.split("|");
    const worker = workers[parseInt(workIndex)];
    const message = [parseInt(playerIndex), actionChar.trim()];
    worker.postMessage(message);
}

function incremetThroughput(_throughput: number) {
    throughput += _throughput;
}

const donesCache: { [k: number]: number } = {};
for (let i = 0; i < maxWorkers; i++) {
    donesCache[i] = 0;
}

const emptyWriteObject = {
    write: (message: Buffer) => {
        return;
    },
};

function createWorker(
    workerIndex: number,
    clientSocket: net.Socket | typeof emptyWriteObject = emptyWriteObject
) {
    const worker = new Worker(path.resolve(__dirname, "worker.js"), {
        workerData: { workerIndex },
    });
    worker.on("message", (message: Int8Array) => {
        incremetThroughput(1);
        const buffer = Buffer.from(message);

        const workerIndex = buffer[0];
        const done = buffer[2];

        donesCache[workerIndex] += done;

        switch (donesCache[workerIndex]) {
            case 0:
                if (debug) {
                    const playerIndex = buffer[1];
                    const legalMask = buffer.slice(-10);
                    const randomAction = weightedRandomSample(
                        numArange,
                        new Array(...legalMask),
                        1
                    );
                    processInput(
                        `${workerIndex}|${playerIndex}|${randomAction}`
                    );
                }
                break;
            case 2:
                const worker = workers[workerIndex];
                worker.postMessage("s");
                donesCache[workerIndex] = 0;
                break;
        }

        clientSocket.write(buffer);
    });
    workers.push(worker);
}

let prevTime = Date.now();
let throughputs: number[] = [];

setInterval(() => {
    let currTime = Date.now();
    const average = (1000 * throughput) / (currTime - prevTime);
    throughputs.push(average);
    while (throughputs.length > 5 * serverUpdateFreq) {
        throughputs.shift();
    }
    const totalAverage =
        throughputs.reduce((a, b) => a + b) / throughputs.length;
    const formattedAverage = totalAverage.toFixed(2).padStart(8); // Adjust the number 8 according to your desired width
    process.stdout.write(`\rThroughput: ${formattedAverage} steps / sec`);
    throughput = 0;
    prevTime = currTime;
}, 1000 / serverUpdateFreq);

let numWorkers = 0;

if (debug) {
    for (let workerIndex = 0; workerIndex < maxWorkers; workerIndex++) {
        createWorker(numWorkers);
    }
} else {
    interface InternalState {
        workerIndex: number;
    }

    const socketStates = new Map<net.Socket, InternalState>();

    const decoder = new TextDecoder("utf-8");
    const server = net.createServer((socket) => {
        console.log(`Client${numWorkers} connected`);
        createWorker(numWorkers, socket);

        const state: InternalState = {
            workerIndex: numWorkers,
        };
        numWorkers += 1;

        socketStates.set(socket, state);

        socket.on("data", (data) => {
            const socketState = socketStates.get(socket);
            if (socketState) {
                const workerIndex = socketState.workerIndex;
                processInput(`${workerIndex}|` + decoder.decode(data));
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
}
