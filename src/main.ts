import path from "node:path";
import { Worker } from "node:worker_threads";

import * as net from "net";
import * as fs from "fs";
import * as yaml from "js-yaml";
import { arange } from "./random";

const numWorkers = Math.max(parseInt(process.argv[2] ?? 1), 1);
console.log(`Num Workers: ${numWorkers}`);

type Config = { [k: string]: any };
const config = yaml.load(fs.readFileSync("config.yml", "utf-8")) as Config;
console.log(config);

const socketPath = config.socket_path as string;
const serverUpdateFreq = config.server_update_freq as number;

const workers: Worker[] = [];
const buffers: Int8Array[] = [];

const debug = false;

let throughput = 0;

// Delete the socket if it already exists
if (fs.existsSync(socketPath)) {
    fs.unlinkSync(socketPath);
}

let clientSocket: net.Socket | null = null;
let stateSize: number | undefined = undefined;

function processInput(input: string) {
    const [workIndex, playerIndex, actionChar] = input.split("|");
    const worker = workers[parseInt(workIndex)];
    const message = [parseInt(playerIndex), actionChar];
    worker.postMessage(message);
}

function getConcatenatedBuffer() {
    if (stateSize === undefined) {
        stateSize = buffers[0].length;
    }
    const concatenatedBuffer = new Buffer(
        buffers.length * stateSize //+ stopBytes.length
    );
    let offset = 0;
    for (const buffer of buffers) {
        concatenatedBuffer.set(buffer, offset);
        offset += buffer.length;
    }
    // concatenatedBuffer.set(stopBytes, offset);
    return concatenatedBuffer;
}

function sendConcatenatedBuffers(concatenatedBuffers: Buffer) {
    if (clientSocket) {
        clientSocket.write(Buffer.from([buffers.length]));
        clientSocket.write(concatenatedBuffers);
    } else if (debug) {
        // process.stdout.write(concatenatedBuffers);
    } else {
        console.error("Client socket is not connected");
    }
}

function incremetThroughput(_throughput: number) {
    throughput += _throughput;
}

const validActionTokens = ["r", "d"];

function randomActionToken(): string {
    return validActionTokens[
        Math.round(Math.random() * validActionTokens.length)
    ];
}

function sendBuffers() {
    if (buffers.length > 0) {
        incremetThroughput(buffers.length);
        const concatenatedBuffers = getConcatenatedBuffer();
        sendConcatenatedBuffers(concatenatedBuffers);

        if (debug) {
            let workerIndex: number;
            let playerIndex: number;
            for (let buffer of buffers) {
                workerIndex = buffer[0];
                playerIndex = buffer[1];
                processInput(
                    `${workerIndex}|${playerIndex}|${randomActionToken()}`
                );
            }
        }

        buffers.splice(0, buffers.length); // Empty the buffers array
    }
}

const trigger = Math.max(1, numWorkers - 2);

function start() {
    for (let workerIndex = 0; workerIndex < numWorkers; workerIndex++) {
        const worker = new Worker(path.resolve(__dirname, "worker.js"), {
            workerData: { workerIndex },
        });
        worker.on("message", (message) => {
            buffers.push(message);

            if (buffers.length === trigger) {
                sendBuffers();
            }
        });
        workers.push(worker);
    }
}

// setInterval(() => {
//     sendBuffers();
// }, timeout);

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

if (debug) {
    start();
} else {
    const decoder = new TextDecoder("utf-8");
    const server = net.createServer((client) => {
        console.log("Client connected");
        start();
        clientSocket = client; // Store the client socket for later use

        client.on("data", (data) => {
            let splitCmds = decoder.decode(data).split("\n");
            for (let cmd of splitCmds) {
                processInput(cmd);
            }
        });

        client.on("end", () => {
            console.log("Client disconnected");
            clientSocket = null; // Clear the client socket when the client disconnects
            workers.splice(workers.length);
        });
    });

    server.listen(socketPath, () => {
        console.log("Server listening on", socketPath);
    });

    server.on("error", (err) => {
        console.error(err);
    });
}
