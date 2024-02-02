import WebSocket = require("ws");

import path from "node:path";

import * as net from "net";
import * as fs from "fs";
import * as yaml from "js-yaml";

import { AnyObject, Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { Pokemon, Battle as clientBattle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import {
    AsyncQueue,
    BattlesHandler,
    InternalState,
    actionCharToString,
    isAction,
    isActionRequired,
} from "../helpers";
import { formatId } from "../data";

type Config = { [k: string]: any };
const config = yaml.load(
    fs.readFileSync(path.resolve("config.yml"), "utf-8"),
) as Config;
console.log(config);

const numWorkers = config.num_workers as number;
const socketPath = config.socket_path as string;
console.log(`Max Workers: ${numWorkers}`);

// Delete the socket if it already exists
if (fs.existsSync(socketPath)) {
    fs.unlinkSync(socketPath);
}

Teams.setGeneratorFactory(TeamGenerators);
const generations = new Generations(Dex);

function logRequest(request: AnyObject) {
    const actions: string[] = [];
    try {
        actions.push(
            ...((request ?? {}).active ?? [])[0].moves.map(
                (x: { [k: string]: any }) => x.id,
            ),
        );
    } catch {}
    try {
        actions.push(
            ...(((request ?? {}).side ?? {}).pokemon ?? []).map(
                (x: { [k: string]: any }) => x.ident,
            ),
        );
    } catch {}

    console.log(actions);
}

const localHost = "localhost:8000";
const online = "sim3.psim.us";

class PokemonShowdownBot {
    private ws: WebSocket;
    private readonly url: string = `ws://${localHost}/showdown/websocket`;
    username: string;
    password: string | undefined;
    clientBattle: clientBattle;
    handler: BattlesHandler;
    playerIndex: number | undefined;
    playerIndexIsSet: boolean;
    clientSocket: net.Socket;
    msgQueue: AsyncQueue;
    started: boolean;
    battleId: string | undefined;

    constructor(clientSocket: net.Socket, username: string, password?: string) {
        this.username = username;
        this.password = password;
        this.ws = new WebSocket(this.url);
        this.ws.on("open", () => console.log("Connected to server"));
        this.ws.on("message", this.onMessage.bind(this));
        this.clientBattle = new clientBattle(generations);
        this.handler = new BattlesHandler(0, [this.clientBattle]);
        this.playerIndex = undefined;
        this.playerIndexIsSet = false;
        this.clientSocket = clientSocket;
        this.battleId = undefined;
        (this.started = false), (this.msgQueue = new AsyncQueue());
    }

    private async onMessage(data: WebSocket.Data): Promise<void> {
        const message: string = data.toString();
        console.log("Received:", message);

        if (message.startsWith("|challstr|")) {
            const parts = message.split("|");
            const challstr = parts[2] + "|" + parts[3];

            if (this.url.includes(localHost)) {
                this.sendMessage("", `/search ${formatId}`);
            } else {
                const assertion = await this.getAssertion(challstr);

                if (assertion) {
                    this.login(this.username, assertion);
                    this.sendMessage("", `/search ${formatId}`);
                }
            }
        } else if (message.startsWith(">")) {
            const turn = this.clientBattle.turn ?? 0;

            for (const line of message.split("\n")) {
                if (line.startsWith(">")) {
                    this.battleId = line.slice(1);
                }
                if (line.startsWith("|error")) {
                    console.error(line);
                }
                // if (isAction(line)) {
                if (this.started) {
                    this.handler.appendTurnLine(this.playerIndex ?? 0, 0, line);
                }
                try {
                    this.clientBattle.add(line);
                } catch (err) {}
                if (line.startsWith("|start")) {
                    this.started = true;

                    if (this.url.includes(online)) {
                        this.sendMessage(this.battleId ?? "", "/timer on");
                    }
                }
                if (line.startsWith("|win") || line.startsWith("|tie\n")) {
                    const state = this.handler.getState({
                        done: 1,
                        playerIndex: this.playerIndex ?? 0,
                    });
                    this.clientSocket.write(state);
                    const actionChar = await this.msgQueue.dequeue();
                    const action = actionCharToString(actionChar);
                    this.handler.reset();
                    for (const side of this.clientBattle.sides) {
                        side.reset();
                    }
                    this.clientBattle.reset();

                    this.sendMessage("", `/noreply /leave ${this.battleId}`);
                    this.sendMessage("", `/search ${formatId}`);
                }
            }
            if (message.includes("|request")) {
                this.clientBattle.update(); // optional, only relevant if stream contains |request|
                try {
                    if (!this.playerIndexIsSet) {
                        this.playerIndex =
                            parseInt(
                                (
                                    this.clientBattle.request as AnyObject
                                ).side.id.slice(1),
                            ) - 1;
                        this.playerIndexIsSet = true;
                        this.handler.playerIndex = this.playerIndex;
                    }
                } catch (err) {}
            }

            if (isActionRequired(this.clientBattle, message)) {
                const rqid = this.clientBattle.request?.rqid;
                logRequest(this.clientBattle.request ?? {});
                const state = this.handler.getState({
                    done: 0,
                    playerIndex: this.playerIndex ?? 0,
                });
                this.clientSocket.write(state);
                const actionChar = await this.msgQueue.dequeue();
                const action = actionCharToString(actionChar);
                this.sendMessage(
                    this.battleId ?? "",
                    `/choose ${action}|${rqid}`,
                );
            }
        }
    }
    private sendMessage(room: string, ...messageList: string[]) {
        const messageToSend = room + "|" + messageList.join("|");
        this.ws.send(messageToSend);
        console.log(`Sent: ${messageToSend}`);
    }

    private async getAssertion(challstr: string): Promise<string | null> {
        if (this.password !== undefined) {
            try {
                const params = {
                    name: this.username,
                    pass: this.password,
                    challstr: challstr,
                };

                const response = await fetch(
                    "https://play.pokemonshowdown.com/api/login",
                    { method: "POST", body: JSON.stringify(params) },
                );
                const assertion: string = await response.text();
                const json = JSON.parse(assertion.slice(1));
                return json.assertion;
            } catch (error) {
                console.error("Error getting assertion:", error);
                return null;
            }
        } else {
            try {
                const params = {
                    act: "getassertion",
                    userid: this.username,
                    challstr: challstr,
                };

                const response = await fetch(
                    "https://play.pokemonshowdown.com/api/login",
                    { method: "POST", body: JSON.stringify(params) },
                );
                const assertion: string = await response.text();

                if (assertion.slice(0, 2) === ";;") {
                    console.error("Error:", assertion);
                    return null;
                }

                return assertion;
            } catch (error) {
                console.error("Error getting assertion:", error);
                return null;
            }
        }
    }

    postMessage(actionChar: string) {
        this.msgQueue.enqueue(actionChar);
    }

    private login(username: string, assertion: string): void {
        this.ws.send("|/trn " + username + ",0," + assertion);
    }
}

let numConnections = 0;
const workers: PokemonShowdownBot[] = [];

function createWorker(socket: net.Socket) {
    const worker = new PokemonShowdownBot(socket, "jtwin", "trex123");
    workers.push(worker);
}

function processInput(input: string) {
    const [workIndex, playerIndex, actionChar] = input.split("|");
    const worker = workers[parseInt(workIndex)];
    worker.postMessage(actionChar);
}

const socketStates = new Map<net.Socket, InternalState>();
const decoder = new TextDecoder("utf-8");

const server = net.createServer((socket) => {
    console.log(`Client${numConnections} connected`);
    createWorker(socket);

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
