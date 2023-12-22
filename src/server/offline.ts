import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";

const battle = new Battle(new Generations(Dex));

// import { AsyncResource } from "node:async_hooks";
// import { EventEmitter } from "node:events";
// import { Worker as NodeWorker } from "node:worker_threads";

// const kTaskInfo = Symbol("kTaskInfo");
// const kWorkerFreedEvent = Symbol("kWorkerFreedEvent");

// class WorkerPoolTaskInfo extends AsyncResource {
//     callback: (err: Error | null, result?: any) => void;

//     constructor(callback: (err: Error | null, result?: any) => void) {
//         super("WorkerPoolTaskInfo");
//         this.callback = callback;
//     }

//     done(err: Error | null, result?: any) {
//         this.runInAsyncScope(this.callback, null, err, result);
//         this.emitDestroy(); // `TaskInfo`s are used only once.
//     }
// }

// // Extend the Worker class to include the kTaskInfo property
// class Worker extends NodeWorker {
//     [kTaskInfo]?: WorkerPoolTaskInfo;
// }

// export default class WorkerPool extends EventEmitter {
//     numThreads: number;
//     workers: Worker[];
//     freeWorkers: Worker[];
//     tasks: { task: any; callback: (err: Error | null, result?: any) => void }[];

//     constructor(numThreads: number) {
//         super();
//         this.numThreads = numThreads;
//         this.workers = [];
//         this.freeWorkers = [];
//         this.tasks = [];

//         for (let i = 0; i < numThreads; i++) {
//             this.addNewWorker();
//         }

//         this.on(kWorkerFreedEvent, () => {
//             if (this.tasks.length > 0) {
//                 const { task, callback } = this.tasks.shift()!;
//                 this.runTask(task, callback);
//             }
//         });
//     }

//     addNewWorker() {
//         const worker = new Worker(
//             new URL("task_processor.js", import.meta.url)
//         );
//         worker.on("message", (result: any) => {
//             worker[kTaskInfo]!.done(null, result);
//             worker[kTaskInfo] = undefined;
//             this.freeWorkers.push(worker);
//             this.emit(kWorkerFreedEvent);
//         });
//         worker.on("error", (err: Error) => {
//             if (worker[kTaskInfo]) worker[kTaskInfo].done(err, null);
//             else this.emit("error", err);

//             this.workers.splice(this.workers.indexOf(worker), 1);
//             this.addNewWorker();
//         });
//         this.workers.push(worker);
//         this.freeWorkers.push(worker);
//         this.emit(kWorkerFreedEvent);
//     }

//     runTask(task: any, callback: (err: Error | null, result?: any) => void) {
//         if (this.freeWorkers.length === 0) {
//             this.tasks.push({ task, callback });
//             return;
//         }

//         const worker = this.freeWorkers.pop()!;
//         worker[kTaskInfo] = new WorkerPoolTaskInfo(callback);
//         worker.postMessage(task);
//     }

//     close() {
//         for (const worker of this.workers) worker.terminate();
//     }
// }
