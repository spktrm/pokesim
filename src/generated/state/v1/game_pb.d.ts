// package: state.v1
// file: state/v1/game.proto

import * as jspb from "google-protobuf";

export class Game extends jspb.Message {
  getWorkerIndex(): number;
  setWorkerIndex(value: number): void;

  getPlayerIndex(): boolean;
  setPlayerIndex(value: boolean): void;

  getDone(): boolean;
  setDone(value: boolean): void;

  getReward(): number;
  setReward(value: number): void;

  getTurn(): number;
  setTurn(value: number): void;

  getHeuristicAction(): number;
  setHeuristicAction(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Game.AsObject;
  static toObject(includeInstance: boolean, msg: Game): Game.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Game, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Game;
  static deserializeBinaryFromReader(message: Game, reader: jspb.BinaryReader): Game;
}

export namespace Game {
  export type AsObject = {
    workerIndex: number,
    playerIndex: boolean,
    done: boolean,
    reward: number,
    turn: number,
    heuristicAction: number,
  }
}

