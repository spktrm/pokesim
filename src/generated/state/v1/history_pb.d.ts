// package: state.v1
// file: state/v1/history.proto

import * as jspb from "google-protobuf";
import * as state_v1_pokemon_pb from "../../state/v1/pokemon_pb";
import * as state_v1_context_pb from "../../state/v1/context_pb";

export class Stats extends jspb.Message {
  getOrder(): number;
  setOrder(value: number): void;

  getIsCritical(): boolean;
  setIsCritical(value: boolean): void;

  getDamage(): number;
  setDamage(value: number): void;

  getEffectiveness(): number;
  setEffectiveness(value: number): void;

  getMissed(): boolean;
  setMissed(value: boolean): void;

  getMove(): number;
  setMove(value: number): void;

  getTargetFainted(): boolean;
  setTargetFainted(value: boolean): void;

  getMoveCounter(): number;
  setMoveCounter(value: number): void;

  getSwitchCounter(): number;
  setSwitchCounter(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Stats.AsObject;
  static toObject(includeInstance: boolean, msg: Stats): Stats.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Stats, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Stats;
  static deserializeBinaryFromReader(message: Stats, reader: jspb.BinaryReader): Stats;
}

export namespace Stats {
  export type AsObject = {
    order: number,
    isCritical: boolean,
    damage: number,
    effectiveness: number,
    missed: boolean,
    move: number,
    targetFainted: boolean,
    moveCounter: number,
    switchCounter: number,
  }
}

export class HistoryItem extends jspb.Message {
  hasUser(): boolean;
  clearUser(): void;
  getUser(): state_v1_pokemon_pb.Pokemon | undefined;
  setUser(value?: state_v1_pokemon_pb.Pokemon): void;

  hasTarget(): boolean;
  clearTarget(): void;
  getTarget(): state_v1_pokemon_pb.Pokemon | undefined;
  setTarget(value?: state_v1_pokemon_pb.Pokemon): void;

  hasContext(): boolean;
  clearContext(): void;
  getContext(): state_v1_context_pb.Context | undefined;
  setContext(value?: state_v1_context_pb.Context): void;

  hasStats(): boolean;
  clearStats(): void;
  getStats(): Stats | undefined;
  setStats(value?: Stats): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): HistoryItem.AsObject;
  static toObject(includeInstance: boolean, msg: HistoryItem): HistoryItem.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: HistoryItem, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): HistoryItem;
  static deserializeBinaryFromReader(message: HistoryItem, reader: jspb.BinaryReader): HistoryItem;
}

export namespace HistoryItem {
  export type AsObject = {
    user?: state_v1_pokemon_pb.Pokemon.AsObject,
    target?: state_v1_pokemon_pb.Pokemon.AsObject,
    context?: state_v1_context_pb.Context.AsObject,
    stats?: Stats.AsObject,
  }
}

export class History extends jspb.Message {
  clearHistoryItemsList(): void;
  getHistoryItemsList(): Array<HistoryItem>;
  setHistoryItemsList(value: Array<HistoryItem>): void;
  addHistoryItems(value?: HistoryItem, index?: number): HistoryItem;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): History.AsObject;
  static toObject(includeInstance: boolean, msg: History): History.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: History, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): History;
  static deserializeBinaryFromReader(message: History, reader: jspb.BinaryReader): History;
}

export namespace History {
  export type AsObject = {
    historyItemsList: Array<HistoryItem.AsObject>,
  }
}

