// package: state.v1
// file: state/v1/state.proto

import * as jspb from "google-protobuf";
import * as state_v1_game_pb from "../../state/v1/game_pb";
import * as state_v1_pokemon_pb from "../../state/v1/pokemon_pb";
import * as state_v1_context_pb from "../../state/v1/context_pb";

export class LegalMask extends jspb.Message {
  getMove1(): boolean;
  setMove1(value: boolean): void;

  getMove2(): boolean;
  setMove2(value: boolean): void;

  getMove3(): boolean;
  setMove3(value: boolean): void;

  getMove4(): boolean;
  setMove4(value: boolean): void;

  getSwitch1(): boolean;
  setSwitch1(value: boolean): void;

  getSwitch2(): boolean;
  setSwitch2(value: boolean): void;

  getSwitch3(): boolean;
  setSwitch3(value: boolean): void;

  getSwitch4(): boolean;
  setSwitch4(value: boolean): void;

  getSwitch5(): boolean;
  setSwitch5(value: boolean): void;

  getSwitch6(): boolean;
  setSwitch6(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): LegalMask.AsObject;
  static toObject(includeInstance: boolean, msg: LegalMask): LegalMask.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: LegalMask, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): LegalMask;
  static deserializeBinaryFromReader(message: LegalMask, reader: jspb.BinaryReader): LegalMask;
}

export namespace LegalMask {
  export type AsObject = {
    move1: boolean,
    move2: boolean,
    move3: boolean,
    move4: boolean,
    switch1: boolean,
    switch2: boolean,
    switch3: boolean,
    switch4: boolean,
    switch5: boolean,
    switch6: boolean,
  }
}

export class State extends jspb.Message {
  hasGame(): boolean;
  clearGame(): void;
  getGame(): state_v1_game_pb.Game | undefined;
  setGame(value?: state_v1_game_pb.Game): void;

  hasPrivateTeam(): boolean;
  clearPrivateTeam(): void;
  getPrivateTeam(): state_v1_pokemon_pb.Team | undefined;
  setPrivateTeam(value?: state_v1_pokemon_pb.Team): void;

  hasPublicTeam1(): boolean;
  clearPublicTeam1(): void;
  getPublicTeam1(): state_v1_pokemon_pb.Team | undefined;
  setPublicTeam1(value?: state_v1_pokemon_pb.Team): void;

  hasPublicTeam2(): boolean;
  clearPublicTeam2(): void;
  getPublicTeam2(): state_v1_pokemon_pb.Team | undefined;
  setPublicTeam2(value?: state_v1_pokemon_pb.Team): void;

  hasContext(): boolean;
  clearContext(): void;
  getContext(): state_v1_context_pb.Context | undefined;
  setContext(value?: state_v1_context_pb.Context): void;

  hasLegalMask(): boolean;
  clearLegalMask(): void;
  getLegalMask(): LegalMask | undefined;
  setLegalMask(value?: LegalMask): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): State.AsObject;
  static toObject(includeInstance: boolean, msg: State): State.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: State, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): State;
  static deserializeBinaryFromReader(message: State, reader: jspb.BinaryReader): State;
}

export namespace State {
  export type AsObject = {
    game?: state_v1_game_pb.Game.AsObject,
    privateTeam?: state_v1_pokemon_pb.Team.AsObject,
    publicTeam1?: state_v1_pokemon_pb.Team.AsObject,
    publicTeam2?: state_v1_pokemon_pb.Team.AsObject,
    context?: state_v1_context_pb.Context.AsObject,
    legalMask?: LegalMask.AsObject,
  }
}

