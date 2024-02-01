// package: state.v1
// file: state/v1/game.proto

import * as jspb from "google-protobuf";

export class Game extends jspb.Message {
  getFirstName(): string;
  setFirstName(value: string): void;

  getLastName(): string;
  setLastName(value: string): void;

  getEmail(): string;
  setEmail(value: string): void;

  getPhoneNumber(): string;
  setPhoneNumber(value: string): void;

  getIsBlocked(): boolean;
  setIsBlocked(value: boolean): void;

  getIsFavorite(): boolean;
  setIsFavorite(value: boolean): void;

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
    firstName: string,
    lastName: string,
    email: string,
    phoneNumber: string,
    isBlocked: boolean,
    isFavorite: boolean,
  }
}

