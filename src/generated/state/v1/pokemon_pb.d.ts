// package: state.v1
// file: state/v1/pokemon.proto

import * as jspb from "google-protobuf";
import * as state_v1_enum_pb from "../../state/v1/enum_pb";

export class Pokemon extends jspb.Message {
    getSpecies(): state_v1_enum_pb.SpeciesMap[keyof state_v1_enum_pb.SpeciesMap];
    setSpecies(
        value: state_v1_enum_pb.SpeciesMap[keyof state_v1_enum_pb.SpeciesMap],
    ): void;

    getItem(): state_v1_enum_pb.ItemsMap[keyof state_v1_enum_pb.ItemsMap];
    setItem(
        value: state_v1_enum_pb.ItemsMap[keyof state_v1_enum_pb.ItemsMap],
    ): void;

    getAbility(): state_v1_enum_pb.AbilitiesMap[keyof state_v1_enum_pb.AbilitiesMap];
    setAbility(
        value: state_v1_enum_pb.AbilitiesMap[keyof state_v1_enum_pb.AbilitiesMap],
    ): void;

    getHp(): number;
    setHp(value: number): void;

    getActive(): boolean;
    setActive(value: boolean): void;

    getFainted(): boolean;
    setFainted(value: boolean): void;

    getStatus(): number;
    setStatus(value: number): void;

    getLastMove(): number;
    setLastMove(value: number): void;

    getPublic(): boolean;
    setPublic(value: boolean): void;

    getSide(): boolean;
    setSide(value: boolean): void;

    getSleepTurns(): number;
    setSleepTurns(value: number): void;

    getToxicTurns(): number;
    setToxicTurns(value: number): void;

    getMove1Ppleft(): number;
    setMove1Ppleft(value: number): void;

    getMove2Ppleft(): number;
    setMove2Ppleft(value: number): void;

    getMove3Ppleft(): number;
    setMove3Ppleft(value: number): void;

    getMove4Ppleft(): number;
    setMove4Ppleft(value: number): void;

    getMove1Ppmax(): number;
    setMove1Ppmax(value: number): void;

    getMove2Ppmax(): number;
    setMove2Ppmax(value: number): void;

    getMove3Ppmax(): number;
    setMove3Ppmax(value: number): void;

    getMove4Ppmax(): number;
    setMove4Ppmax(value: number): void;

    getMove1(): state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap];
    setMove1(
        value: state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap],
    ): void;

    getMove2(): state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap];
    setMove2(
        value: state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap],
    ): void;

    getMove3(): state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap];
    setMove3(
        value: state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap],
    ): void;

    getMove4(): state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap];
    setMove4(
        value: state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap],
    ): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): Pokemon.AsObject;
    static toObject(includeInstance: boolean, msg: Pokemon): Pokemon.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: Pokemon,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): Pokemon;
    static deserializeBinaryFromReader(
        message: Pokemon,
        reader: jspb.BinaryReader,
    ): Pokemon;
}

export namespace Pokemon {
    export type AsObject = {
        species: state_v1_enum_pb.SpeciesMap[keyof state_v1_enum_pb.SpeciesMap];
        item: state_v1_enum_pb.ItemsMap[keyof state_v1_enum_pb.ItemsMap];
        ability: state_v1_enum_pb.AbilitiesMap[keyof state_v1_enum_pb.AbilitiesMap];
        hp: number;
        active: boolean;
        fainted: boolean;
        status: number;
        lastMove: number;
        pb_public: boolean;
        side: boolean;
        sleepTurns: number;
        toxicTurns: number;
        move1Ppleft: number;
        move2Ppleft: number;
        move3Ppleft: number;
        move4Ppleft: number;
        move1Ppmax: number;
        move2Ppmax: number;
        move3Ppmax: number;
        move4Ppmax: number;
        move1: state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap];
        move2: state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap];
        move3: state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap];
        move4: state_v1_enum_pb.MovesMap[keyof state_v1_enum_pb.MovesMap];
    };
}

export class Team extends jspb.Message {
    hasPokemon1(): boolean;
    clearPokemon1(): void;
    getPokemon1(): Pokemon | undefined;
    setPokemon1(value?: Pokemon): void;

    hasPokemon2(): boolean;
    clearPokemon2(): void;
    getPokemon2(): Pokemon | undefined;
    setPokemon2(value?: Pokemon): void;

    hasPokemon3(): boolean;
    clearPokemon3(): void;
    getPokemon3(): Pokemon | undefined;
    setPokemon3(value?: Pokemon): void;

    hasPokemon4(): boolean;
    clearPokemon4(): void;
    getPokemon4(): Pokemon | undefined;
    setPokemon4(value?: Pokemon): void;

    hasPokemon5(): boolean;
    clearPokemon5(): void;
    getPokemon5(): Pokemon | undefined;
    setPokemon5(value?: Pokemon): void;

    hasPokemon6(): boolean;
    clearPokemon6(): void;
    getPokemon6(): Pokemon | undefined;
    setPokemon6(value?: Pokemon): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): Team.AsObject;
    static toObject(includeInstance: boolean, msg: Team): Team.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: Team,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): Team;
    static deserializeBinaryFromReader(
        message: Team,
        reader: jspb.BinaryReader,
    ): Team;
}

export namespace Team {
    export type AsObject = {
        pokemon1?: Pokemon.AsObject;
        pokemon2?: Pokemon.AsObject;
        pokemon3?: Pokemon.AsObject;
        pokemon4?: Pokemon.AsObject;
        pokemon5?: Pokemon.AsObject;
        pokemon6?: Pokemon.AsObject;
    };
}
