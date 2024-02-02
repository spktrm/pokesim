// package: state.v1
// file: state/v1/context.proto

import * as jspb from "google-protobuf";
import * as state_v1_enum_pb from "../../state/v1/enum_pb";

export class SideConditions extends jspb.Message {
    getAuroraveil(): number;
    setAuroraveil(value: number): void;

    getCraftyshield(): number;
    setCraftyshield(value: number): void;

    getLightscreen(): number;
    setLightscreen(value: number): void;

    getLuckychant(): number;
    setLuckychant(value: number): void;

    getMatblock(): number;
    setMatblock(value: number): void;

    getMist(): number;
    setMist(value: number): void;

    getQuickguard(): number;
    setQuickguard(value: number): void;

    getReflect(): number;
    setReflect(value: number): void;

    getSafeguard(): number;
    setSafeguard(value: number): void;

    getSpikes(): number;
    setSpikes(value: number): void;

    getStealthrock(): number;
    setStealthrock(value: number): void;

    getStickyweb(): number;
    setStickyweb(value: number): void;

    getTailwind(): number;
    setTailwind(value: number): void;

    getToxicspikes(): number;
    setToxicspikes(value: number): void;

    getWideguard(): number;
    setWideguard(value: number): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): SideConditions.AsObject;
    static toObject(
        includeInstance: boolean,
        msg: SideConditions,
    ): SideConditions.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: SideConditions,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): SideConditions;
    static deserializeBinaryFromReader(
        message: SideConditions,
        reader: jspb.BinaryReader,
    ): SideConditions;
}

export namespace SideConditions {
    export type AsObject = {
        auroraveil: number;
        craftyshield: number;
        lightscreen: number;
        luckychant: number;
        matblock: number;
        mist: number;
        quickguard: number;
        reflect: number;
        safeguard: number;
        spikes: number;
        stealthrock: number;
        stickyweb: number;
        tailwind: number;
        toxicspikes: number;
        wideguard: number;
    };
}

export class VolatileStatuses extends jspb.Message {
    getAttract(): number;
    setAttract(value: number): void;

    getBanefulbunker(): number;
    setBanefulbunker(value: number): void;

    getBeakblast(): number;
    setBeakblast(value: number): void;

    getBide(): number;
    setBide(value: number): void;

    getBounce(): number;
    setBounce(value: number): void;

    getBurningbulwark(): number;
    setBurningbulwark(value: number): void;

    getCharge(): number;
    setCharge(value: number): void;

    getChoicelock(): number;
    setChoicelock(value: number): void;

    getCommanded(): number;
    setCommanded(value: number): void;

    getCommanding(): number;
    setCommanding(value: number): void;

    getConfusion(): number;
    setConfusion(value: number): void;

    getCounter(): number;
    setCounter(value: number): void;

    getCudchew(): number;
    setCudchew(value: number): void;

    getCurse(): number;
    setCurse(value: number): void;

    getDefensecurl(): number;
    setDefensecurl(value: number): void;

    getDestinybond(): number;
    setDestinybond(value: number): void;

    getDetect(): number;
    setDetect(value: number): void;

    getDig(): number;
    setDig(value: number): void;

    getDisable(): number;
    setDisable(value: number): void;

    getDive(): number;
    setDive(value: number): void;

    getDragoncheer(): number;
    setDragoncheer(value: number): void;

    getDynamax(): number;
    setDynamax(value: number): void;

    getElectrify(): number;
    setElectrify(value: number): void;

    getEmbargo(): number;
    setEmbargo(value: number): void;

    getEncore(): number;
    setEncore(value: number): void;

    getEndure(): number;
    setEndure(value: number): void;

    getFlashfire(): number;
    setFlashfire(value: number): void;

    getFlinch(): number;
    setFlinch(value: number): void;

    getFling(): number;
    setFling(value: number): void;

    getFly(): number;
    setFly(value: number): void;

    getFocusenergy(): number;
    setFocusenergy(value: number): void;

    getFocuspunch(): number;
    setFocuspunch(value: number): void;

    getFollowme(): number;
    setFollowme(value: number): void;

    getForesight(): number;
    setForesight(value: number): void;

    getFurycutter(): number;
    setFurycutter(value: number): void;

    getGastroacid(): number;
    setGastroacid(value: number): void;

    getGem(): number;
    setGem(value: number): void;

    getGlaiverush(): number;
    setGlaiverush(value: number): void;

    getGrudge(): number;
    setGrudge(value: number): void;

    getHealblock(): number;
    setHealblock(value: number): void;

    getHelpinghand(): number;
    setHelpinghand(value: number): void;

    getIceball(): number;
    setIceball(value: number): void;

    getImprison(): number;
    setImprison(value: number): void;

    getIngrain(): number;
    setIngrain(value: number): void;

    getKingsshield(): number;
    setKingsshield(value: number): void;

    getLaserfocus(): number;
    setLaserfocus(value: number): void;

    getLeechseed(): number;
    setLeechseed(value: number): void;

    getLeppaberry(): number;
    setLeppaberry(value: number): void;

    getLockedmove(): number;
    setLockedmove(value: number): void;

    getLockon(): number;
    setLockon(value: number): void;

    getMagiccoat(): number;
    setMagiccoat(value: number): void;

    getMagnetrise(): number;
    setMagnetrise(value: number): void;

    getMaxguard(): number;
    setMaxguard(value: number): void;

    getMetronome(): number;
    setMetronome(value: number): void;

    getMicleberry(): number;
    setMicleberry(value: number): void;

    getMinimize(): number;
    setMinimize(value: number): void;

    getMiracleeye(): number;
    setMiracleeye(value: number): void;

    getMirrorcoat(): number;
    setMirrorcoat(value: number): void;

    getMustrecharge(): number;
    setMustrecharge(value: number): void;

    getNightmare(): number;
    setNightmare(value: number): void;

    getNoretreat(): number;
    setNoretreat(value: number): void;

    getObstruct(): number;
    setObstruct(value: number): void;

    getOctolock(): number;
    setOctolock(value: number): void;

    getPartiallytrapped(): number;
    setPartiallytrapped(value: number): void;

    getPartialtrappinglock(): number;
    setPartialtrappinglock(value: number): void;

    getPerishsong(): number;
    setPerishsong(value: number): void;

    getPhantomforce(): number;
    setPhantomforce(value: number): void;

    getPowder(): number;
    setPowder(value: number): void;

    getPowershift(): number;
    setPowershift(value: number): void;

    getPowertrick(): number;
    setPowertrick(value: number): void;

    getProtect(): number;
    setProtect(value: number): void;

    getProtosynthesis(): number;
    setProtosynthesis(value: number): void;

    getQuarkdrive(): number;
    setQuarkdrive(value: number): void;

    getRage(): number;
    setRage(value: number): void;

    getRagepowder(): number;
    setRagepowder(value: number): void;

    getRollout(): number;
    setRollout(value: number): void;

    getRolloutstorage(): number;
    setRolloutstorage(value: number): void;

    getRoost(): number;
    setRoost(value: number): void;

    getSaltcure(): number;
    setSaltcure(value: number): void;

    getShadowforce(): number;
    setShadowforce(value: number): void;

    getShelltrap(): number;
    setShelltrap(value: number): void;

    getSilktrap(): number;
    setSilktrap(value: number): void;

    getSkydrop(): number;
    setSkydrop(value: number): void;

    getSlowstart(): number;
    setSlowstart(value: number): void;

    getSmackdown(): number;
    setSmackdown(value: number): void;

    getSnatch(): number;
    setSnatch(value: number): void;

    getSparklingaria(): number;
    setSparklingaria(value: number): void;

    getSpikyshield(): number;
    setSpikyshield(value: number): void;

    getSpotlight(): number;
    setSpotlight(value: number): void;

    getStall(): number;
    setStall(value: number): void;

    getStockpile(): number;
    setStockpile(value: number): void;

    getSubstitute(): number;
    setSubstitute(value: number): void;

    getSubstitutebroken(): number;
    setSubstitutebroken(value: number): void;

    getSyrupbomb(): number;
    setSyrupbomb(value: number): void;

    getTarshot(): number;
    setTarshot(value: number): void;

    getTaunt(): number;
    setTaunt(value: number): void;

    getTelekinesis(): number;
    setTelekinesis(value: number): void;

    getThroatchop(): number;
    setThroatchop(value: number): void;

    getTorment(): number;
    setTorment(value: number): void;

    getTrapped(): number;
    setTrapped(value: number): void;

    getTruant(): number;
    setTruant(value: number): void;

    getTwoturnmove(): number;
    setTwoturnmove(value: number): void;

    getUnburden(): number;
    setUnburden(value: number): void;

    getUproar(): number;
    setUproar(value: number): void;

    getYawn(): number;
    setYawn(value: number): void;

    getZenmode(): number;
    setZenmode(value: number): void;

    getAquaring(): number;
    setAquaring(value: number): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): VolatileStatuses.AsObject;
    static toObject(
        includeInstance: boolean,
        msg: VolatileStatuses,
    ): VolatileStatuses.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: VolatileStatuses,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): VolatileStatuses;
    static deserializeBinaryFromReader(
        message: VolatileStatuses,
        reader: jspb.BinaryReader,
    ): VolatileStatuses;
}

export namespace VolatileStatuses {
    export type AsObject = {
        attract: number;
        banefulbunker: number;
        beakblast: number;
        bide: number;
        bounce: number;
        burningbulwark: number;
        charge: number;
        choicelock: number;
        commanded: number;
        commanding: number;
        confusion: number;
        counter: number;
        cudchew: number;
        curse: number;
        defensecurl: number;
        destinybond: number;
        detect: number;
        dig: number;
        disable: number;
        dive: number;
        dragoncheer: number;
        dynamax: number;
        electrify: number;
        embargo: number;
        encore: number;
        endure: number;
        flashfire: number;
        flinch: number;
        fling: number;
        fly: number;
        focusenergy: number;
        focuspunch: number;
        followme: number;
        foresight: number;
        furycutter: number;
        gastroacid: number;
        gem: number;
        glaiverush: number;
        grudge: number;
        healblock: number;
        helpinghand: number;
        iceball: number;
        imprison: number;
        ingrain: number;
        kingsshield: number;
        laserfocus: number;
        leechseed: number;
        leppaberry: number;
        lockedmove: number;
        lockon: number;
        magiccoat: number;
        magnetrise: number;
        maxguard: number;
        metronome: number;
        micleberry: number;
        minimize: number;
        miracleeye: number;
        mirrorcoat: number;
        mustrecharge: number;
        nightmare: number;
        noretreat: number;
        obstruct: number;
        octolock: number;
        partiallytrapped: number;
        partialtrappinglock: number;
        perishsong: number;
        phantomforce: number;
        powder: number;
        powershift: number;
        powertrick: number;
        protect: number;
        protosynthesis: number;
        quarkdrive: number;
        rage: number;
        ragepowder: number;
        rollout: number;
        rolloutstorage: number;
        roost: number;
        saltcure: number;
        shadowforce: number;
        shelltrap: number;
        silktrap: number;
        skydrop: number;
        slowstart: number;
        smackdown: number;
        snatch: number;
        sparklingaria: number;
        spikyshield: number;
        spotlight: number;
        stall: number;
        stockpile: number;
        substitute: number;
        substitutebroken: number;
        syrupbomb: number;
        tarshot: number;
        taunt: number;
        telekinesis: number;
        throatchop: number;
        torment: number;
        trapped: number;
        truant: number;
        twoturnmove: number;
        unburden: number;
        uproar: number;
        yawn: number;
        zenmode: number;
        aquaring: number;
    };
}

export class Boosts extends jspb.Message {
    getAtk(): number;
    setAtk(value: number): void;

    getDef(): number;
    setDef(value: number): void;

    getSpa(): number;
    setSpa(value: number): void;

    getSpd(): number;
    setSpd(value: number): void;

    getSpe(): number;
    setSpe(value: number): void;

    getAccuracy(): number;
    setAccuracy(value: number): void;

    getEvasion(): number;
    setEvasion(value: number): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): Boosts.AsObject;
    static toObject(includeInstance: boolean, msg: Boosts): Boosts.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: Boosts,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): Boosts;
    static deserializeBinaryFromReader(
        message: Boosts,
        reader: jspb.BinaryReader,
    ): Boosts;
}

export namespace Boosts {
    export type AsObject = {
        atk: number;
        def: number;
        spa: number;
        spd: number;
        spe: number;
        accuracy: number;
        evasion: number;
    };
}

export class Pseudoweathers extends jspb.Message {
    getFairylock(): boolean;
    setFairylock(value: boolean): void;

    getGravity(): boolean;
    setGravity(value: boolean): void;

    getIondeluge(): boolean;
    setIondeluge(value: boolean): void;

    getMagicroom(): boolean;
    setMagicroom(value: boolean): void;

    getMudsport(): boolean;
    setMudsport(value: boolean): void;

    getTrickroom(): boolean;
    setTrickroom(value: boolean): void;

    getWatersport(): boolean;
    setWatersport(value: boolean): void;

    getWonderroom(): boolean;
    setWonderroom(value: boolean): void;

    getFairylockMinDur(): number;
    setFairylockMinDur(value: number): void;

    getGravityMinDur(): number;
    setGravityMinDur(value: number): void;

    getIondelugeMinDur(): number;
    setIondelugeMinDur(value: number): void;

    getMagicroomMinDur(): number;
    setMagicroomMinDur(value: number): void;

    getMudsportMinDur(): number;
    setMudsportMinDur(value: number): void;

    getTrickroomMinDur(): number;
    setTrickroomMinDur(value: number): void;

    getWatersportMinDur(): number;
    setWatersportMinDur(value: number): void;

    getWonderroomMinDur(): number;
    setWonderroomMinDur(value: number): void;

    getFairylockMaxDur(): number;
    setFairylockMaxDur(value: number): void;

    getGravityMaxDur(): number;
    setGravityMaxDur(value: number): void;

    getIondelugeMaxDur(): number;
    setIondelugeMaxDur(value: number): void;

    getMagicroomMaxDur(): number;
    setMagicroomMaxDur(value: number): void;

    getMudsportMaxDur(): number;
    setMudsportMaxDur(value: number): void;

    getTrickroomMaxDur(): number;
    setTrickroomMaxDur(value: number): void;

    getWatersportMaxDur(): number;
    setWatersportMaxDur(value: number): void;

    getWonderroomMaxDur(): number;
    setWonderroomMaxDur(value: number): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): Pseudoweathers.AsObject;
    static toObject(
        includeInstance: boolean,
        msg: Pseudoweathers,
    ): Pseudoweathers.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: Pseudoweathers,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): Pseudoweathers;
    static deserializeBinaryFromReader(
        message: Pseudoweathers,
        reader: jspb.BinaryReader,
    ): Pseudoweathers;
}

export namespace Pseudoweathers {
    export type AsObject = {
        fairylock: boolean;
        gravity: boolean;
        iondeluge: boolean;
        magicroom: boolean;
        mudsport: boolean;
        trickroom: boolean;
        watersport: boolean;
        wonderroom: boolean;
        fairylockMinDur: number;
        gravityMinDur: number;
        iondelugeMinDur: number;
        magicroomMinDur: number;
        mudsportMinDur: number;
        trickroomMinDur: number;
        watersportMinDur: number;
        wonderroomMinDur: number;
        fairylockMaxDur: number;
        gravityMaxDur: number;
        iondelugeMaxDur: number;
        magicroomMaxDur: number;
        mudsportMaxDur: number;
        trickroomMaxDur: number;
        watersportMaxDur: number;
        wonderroomMaxDur: number;
    };
}

export class Weather extends jspb.Message {
    getWeather(): state_v1_enum_pb.WeathersMap[keyof state_v1_enum_pb.WeathersMap];
    setWeather(
        value: state_v1_enum_pb.WeathersMap[keyof state_v1_enum_pb.WeathersMap],
    ): void;

    getMinDur(): number;
    setMinDur(value: number): void;

    getMaxDur(): number;
    setMaxDur(value: number): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): Weather.AsObject;
    static toObject(includeInstance: boolean, msg: Weather): Weather.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: Weather,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): Weather;
    static deserializeBinaryFromReader(
        message: Weather,
        reader: jspb.BinaryReader,
    ): Weather;
}

export namespace Weather {
    export type AsObject = {
        weather: state_v1_enum_pb.WeathersMap[keyof state_v1_enum_pb.WeathersMap];
        minDur: number;
        maxDur: number;
    };
}

export class Terrains extends jspb.Message {
    getTerrain(): state_v1_enum_pb.TerrainMap[keyof state_v1_enum_pb.TerrainMap];
    setTerrain(
        value: state_v1_enum_pb.TerrainMap[keyof state_v1_enum_pb.TerrainMap],
    ): void;

    getMinDur(): number;
    setMinDur(value: number): void;

    getMaxDur(): number;
    setMaxDur(value: number): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): Terrains.AsObject;
    static toObject(includeInstance: boolean, msg: Terrains): Terrains.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: Terrains,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): Terrains;
    static deserializeBinaryFromReader(
        message: Terrains,
        reader: jspb.BinaryReader,
    ): Terrains;
}

export namespace Terrains {
    export type AsObject = {
        terrain: state_v1_enum_pb.TerrainMap[keyof state_v1_enum_pb.TerrainMap];
        minDur: number;
        maxDur: number;
    };
}

export class Field extends jspb.Message {
    hasPseudoweather(): boolean;
    clearPseudoweather(): void;
    getPseudoweather(): Pseudoweathers | undefined;
    setPseudoweather(value?: Pseudoweathers): void;

    hasWeather(): boolean;
    clearWeather(): void;
    getWeather(): Weather | undefined;
    setWeather(value?: Weather): void;

    hasTerrain(): boolean;
    clearTerrain(): void;
    getTerrain(): Terrains | undefined;
    setTerrain(value?: Terrains): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): Field.AsObject;
    static toObject(includeInstance: boolean, msg: Field): Field.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: Field,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): Field;
    static deserializeBinaryFromReader(
        message: Field,
        reader: jspb.BinaryReader,
    ): Field;
}

export namespace Field {
    export type AsObject = {
        pseudoweather?: Pseudoweathers.AsObject;
        weather?: Weather.AsObject;
        terrain?: Terrains.AsObject;
    };
}

export class Context extends jspb.Message {
    hasSideConditions1(): boolean;
    clearSideConditions1(): void;
    getSideConditions1(): SideConditions | undefined;
    setSideConditions1(value?: SideConditions): void;

    hasSideConditions2(): boolean;
    clearSideConditions2(): void;
    getSideConditions2(): SideConditions | undefined;
    setSideConditions2(value?: SideConditions): void;

    hasVolatileStatus1(): boolean;
    clearVolatileStatus1(): void;
    getVolatileStatus1(): VolatileStatuses | undefined;
    setVolatileStatus1(value?: VolatileStatuses): void;

    hasVolatileStatus2(): boolean;
    clearVolatileStatus2(): void;
    getVolatileStatus2(): VolatileStatuses | undefined;
    setVolatileStatus2(value?: VolatileStatuses): void;

    hasBoosts1(): boolean;
    clearBoosts1(): void;
    getBoosts1(): Boosts | undefined;
    setBoosts1(value?: Boosts): void;

    hasBoosts2(): boolean;
    clearBoosts2(): void;
    getBoosts2(): Boosts | undefined;
    setBoosts2(value?: Boosts): void;

    hasField(): boolean;
    clearField(): void;
    getField(): Field | undefined;
    setField(value?: Field): void;

    serializeBinary(): Uint8Array;
    toObject(includeInstance?: boolean): Context.AsObject;
    static toObject(includeInstance: boolean, msg: Context): Context.AsObject;
    static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
    static extensionsBinary: {
        [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
    };
    static serializeBinaryToWriter(
        message: Context,
        writer: jspb.BinaryWriter,
    ): void;
    static deserializeBinary(bytes: Uint8Array): Context;
    static deserializeBinaryFromReader(
        message: Context,
        reader: jspb.BinaryReader,
    ): Context;
}

export namespace Context {
    export type AsObject = {
        sideConditions1?: SideConditions.AsObject;
        sideConditions2?: SideConditions.AsObject;
        volatileStatus1?: VolatileStatuses.AsObject;
        volatileStatus2?: VolatileStatuses.AsObject;
        boosts1?: Boosts.AsObject;
        boosts2?: Boosts.AsObject;
        field?: Field.AsObject;
    };
}
