// package: state.v1
// file: state/v1/context.proto

import * as jspb from "google-protobuf";
import * as state_v1_enum_pb from "../../state/v1/enum_pb";

export class SideConditions extends jspb.Message {
  getAuroraveil(): boolean;
  setAuroraveil(value: boolean): void;

  getCraftyshield(): boolean;
  setCraftyshield(value: boolean): void;

  getLightscreen(): boolean;
  setLightscreen(value: boolean): void;

  getLuckychant(): boolean;
  setLuckychant(value: boolean): void;

  getMatblock(): boolean;
  setMatblock(value: boolean): void;

  getMist(): boolean;
  setMist(value: boolean): void;

  getQuickguard(): boolean;
  setQuickguard(value: boolean): void;

  getReflect(): boolean;
  setReflect(value: boolean): void;

  getSafeguard(): boolean;
  setSafeguard(value: boolean): void;

  getSpikes(): boolean;
  setSpikes(value: boolean): void;

  getStealthrock(): boolean;
  setStealthrock(value: boolean): void;

  getStickyweb(): boolean;
  setStickyweb(value: boolean): void;

  getTailwind(): boolean;
  setTailwind(value: boolean): void;

  getToxicspikes(): boolean;
  setToxicspikes(value: boolean): void;

  getWideguard(): boolean;
  setWideguard(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): SideConditions.AsObject;
  static toObject(includeInstance: boolean, msg: SideConditions): SideConditions.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: SideConditions, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): SideConditions;
  static deserializeBinaryFromReader(message: SideConditions, reader: jspb.BinaryReader): SideConditions;
}

export namespace SideConditions {
  export type AsObject = {
    auroraveil: boolean,
    craftyshield: boolean,
    lightscreen: boolean,
    luckychant: boolean,
    matblock: boolean,
    mist: boolean,
    quickguard: boolean,
    reflect: boolean,
    safeguard: boolean,
    spikes: boolean,
    stealthrock: boolean,
    stickyweb: boolean,
    tailwind: boolean,
    toxicspikes: boolean,
    wideguard: boolean,
  }
}

export class VolatileStatuses extends jspb.Message {
  getAttract(): boolean;
  setAttract(value: boolean): void;

  getBanefulbunker(): boolean;
  setBanefulbunker(value: boolean): void;

  getBeakblast(): boolean;
  setBeakblast(value: boolean): void;

  getBide(): boolean;
  setBide(value: boolean): void;

  getBounce(): boolean;
  setBounce(value: boolean): void;

  getBurningbulwark(): boolean;
  setBurningbulwark(value: boolean): void;

  getCharge(): boolean;
  setCharge(value: boolean): void;

  getChoicelock(): boolean;
  setChoicelock(value: boolean): void;

  getCommanded(): boolean;
  setCommanded(value: boolean): void;

  getCommanding(): boolean;
  setCommanding(value: boolean): void;

  getConfusion(): boolean;
  setConfusion(value: boolean): void;

  getCounter(): boolean;
  setCounter(value: boolean): void;

  getCudchew(): boolean;
  setCudchew(value: boolean): void;

  getCurse(): boolean;
  setCurse(value: boolean): void;

  getDefensecurl(): boolean;
  setDefensecurl(value: boolean): void;

  getDestinybond(): boolean;
  setDestinybond(value: boolean): void;

  getDetect(): boolean;
  setDetect(value: boolean): void;

  getDig(): boolean;
  setDig(value: boolean): void;

  getDisable(): boolean;
  setDisable(value: boolean): void;

  getDive(): boolean;
  setDive(value: boolean): void;

  getDragoncheer(): boolean;
  setDragoncheer(value: boolean): void;

  getDynamax(): boolean;
  setDynamax(value: boolean): void;

  getElectrify(): boolean;
  setElectrify(value: boolean): void;

  getEmbargo(): boolean;
  setEmbargo(value: boolean): void;

  getEncore(): boolean;
  setEncore(value: boolean): void;

  getEndure(): boolean;
  setEndure(value: boolean): void;

  getFlashfire(): boolean;
  setFlashfire(value: boolean): void;

  getFlinch(): boolean;
  setFlinch(value: boolean): void;

  getFling(): boolean;
  setFling(value: boolean): void;

  getFly(): boolean;
  setFly(value: boolean): void;

  getFocusenergy(): boolean;
  setFocusenergy(value: boolean): void;

  getFocuspunch(): boolean;
  setFocuspunch(value: boolean): void;

  getFollowme(): boolean;
  setFollowme(value: boolean): void;

  getForesight(): boolean;
  setForesight(value: boolean): void;

  getFurycutter(): boolean;
  setFurycutter(value: boolean): void;

  getGastroacid(): boolean;
  setGastroacid(value: boolean): void;

  getGem(): boolean;
  setGem(value: boolean): void;

  getGlaiverush(): boolean;
  setGlaiverush(value: boolean): void;

  getGrudge(): boolean;
  setGrudge(value: boolean): void;

  getHealblock(): boolean;
  setHealblock(value: boolean): void;

  getHelpinghand(): boolean;
  setHelpinghand(value: boolean): void;

  getIceball(): boolean;
  setIceball(value: boolean): void;

  getImprison(): boolean;
  setImprison(value: boolean): void;

  getIngrain(): boolean;
  setIngrain(value: boolean): void;

  getKingsshield(): boolean;
  setKingsshield(value: boolean): void;

  getLaserfocus(): boolean;
  setLaserfocus(value: boolean): void;

  getLeechseed(): boolean;
  setLeechseed(value: boolean): void;

  getLeppaberry(): boolean;
  setLeppaberry(value: boolean): void;

  getLockedmove(): boolean;
  setLockedmove(value: boolean): void;

  getLockon(): boolean;
  setLockon(value: boolean): void;

  getMagiccoat(): boolean;
  setMagiccoat(value: boolean): void;

  getMagnetrise(): boolean;
  setMagnetrise(value: boolean): void;

  getMaxguard(): boolean;
  setMaxguard(value: boolean): void;

  getMetronome(): boolean;
  setMetronome(value: boolean): void;

  getMicleberry(): boolean;
  setMicleberry(value: boolean): void;

  getMinimize(): boolean;
  setMinimize(value: boolean): void;

  getMiracleeye(): boolean;
  setMiracleeye(value: boolean): void;

  getMirrorcoat(): boolean;
  setMirrorcoat(value: boolean): void;

  getMustrecharge(): boolean;
  setMustrecharge(value: boolean): void;

  getNightmare(): boolean;
  setNightmare(value: boolean): void;

  getNoretreat(): boolean;
  setNoretreat(value: boolean): void;

  getObstruct(): boolean;
  setObstruct(value: boolean): void;

  getOctolock(): boolean;
  setOctolock(value: boolean): void;

  getPartiallytrapped(): boolean;
  setPartiallytrapped(value: boolean): void;

  getPartialtrappinglock(): boolean;
  setPartialtrappinglock(value: boolean): void;

  getPerishsong(): boolean;
  setPerishsong(value: boolean): void;

  getPhantomforce(): boolean;
  setPhantomforce(value: boolean): void;

  getPowder(): boolean;
  setPowder(value: boolean): void;

  getPowershift(): boolean;
  setPowershift(value: boolean): void;

  getPowertrick(): boolean;
  setPowertrick(value: boolean): void;

  getProtect(): boolean;
  setProtect(value: boolean): void;

  getProtosynthesis(): boolean;
  setProtosynthesis(value: boolean): void;

  getQuarkdrive(): boolean;
  setQuarkdrive(value: boolean): void;

  getRage(): boolean;
  setRage(value: boolean): void;

  getRagepowder(): boolean;
  setRagepowder(value: boolean): void;

  getRollout(): boolean;
  setRollout(value: boolean): void;

  getRolloutstorage(): boolean;
  setRolloutstorage(value: boolean): void;

  getRoost(): boolean;
  setRoost(value: boolean): void;

  getSaltcure(): boolean;
  setSaltcure(value: boolean): void;

  getShadowforce(): boolean;
  setShadowforce(value: boolean): void;

  getShelltrap(): boolean;
  setShelltrap(value: boolean): void;

  getSilktrap(): boolean;
  setSilktrap(value: boolean): void;

  getSkydrop(): boolean;
  setSkydrop(value: boolean): void;

  getSlowstart(): boolean;
  setSlowstart(value: boolean): void;

  getSmackdown(): boolean;
  setSmackdown(value: boolean): void;

  getSnatch(): boolean;
  setSnatch(value: boolean): void;

  getSparklingaria(): boolean;
  setSparklingaria(value: boolean): void;

  getSpikyshield(): boolean;
  setSpikyshield(value: boolean): void;

  getSpotlight(): boolean;
  setSpotlight(value: boolean): void;

  getStall(): boolean;
  setStall(value: boolean): void;

  getStockpile(): boolean;
  setStockpile(value: boolean): void;

  getSubstitute(): boolean;
  setSubstitute(value: boolean): void;

  getSubstitutebroken(): boolean;
  setSubstitutebroken(value: boolean): void;

  getSyrupbomb(): boolean;
  setSyrupbomb(value: boolean): void;

  getTarshot(): boolean;
  setTarshot(value: boolean): void;

  getTaunt(): boolean;
  setTaunt(value: boolean): void;

  getTelekinesis(): boolean;
  setTelekinesis(value: boolean): void;

  getThroatchop(): boolean;
  setThroatchop(value: boolean): void;

  getTorment(): boolean;
  setTorment(value: boolean): void;

  getTrapped(): boolean;
  setTrapped(value: boolean): void;

  getTruant(): boolean;
  setTruant(value: boolean): void;

  getTwoturnmove(): boolean;
  setTwoturnmove(value: boolean): void;

  getUnburden(): boolean;
  setUnburden(value: boolean): void;

  getUproar(): boolean;
  setUproar(value: boolean): void;

  getYawn(): boolean;
  setYawn(value: boolean): void;

  getZenmode(): boolean;
  setZenmode(value: boolean): void;

  getAquaring(): boolean;
  setAquaring(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): VolatileStatuses.AsObject;
  static toObject(includeInstance: boolean, msg: VolatileStatuses): VolatileStatuses.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: VolatileStatuses, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): VolatileStatuses;
  static deserializeBinaryFromReader(message: VolatileStatuses, reader: jspb.BinaryReader): VolatileStatuses;
}

export namespace VolatileStatuses {
  export type AsObject = {
    attract: boolean,
    banefulbunker: boolean,
    beakblast: boolean,
    bide: boolean,
    bounce: boolean,
    burningbulwark: boolean,
    charge: boolean,
    choicelock: boolean,
    commanded: boolean,
    commanding: boolean,
    confusion: boolean,
    counter: boolean,
    cudchew: boolean,
    curse: boolean,
    defensecurl: boolean,
    destinybond: boolean,
    detect: boolean,
    dig: boolean,
    disable: boolean,
    dive: boolean,
    dragoncheer: boolean,
    dynamax: boolean,
    electrify: boolean,
    embargo: boolean,
    encore: boolean,
    endure: boolean,
    flashfire: boolean,
    flinch: boolean,
    fling: boolean,
    fly: boolean,
    focusenergy: boolean,
    focuspunch: boolean,
    followme: boolean,
    foresight: boolean,
    furycutter: boolean,
    gastroacid: boolean,
    gem: boolean,
    glaiverush: boolean,
    grudge: boolean,
    healblock: boolean,
    helpinghand: boolean,
    iceball: boolean,
    imprison: boolean,
    ingrain: boolean,
    kingsshield: boolean,
    laserfocus: boolean,
    leechseed: boolean,
    leppaberry: boolean,
    lockedmove: boolean,
    lockon: boolean,
    magiccoat: boolean,
    magnetrise: boolean,
    maxguard: boolean,
    metronome: boolean,
    micleberry: boolean,
    minimize: boolean,
    miracleeye: boolean,
    mirrorcoat: boolean,
    mustrecharge: boolean,
    nightmare: boolean,
    noretreat: boolean,
    obstruct: boolean,
    octolock: boolean,
    partiallytrapped: boolean,
    partialtrappinglock: boolean,
    perishsong: boolean,
    phantomforce: boolean,
    powder: boolean,
    powershift: boolean,
    powertrick: boolean,
    protect: boolean,
    protosynthesis: boolean,
    quarkdrive: boolean,
    rage: boolean,
    ragepowder: boolean,
    rollout: boolean,
    rolloutstorage: boolean,
    roost: boolean,
    saltcure: boolean,
    shadowforce: boolean,
    shelltrap: boolean,
    silktrap: boolean,
    skydrop: boolean,
    slowstart: boolean,
    smackdown: boolean,
    snatch: boolean,
    sparklingaria: boolean,
    spikyshield: boolean,
    spotlight: boolean,
    stall: boolean,
    stockpile: boolean,
    substitute: boolean,
    substitutebroken: boolean,
    syrupbomb: boolean,
    tarshot: boolean,
    taunt: boolean,
    telekinesis: boolean,
    throatchop: boolean,
    torment: boolean,
    trapped: boolean,
    truant: boolean,
    twoturnmove: boolean,
    unburden: boolean,
    uproar: boolean,
    yawn: boolean,
    zenmode: boolean,
    aquaring: boolean,
  }
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
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Boosts, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Boosts;
  static deserializeBinaryFromReader(message: Boosts, reader: jspb.BinaryReader): Boosts;
}

export namespace Boosts {
  export type AsObject = {
    atk: number,
    def: number,
    spa: number,
    spd: number,
    spe: number,
    accuracy: number,
    evasion: number,
  }
}

export class Pseudoweathers extends jspb.Message {
  getFairylock(): boolean;
  setFairylock(value: boolean): void;

  getFairylockMinDur(): number;
  setFairylockMinDur(value: number): void;

  getFairylockMaxDur(): number;
  setFairylockMaxDur(value: number): void;

  getGravity(): boolean;
  setGravity(value: boolean): void;

  getGravityMinDur(): number;
  setGravityMinDur(value: number): void;

  getGravityMaxDur(): number;
  setGravityMaxDur(value: number): void;

  getIondeluge(): boolean;
  setIondeluge(value: boolean): void;

  getIondelugeMinDur(): number;
  setIondelugeMinDur(value: number): void;

  getIondelugeMaxDur(): number;
  setIondelugeMaxDur(value: number): void;

  getMagicroom(): boolean;
  setMagicroom(value: boolean): void;

  getMagicroomMinDur(): number;
  setMagicroomMinDur(value: number): void;

  getMagicroomMaxDur(): number;
  setMagicroomMaxDur(value: number): void;

  getMudsport(): boolean;
  setMudsport(value: boolean): void;

  getMudsportMinDur(): number;
  setMudsportMinDur(value: number): void;

  getMudsportMaxDur(): number;
  setMudsportMaxDur(value: number): void;

  getTrickroom(): boolean;
  setTrickroom(value: boolean): void;

  getTrickroomMinDur(): number;
  setTrickroomMinDur(value: number): void;

  getTrickroomMaxDur(): number;
  setTrickroomMaxDur(value: number): void;

  getWatersport(): boolean;
  setWatersport(value: boolean): void;

  getWatersportMinDur(): number;
  setWatersportMinDur(value: number): void;

  getWatersportMaxDur(): number;
  setWatersportMaxDur(value: number): void;

  getWonderroom(): boolean;
  setWonderroom(value: boolean): void;

  getWonderroomMinDur(): number;
  setWonderroomMinDur(value: number): void;

  getWonderroomMaxDur(): number;
  setWonderroomMaxDur(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Pseudoweathers.AsObject;
  static toObject(includeInstance: boolean, msg: Pseudoweathers): Pseudoweathers.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Pseudoweathers, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Pseudoweathers;
  static deserializeBinaryFromReader(message: Pseudoweathers, reader: jspb.BinaryReader): Pseudoweathers;
}

export namespace Pseudoweathers {
  export type AsObject = {
    fairylock: boolean,
    fairylockMinDur: number,
    fairylockMaxDur: number,
    gravity: boolean,
    gravityMinDur: number,
    gravityMaxDur: number,
    iondeluge: boolean,
    iondelugeMinDur: number,
    iondelugeMaxDur: number,
    magicroom: boolean,
    magicroomMinDur: number,
    magicroomMaxDur: number,
    mudsport: boolean,
    mudsportMinDur: number,
    mudsportMaxDur: number,
    trickroom: boolean,
    trickroomMinDur: number,
    trickroomMaxDur: number,
    watersport: boolean,
    watersportMinDur: number,
    watersportMaxDur: number,
    wonderroom: boolean,
    wonderroomMinDur: number,
    wonderroomMaxDur: number,
  }
}

export class Weather extends jspb.Message {
  getWeather(): state_v1_enum_pb.WeathersMap[keyof state_v1_enum_pb.WeathersMap];
  setWeather(value: state_v1_enum_pb.WeathersMap[keyof state_v1_enum_pb.WeathersMap]): void;

  getMinDur(): number;
  setMinDur(value: number): void;

  getMaxDur(): number;
  setMaxDur(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Weather.AsObject;
  static toObject(includeInstance: boolean, msg: Weather): Weather.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Weather, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Weather;
  static deserializeBinaryFromReader(message: Weather, reader: jspb.BinaryReader): Weather;
}

export namespace Weather {
  export type AsObject = {
    weather: state_v1_enum_pb.WeathersMap[keyof state_v1_enum_pb.WeathersMap],
    minDur: number,
    maxDur: number,
  }
}

export class Terrains extends jspb.Message {
  getTerrain(): state_v1_enum_pb.TerrainMap[keyof state_v1_enum_pb.TerrainMap];
  setTerrain(value: state_v1_enum_pb.TerrainMap[keyof state_v1_enum_pb.TerrainMap]): void;

  getMinDur(): number;
  setMinDur(value: number): void;

  getMaxDur(): number;
  setMaxDur(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Terrains.AsObject;
  static toObject(includeInstance: boolean, msg: Terrains): Terrains.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Terrains, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Terrains;
  static deserializeBinaryFromReader(message: Terrains, reader: jspb.BinaryReader): Terrains;
}

export namespace Terrains {
  export type AsObject = {
    terrain: state_v1_enum_pb.TerrainMap[keyof state_v1_enum_pb.TerrainMap],
    minDur: number,
    maxDur: number,
  }
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
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Field, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Field;
  static deserializeBinaryFromReader(message: Field, reader: jspb.BinaryReader): Field;
}

export namespace Field {
  export type AsObject = {
    pseudoweather?: Pseudoweathers.AsObject,
    weather?: Weather.AsObject,
    terrain?: Terrains.AsObject,
  }
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
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Context, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Context;
  static deserializeBinaryFromReader(message: Context, reader: jspb.BinaryReader): Context;
}

export namespace Context {
  export type AsObject = {
    sideConditions1?: SideConditions.AsObject,
    sideConditions2?: SideConditions.AsObject,
    volatileStatus1?: VolatileStatuses.AsObject,
    volatileStatus2?: VolatileStatuses.AsObject,
    boosts1?: Boosts.AsObject,
    boosts2?: Boosts.AsObject,
    field?: Field.AsObject,
  }
}

