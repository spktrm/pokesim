// source: state/v1/pokemon.proto
/**
 * @fileoverview
 * @enhanceable
 * @suppress {messageConventions} JS Compiler reports an error if a variable or
 *     field starts with 'MSG_' and isn't a translatable message.
 * @public
 */
// GENERATED CODE -- DO NOT EDIT!

var jspb = require("google-protobuf");
var goog = jspb;
var global = Function("return this")();

var state_v1_enum_pb = require("../../state/v1/enum_pb.js");
goog.object.extend(proto, state_v1_enum_pb);
goog.exportSymbol("proto.state.v1.Pokemon", null, global);
goog.exportSymbol("proto.state.v1.Team", null, global);
/**
 * Generated by JsPbCodeGenerator.
 * @param {Array=} opt_data Optional initial data array, typically from a
 * server response, or constructed directly in Javascript. The array is used
 * in place and becomes part of the constructed object. It is not cloned.
 * If no data is provided, the constructed object will be empty, but still
 * valid.
 * @extends {jspb.Message}
 * @constructor
 */
proto.state.v1.Pokemon = function (opt_data) {
    jspb.Message.initialize(this, opt_data, 0, -1, null, null);
};
goog.inherits(proto.state.v1.Pokemon, jspb.Message);
if (goog.DEBUG && !COMPILED) {
    /**
     * @public
     * @override
     */
    proto.state.v1.Pokemon.displayName = "proto.state.v1.Pokemon";
}
/**
 * Generated by JsPbCodeGenerator.
 * @param {Array=} opt_data Optional initial data array, typically from a
 * server response, or constructed directly in Javascript. The array is used
 * in place and becomes part of the constructed object. It is not cloned.
 * If no data is provided, the constructed object will be empty, but still
 * valid.
 * @extends {jspb.Message}
 * @constructor
 */
proto.state.v1.Team = function (opt_data) {
    jspb.Message.initialize(this, opt_data, 0, -1, null, null);
};
goog.inherits(proto.state.v1.Team, jspb.Message);
if (goog.DEBUG && !COMPILED) {
    /**
     * @public
     * @override
     */
    proto.state.v1.Team.displayName = "proto.state.v1.Team";
}

if (jspb.Message.GENERATE_TO_OBJECT) {
    /**
     * Creates an object representation of this proto.
     * Field names that are reserved in JavaScript and will be renamed to pb_name.
     * Optional fields that are not set will be set to undefined.
     * To access a reserved field use, foo.pb_<name>, eg, foo.pb_default.
     * For the list of reserved names please see:
     *     net/proto2/compiler/js/internal/generator.cc#kKeyword.
     * @param {boolean=} opt_includeInstance Deprecated. whether to include the
     *     JSPB instance for transitional soy proto support:
     *     http://goto/soy-param-migration
     * @return {!Object}
     */
    proto.state.v1.Pokemon.prototype.toObject = function (opt_includeInstance) {
        return proto.state.v1.Pokemon.toObject(opt_includeInstance, this);
    };

    /**
     * Static version of the {@see toObject} method.
     * @param {boolean|undefined} includeInstance Deprecated. Whether to include
     *     the JSPB instance for transitional soy proto support:
     *     http://goto/soy-param-migration
     * @param {!proto.state.v1.Pokemon} msg The msg instance to transform.
     * @return {!Object}
     * @suppress {unusedLocalVariables} f is only used for nested messages
     */
    proto.state.v1.Pokemon.toObject = function (includeInstance, msg) {
        var f,
            obj = {
                species: jspb.Message.getFieldWithDefault(msg, 1, 0),
                item: jspb.Message.getFieldWithDefault(msg, 2, 0),
                ability: jspb.Message.getFieldWithDefault(msg, 3, 0),
                hp: jspb.Message.getFloatingPointFieldWithDefault(msg, 4, 0.0),
                active: jspb.Message.getBooleanFieldWithDefault(msg, 5, false),
                fainted: jspb.Message.getBooleanFieldWithDefault(msg, 6, false),
                status: jspb.Message.getFieldWithDefault(msg, 7, 0),
                lastMove: jspb.Message.getFieldWithDefault(msg, 8, 0),
                pb_public: jspb.Message.getBooleanFieldWithDefault(
                    msg,
                    9,
                    false,
                ),
                side: jspb.Message.getBooleanFieldWithDefault(msg, 10, false),
                sleepTurns: jspb.Message.getFieldWithDefault(msg, 11, 0),
                toxicTurns: jspb.Message.getFieldWithDefault(msg, 12, 0),
                move1Ppleft: jspb.Message.getFieldWithDefault(msg, 13, 0),
                move2Ppleft: jspb.Message.getFieldWithDefault(msg, 14, 0),
                move3Ppleft: jspb.Message.getFieldWithDefault(msg, 15, 0),
                move4Ppleft: jspb.Message.getFieldWithDefault(msg, 16, 0),
                move1Ppmax: jspb.Message.getFieldWithDefault(msg, 17, 0),
                move2Ppmax: jspb.Message.getFieldWithDefault(msg, 18, 0),
                move3Ppmax: jspb.Message.getFieldWithDefault(msg, 19, 0),
                move4Ppmax: jspb.Message.getFieldWithDefault(msg, 20, 0),
                move1: jspb.Message.getFieldWithDefault(msg, 21, 0),
                move2: jspb.Message.getFieldWithDefault(msg, 22, 0),
                move3: jspb.Message.getFieldWithDefault(msg, 23, 0),
                move4: jspb.Message.getFieldWithDefault(msg, 24, 0),
            };

        if (includeInstance) {
            obj.$jspbMessageInstance = msg;
        }
        return obj;
    };
}

/**
 * Deserializes binary data (in protobuf wire format).
 * @param {jspb.ByteSource} bytes The bytes to deserialize.
 * @return {!proto.state.v1.Pokemon}
 */
proto.state.v1.Pokemon.deserializeBinary = function (bytes) {
    var reader = new jspb.BinaryReader(bytes);
    var msg = new proto.state.v1.Pokemon();
    return proto.state.v1.Pokemon.deserializeBinaryFromReader(msg, reader);
};

/**
 * Deserializes binary data (in protobuf wire format) from the
 * given reader into the given message object.
 * @param {!proto.state.v1.Pokemon} msg The message object to deserialize into.
 * @param {!jspb.BinaryReader} reader The BinaryReader to use.
 * @return {!proto.state.v1.Pokemon}
 */
proto.state.v1.Pokemon.deserializeBinaryFromReader = function (msg, reader) {
    while (reader.nextField()) {
        if (reader.isEndGroup()) {
            break;
        }
        var field = reader.getFieldNumber();
        switch (field) {
            case 1:
                var value = /** @type {!proto.state.v1.Species} */ (
                    reader.readEnum()
                );
                msg.setSpecies(value);
                break;
            case 2:
                var value = /** @type {!proto.state.v1.Items} */ (
                    reader.readEnum()
                );
                msg.setItem(value);
                break;
            case 3:
                var value = /** @type {!proto.state.v1.Abilities} */ (
                    reader.readEnum()
                );
                msg.setAbility(value);
                break;
            case 4:
                var value = /** @type {number} */ (reader.readFloat());
                msg.setHp(value);
                break;
            case 5:
                var value = /** @type {boolean} */ (reader.readBool());
                msg.setActive(value);
                break;
            case 6:
                var value = /** @type {boolean} */ (reader.readBool());
                msg.setFainted(value);
                break;
            case 7:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setStatus(value);
                break;
            case 8:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setLastMove(value);
                break;
            case 9:
                var value = /** @type {boolean} */ (reader.readBool());
                msg.setPublic(value);
                break;
            case 10:
                var value = /** @type {boolean} */ (reader.readBool());
                msg.setSide(value);
                break;
            case 11:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setSleepTurns(value);
                break;
            case 12:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setToxicTurns(value);
                break;
            case 13:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setMove1Ppleft(value);
                break;
            case 14:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setMove2Ppleft(value);
                break;
            case 15:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setMove3Ppleft(value);
                break;
            case 16:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setMove4Ppleft(value);
                break;
            case 17:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setMove1Ppmax(value);
                break;
            case 18:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setMove2Ppmax(value);
                break;
            case 19:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setMove3Ppmax(value);
                break;
            case 20:
                var value = /** @type {number} */ (reader.readInt32());
                msg.setMove4Ppmax(value);
                break;
            case 21:
                var value = /** @type {!proto.state.v1.Moves} */ (
                    reader.readEnum()
                );
                msg.setMove1(value);
                break;
            case 22:
                var value = /** @type {!proto.state.v1.Moves} */ (
                    reader.readEnum()
                );
                msg.setMove2(value);
                break;
            case 23:
                var value = /** @type {!proto.state.v1.Moves} */ (
                    reader.readEnum()
                );
                msg.setMove3(value);
                break;
            case 24:
                var value = /** @type {!proto.state.v1.Moves} */ (
                    reader.readEnum()
                );
                msg.setMove4(value);
                break;
            default:
                reader.skipField();
                break;
        }
    }
    return msg;
};

/**
 * Serializes the message to binary data (in protobuf wire format).
 * @return {!Uint8Array}
 */
proto.state.v1.Pokemon.prototype.serializeBinary = function () {
    var writer = new jspb.BinaryWriter();
    proto.state.v1.Pokemon.serializeBinaryToWriter(this, writer);
    return writer.getResultBuffer();
};

/**
 * Serializes the given message to binary data (in protobuf wire
 * format), writing to the given BinaryWriter.
 * @param {!proto.state.v1.Pokemon} message
 * @param {!jspb.BinaryWriter} writer
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.state.v1.Pokemon.serializeBinaryToWriter = function (message, writer) {
    var f = undefined;
    f = message.getSpecies();
    if (f !== 0.0) {
        writer.writeEnum(1, f);
    }
    f = message.getItem();
    if (f !== 0.0) {
        writer.writeEnum(2, f);
    }
    f = message.getAbility();
    if (f !== 0.0) {
        writer.writeEnum(3, f);
    }
    f = message.getHp();
    if (f !== 0.0) {
        writer.writeFloat(4, f);
    }
    f = message.getActive();
    if (f) {
        writer.writeBool(5, f);
    }
    f = message.getFainted();
    if (f) {
        writer.writeBool(6, f);
    }
    f = message.getStatus();
    if (f !== 0) {
        writer.writeInt32(7, f);
    }
    f = message.getLastMove();
    if (f !== 0) {
        writer.writeInt32(8, f);
    }
    f = message.getPublic();
    if (f) {
        writer.writeBool(9, f);
    }
    f = message.getSide();
    if (f) {
        writer.writeBool(10, f);
    }
    f = message.getSleepTurns();
    if (f !== 0) {
        writer.writeInt32(11, f);
    }
    f = message.getToxicTurns();
    if (f !== 0) {
        writer.writeInt32(12, f);
    }
    f = message.getMove1Ppleft();
    if (f !== 0) {
        writer.writeInt32(13, f);
    }
    f = message.getMove2Ppleft();
    if (f !== 0) {
        writer.writeInt32(14, f);
    }
    f = message.getMove3Ppleft();
    if (f !== 0) {
        writer.writeInt32(15, f);
    }
    f = message.getMove4Ppleft();
    if (f !== 0) {
        writer.writeInt32(16, f);
    }
    f = message.getMove1Ppmax();
    if (f !== 0) {
        writer.writeInt32(17, f);
    }
    f = message.getMove2Ppmax();
    if (f !== 0) {
        writer.writeInt32(18, f);
    }
    f = message.getMove3Ppmax();
    if (f !== 0) {
        writer.writeInt32(19, f);
    }
    f = message.getMove4Ppmax();
    if (f !== 0) {
        writer.writeInt32(20, f);
    }
    f = message.getMove1();
    if (f !== 0.0) {
        writer.writeEnum(21, f);
    }
    f = message.getMove2();
    if (f !== 0.0) {
        writer.writeEnum(22, f);
    }
    f = message.getMove3();
    if (f !== 0.0) {
        writer.writeEnum(23, f);
    }
    f = message.getMove4();
    if (f !== 0.0) {
        writer.writeEnum(24, f);
    }
};

/**
 * optional Species species = 1;
 * @return {!proto.state.v1.Species}
 */
proto.state.v1.Pokemon.prototype.getSpecies = function () {
    return /** @type {!proto.state.v1.Species} */ (
        jspb.Message.getFieldWithDefault(this, 1, 0)
    );
};

/**
 * @param {!proto.state.v1.Species} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setSpecies = function (value) {
    return jspb.Message.setProto3EnumField(this, 1, value);
};

/**
 * optional Items item = 2;
 * @return {!proto.state.v1.Items}
 */
proto.state.v1.Pokemon.prototype.getItem = function () {
    return /** @type {!proto.state.v1.Items} */ (
        jspb.Message.getFieldWithDefault(this, 2, 0)
    );
};

/**
 * @param {!proto.state.v1.Items} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setItem = function (value) {
    return jspb.Message.setProto3EnumField(this, 2, value);
};

/**
 * optional Abilities ability = 3;
 * @return {!proto.state.v1.Abilities}
 */
proto.state.v1.Pokemon.prototype.getAbility = function () {
    return /** @type {!proto.state.v1.Abilities} */ (
        jspb.Message.getFieldWithDefault(this, 3, 0)
    );
};

/**
 * @param {!proto.state.v1.Abilities} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setAbility = function (value) {
    return jspb.Message.setProto3EnumField(this, 3, value);
};

/**
 * optional float hp = 4;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getHp = function () {
    return /** @type {number} */ (
        jspb.Message.getFloatingPointFieldWithDefault(this, 4, 0.0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setHp = function (value) {
    return jspb.Message.setProto3FloatField(this, 4, value);
};

/**
 * optional bool active = 5;
 * @return {boolean}
 */
proto.state.v1.Pokemon.prototype.getActive = function () {
    return /** @type {boolean} */ (
        jspb.Message.getBooleanFieldWithDefault(this, 5, false)
    );
};

/**
 * @param {boolean} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setActive = function (value) {
    return jspb.Message.setProto3BooleanField(this, 5, value);
};

/**
 * optional bool fainted = 6;
 * @return {boolean}
 */
proto.state.v1.Pokemon.prototype.getFainted = function () {
    return /** @type {boolean} */ (
        jspb.Message.getBooleanFieldWithDefault(this, 6, false)
    );
};

/**
 * @param {boolean} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setFainted = function (value) {
    return jspb.Message.setProto3BooleanField(this, 6, value);
};

/**
 * optional int32 status = 7;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getStatus = function () {
    return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 7, 0));
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setStatus = function (value) {
    return jspb.Message.setProto3IntField(this, 7, value);
};

/**
 * optional int32 last_move = 8;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getLastMove = function () {
    return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 8, 0));
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setLastMove = function (value) {
    return jspb.Message.setProto3IntField(this, 8, value);
};

/**
 * optional bool public = 9;
 * @return {boolean}
 */
proto.state.v1.Pokemon.prototype.getPublic = function () {
    return /** @type {boolean} */ (
        jspb.Message.getBooleanFieldWithDefault(this, 9, false)
    );
};

/**
 * @param {boolean} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setPublic = function (value) {
    return jspb.Message.setProto3BooleanField(this, 9, value);
};

/**
 * optional bool side = 10;
 * @return {boolean}
 */
proto.state.v1.Pokemon.prototype.getSide = function () {
    return /** @type {boolean} */ (
        jspb.Message.getBooleanFieldWithDefault(this, 10, false)
    );
};

/**
 * @param {boolean} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setSide = function (value) {
    return jspb.Message.setProto3BooleanField(this, 10, value);
};

/**
 * optional int32 sleep_turns = 11;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getSleepTurns = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 11, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setSleepTurns = function (value) {
    return jspb.Message.setProto3IntField(this, 11, value);
};

/**
 * optional int32 toxic_turns = 12;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getToxicTurns = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 12, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setToxicTurns = function (value) {
    return jspb.Message.setProto3IntField(this, 12, value);
};

/**
 * optional int32 move1_ppleft = 13;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getMove1Ppleft = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 13, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove1Ppleft = function (value) {
    return jspb.Message.setProto3IntField(this, 13, value);
};

/**
 * optional int32 move2_ppleft = 14;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getMove2Ppleft = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 14, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove2Ppleft = function (value) {
    return jspb.Message.setProto3IntField(this, 14, value);
};

/**
 * optional int32 move3_ppleft = 15;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getMove3Ppleft = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 15, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove3Ppleft = function (value) {
    return jspb.Message.setProto3IntField(this, 15, value);
};

/**
 * optional int32 move4_ppleft = 16;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getMove4Ppleft = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 16, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove4Ppleft = function (value) {
    return jspb.Message.setProto3IntField(this, 16, value);
};

/**
 * optional int32 move1_ppmax = 17;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getMove1Ppmax = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 17, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove1Ppmax = function (value) {
    return jspb.Message.setProto3IntField(this, 17, value);
};

/**
 * optional int32 move2_ppmax = 18;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getMove2Ppmax = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 18, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove2Ppmax = function (value) {
    return jspb.Message.setProto3IntField(this, 18, value);
};

/**
 * optional int32 move3_ppmax = 19;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getMove3Ppmax = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 19, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove3Ppmax = function (value) {
    return jspb.Message.setProto3IntField(this, 19, value);
};

/**
 * optional int32 move4_ppmax = 20;
 * @return {number}
 */
proto.state.v1.Pokemon.prototype.getMove4Ppmax = function () {
    return /** @type {number} */ (
        jspb.Message.getFieldWithDefault(this, 20, 0)
    );
};

/**
 * @param {number} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove4Ppmax = function (value) {
    return jspb.Message.setProto3IntField(this, 20, value);
};

/**
 * optional Moves move1 = 21;
 * @return {!proto.state.v1.Moves}
 */
proto.state.v1.Pokemon.prototype.getMove1 = function () {
    return /** @type {!proto.state.v1.Moves} */ (
        jspb.Message.getFieldWithDefault(this, 21, 0)
    );
};

/**
 * @param {!proto.state.v1.Moves} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove1 = function (value) {
    return jspb.Message.setProto3EnumField(this, 21, value);
};

/**
 * optional Moves move2 = 22;
 * @return {!proto.state.v1.Moves}
 */
proto.state.v1.Pokemon.prototype.getMove2 = function () {
    return /** @type {!proto.state.v1.Moves} */ (
        jspb.Message.getFieldWithDefault(this, 22, 0)
    );
};

/**
 * @param {!proto.state.v1.Moves} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove2 = function (value) {
    return jspb.Message.setProto3EnumField(this, 22, value);
};

/**
 * optional Moves move3 = 23;
 * @return {!proto.state.v1.Moves}
 */
proto.state.v1.Pokemon.prototype.getMove3 = function () {
    return /** @type {!proto.state.v1.Moves} */ (
        jspb.Message.getFieldWithDefault(this, 23, 0)
    );
};

/**
 * @param {!proto.state.v1.Moves} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove3 = function (value) {
    return jspb.Message.setProto3EnumField(this, 23, value);
};

/**
 * optional Moves move4 = 24;
 * @return {!proto.state.v1.Moves}
 */
proto.state.v1.Pokemon.prototype.getMove4 = function () {
    return /** @type {!proto.state.v1.Moves} */ (
        jspb.Message.getFieldWithDefault(this, 24, 0)
    );
};

/**
 * @param {!proto.state.v1.Moves} value
 * @return {!proto.state.v1.Pokemon} returns this
 */
proto.state.v1.Pokemon.prototype.setMove4 = function (value) {
    return jspb.Message.setProto3EnumField(this, 24, value);
};

if (jspb.Message.GENERATE_TO_OBJECT) {
    /**
     * Creates an object representation of this proto.
     * Field names that are reserved in JavaScript and will be renamed to pb_name.
     * Optional fields that are not set will be set to undefined.
     * To access a reserved field use, foo.pb_<name>, eg, foo.pb_default.
     * For the list of reserved names please see:
     *     net/proto2/compiler/js/internal/generator.cc#kKeyword.
     * @param {boolean=} opt_includeInstance Deprecated. whether to include the
     *     JSPB instance for transitional soy proto support:
     *     http://goto/soy-param-migration
     * @return {!Object}
     */
    proto.state.v1.Team.prototype.toObject = function (opt_includeInstance) {
        return proto.state.v1.Team.toObject(opt_includeInstance, this);
    };

    /**
     * Static version of the {@see toObject} method.
     * @param {boolean|undefined} includeInstance Deprecated. Whether to include
     *     the JSPB instance for transitional soy proto support:
     *     http://goto/soy-param-migration
     * @param {!proto.state.v1.Team} msg The msg instance to transform.
     * @return {!Object}
     * @suppress {unusedLocalVariables} f is only used for nested messages
     */
    proto.state.v1.Team.toObject = function (includeInstance, msg) {
        var f,
            obj = {
                pokemon1:
                    (f = msg.getPokemon1()) &&
                    proto.state.v1.Pokemon.toObject(includeInstance, f),
                pokemon2:
                    (f = msg.getPokemon2()) &&
                    proto.state.v1.Pokemon.toObject(includeInstance, f),
                pokemon3:
                    (f = msg.getPokemon3()) &&
                    proto.state.v1.Pokemon.toObject(includeInstance, f),
                pokemon4:
                    (f = msg.getPokemon4()) &&
                    proto.state.v1.Pokemon.toObject(includeInstance, f),
                pokemon5:
                    (f = msg.getPokemon5()) &&
                    proto.state.v1.Pokemon.toObject(includeInstance, f),
                pokemon6:
                    (f = msg.getPokemon6()) &&
                    proto.state.v1.Pokemon.toObject(includeInstance, f),
            };

        if (includeInstance) {
            obj.$jspbMessageInstance = msg;
        }
        return obj;
    };
}

/**
 * Deserializes binary data (in protobuf wire format).
 * @param {jspb.ByteSource} bytes The bytes to deserialize.
 * @return {!proto.state.v1.Team}
 */
proto.state.v1.Team.deserializeBinary = function (bytes) {
    var reader = new jspb.BinaryReader(bytes);
    var msg = new proto.state.v1.Team();
    return proto.state.v1.Team.deserializeBinaryFromReader(msg, reader);
};

/**
 * Deserializes binary data (in protobuf wire format) from the
 * given reader into the given message object.
 * @param {!proto.state.v1.Team} msg The message object to deserialize into.
 * @param {!jspb.BinaryReader} reader The BinaryReader to use.
 * @return {!proto.state.v1.Team}
 */
proto.state.v1.Team.deserializeBinaryFromReader = function (msg, reader) {
    while (reader.nextField()) {
        if (reader.isEndGroup()) {
            break;
        }
        var field = reader.getFieldNumber();
        switch (field) {
            case 1:
                var value = new proto.state.v1.Pokemon();
                reader.readMessage(
                    value,
                    proto.state.v1.Pokemon.deserializeBinaryFromReader,
                );
                msg.setPokemon1(value);
                break;
            case 2:
                var value = new proto.state.v1.Pokemon();
                reader.readMessage(
                    value,
                    proto.state.v1.Pokemon.deserializeBinaryFromReader,
                );
                msg.setPokemon2(value);
                break;
            case 3:
                var value = new proto.state.v1.Pokemon();
                reader.readMessage(
                    value,
                    proto.state.v1.Pokemon.deserializeBinaryFromReader,
                );
                msg.setPokemon3(value);
                break;
            case 4:
                var value = new proto.state.v1.Pokemon();
                reader.readMessage(
                    value,
                    proto.state.v1.Pokemon.deserializeBinaryFromReader,
                );
                msg.setPokemon4(value);
                break;
            case 5:
                var value = new proto.state.v1.Pokemon();
                reader.readMessage(
                    value,
                    proto.state.v1.Pokemon.deserializeBinaryFromReader,
                );
                msg.setPokemon5(value);
                break;
            case 6:
                var value = new proto.state.v1.Pokemon();
                reader.readMessage(
                    value,
                    proto.state.v1.Pokemon.deserializeBinaryFromReader,
                );
                msg.setPokemon6(value);
                break;
            default:
                reader.skipField();
                break;
        }
    }
    return msg;
};

/**
 * Serializes the message to binary data (in protobuf wire format).
 * @return {!Uint8Array}
 */
proto.state.v1.Team.prototype.serializeBinary = function () {
    var writer = new jspb.BinaryWriter();
    proto.state.v1.Team.serializeBinaryToWriter(this, writer);
    return writer.getResultBuffer();
};

/**
 * Serializes the given message to binary data (in protobuf wire
 * format), writing to the given BinaryWriter.
 * @param {!proto.state.v1.Team} message
 * @param {!jspb.BinaryWriter} writer
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.state.v1.Team.serializeBinaryToWriter = function (message, writer) {
    var f = undefined;
    f = message.getPokemon1();
    if (f != null) {
        writer.writeMessage(
            1,
            f,
            proto.state.v1.Pokemon.serializeBinaryToWriter,
        );
    }
    f = message.getPokemon2();
    if (f != null) {
        writer.writeMessage(
            2,
            f,
            proto.state.v1.Pokemon.serializeBinaryToWriter,
        );
    }
    f = message.getPokemon3();
    if (f != null) {
        writer.writeMessage(
            3,
            f,
            proto.state.v1.Pokemon.serializeBinaryToWriter,
        );
    }
    f = message.getPokemon4();
    if (f != null) {
        writer.writeMessage(
            4,
            f,
            proto.state.v1.Pokemon.serializeBinaryToWriter,
        );
    }
    f = message.getPokemon5();
    if (f != null) {
        writer.writeMessage(
            5,
            f,
            proto.state.v1.Pokemon.serializeBinaryToWriter,
        );
    }
    f = message.getPokemon6();
    if (f != null) {
        writer.writeMessage(
            6,
            f,
            proto.state.v1.Pokemon.serializeBinaryToWriter,
        );
    }
};

/**
 * optional Pokemon pokemon1 = 1;
 * @return {?proto.state.v1.Pokemon}
 */
proto.state.v1.Team.prototype.getPokemon1 = function () {
    return /** @type{?proto.state.v1.Pokemon} */ (
        jspb.Message.getWrapperField(this, proto.state.v1.Pokemon, 1)
    );
};

/**
 * @param {?proto.state.v1.Pokemon|undefined} value
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.setPokemon1 = function (value) {
    return jspb.Message.setWrapperField(this, 1, value);
};

/**
 * Clears the message field making it undefined.
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.clearPokemon1 = function () {
    return this.setPokemon1(undefined);
};

/**
 * Returns whether this field is set.
 * @return {boolean}
 */
proto.state.v1.Team.prototype.hasPokemon1 = function () {
    return jspb.Message.getField(this, 1) != null;
};

/**
 * optional Pokemon pokemon2 = 2;
 * @return {?proto.state.v1.Pokemon}
 */
proto.state.v1.Team.prototype.getPokemon2 = function () {
    return /** @type{?proto.state.v1.Pokemon} */ (
        jspb.Message.getWrapperField(this, proto.state.v1.Pokemon, 2)
    );
};

/**
 * @param {?proto.state.v1.Pokemon|undefined} value
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.setPokemon2 = function (value) {
    return jspb.Message.setWrapperField(this, 2, value);
};

/**
 * Clears the message field making it undefined.
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.clearPokemon2 = function () {
    return this.setPokemon2(undefined);
};

/**
 * Returns whether this field is set.
 * @return {boolean}
 */
proto.state.v1.Team.prototype.hasPokemon2 = function () {
    return jspb.Message.getField(this, 2) != null;
};

/**
 * optional Pokemon pokemon3 = 3;
 * @return {?proto.state.v1.Pokemon}
 */
proto.state.v1.Team.prototype.getPokemon3 = function () {
    return /** @type{?proto.state.v1.Pokemon} */ (
        jspb.Message.getWrapperField(this, proto.state.v1.Pokemon, 3)
    );
};

/**
 * @param {?proto.state.v1.Pokemon|undefined} value
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.setPokemon3 = function (value) {
    return jspb.Message.setWrapperField(this, 3, value);
};

/**
 * Clears the message field making it undefined.
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.clearPokemon3 = function () {
    return this.setPokemon3(undefined);
};

/**
 * Returns whether this field is set.
 * @return {boolean}
 */
proto.state.v1.Team.prototype.hasPokemon3 = function () {
    return jspb.Message.getField(this, 3) != null;
};

/**
 * optional Pokemon pokemon4 = 4;
 * @return {?proto.state.v1.Pokemon}
 */
proto.state.v1.Team.prototype.getPokemon4 = function () {
    return /** @type{?proto.state.v1.Pokemon} */ (
        jspb.Message.getWrapperField(this, proto.state.v1.Pokemon, 4)
    );
};

/**
 * @param {?proto.state.v1.Pokemon|undefined} value
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.setPokemon4 = function (value) {
    return jspb.Message.setWrapperField(this, 4, value);
};

/**
 * Clears the message field making it undefined.
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.clearPokemon4 = function () {
    return this.setPokemon4(undefined);
};

/**
 * Returns whether this field is set.
 * @return {boolean}
 */
proto.state.v1.Team.prototype.hasPokemon4 = function () {
    return jspb.Message.getField(this, 4) != null;
};

/**
 * optional Pokemon pokemon5 = 5;
 * @return {?proto.state.v1.Pokemon}
 */
proto.state.v1.Team.prototype.getPokemon5 = function () {
    return /** @type{?proto.state.v1.Pokemon} */ (
        jspb.Message.getWrapperField(this, proto.state.v1.Pokemon, 5)
    );
};

/**
 * @param {?proto.state.v1.Pokemon|undefined} value
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.setPokemon5 = function (value) {
    return jspb.Message.setWrapperField(this, 5, value);
};

/**
 * Clears the message field making it undefined.
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.clearPokemon5 = function () {
    return this.setPokemon5(undefined);
};

/**
 * Returns whether this field is set.
 * @return {boolean}
 */
proto.state.v1.Team.prototype.hasPokemon5 = function () {
    return jspb.Message.getField(this, 5) != null;
};

/**
 * optional Pokemon pokemon6 = 6;
 * @return {?proto.state.v1.Pokemon}
 */
proto.state.v1.Team.prototype.getPokemon6 = function () {
    return /** @type{?proto.state.v1.Pokemon} */ (
        jspb.Message.getWrapperField(this, proto.state.v1.Pokemon, 6)
    );
};

/**
 * @param {?proto.state.v1.Pokemon|undefined} value
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.setPokemon6 = function (value) {
    return jspb.Message.setWrapperField(this, 6, value);
};

/**
 * Clears the message field making it undefined.
 * @return {!proto.state.v1.Team} returns this
 */
proto.state.v1.Team.prototype.clearPokemon6 = function () {
    return this.setPokemon6(undefined);
};

/**
 * Returns whether this field is set.
 * @return {boolean}
 */
proto.state.v1.Team.prototype.hasPokemon6 = function () {
    return jspb.Message.getField(this, 6) != null;
};

goog.object.extend(exports, proto.state.v1);
