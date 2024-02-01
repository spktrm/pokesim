# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: state/v1/game.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='state/v1/game.proto',
  package='state.v1',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x13state/v1/game.proto\x12\x08state.v1\"x\n\x04Game\x12\x14\n\x0cworker_index\x18\x01 \x01(\x05\x12\x14\n\x0cplayer_index\x18\x02 \x01(\x08\x12\x0c\n\x04\x64one\x18\x03 \x01(\x08\x12\x0e\n\x06reward\x18\x04 \x01(\x05\x12\x0c\n\x04turn\x18\x08 \x01(\x05\x12\x18\n\x10heuristic_action\x18\t \x01(\x05\x62\x06proto3'
)




_GAME = _descriptor.Descriptor(
  name='Game',
  full_name='state.v1.Game',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='worker_index', full_name='state.v1.Game.worker_index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='player_index', full_name='state.v1.Game.player_index', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='done', full_name='state.v1.Game.done', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='reward', full_name='state.v1.Game.reward', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='turn', full_name='state.v1.Game.turn', index=4,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='heuristic_action', full_name='state.v1.Game.heuristic_action', index=5,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=33,
  serialized_end=153,
)

DESCRIPTOR.message_types_by_name['Game'] = _GAME
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Game = _reflection.GeneratedProtocolMessageType('Game', (_message.Message,), {
  'DESCRIPTOR' : _GAME,
  '__module__' : 'state.v1.game_pb2'
  # @@protoc_insertion_point(class_scope:state.v1.Game)
  })
_sym_db.RegisterMessage(Game)


# @@protoc_insertion_point(module_scope)
