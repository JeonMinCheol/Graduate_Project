# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='message.proto',
  package='message',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rmessage.proto\x12\x07message\"F\n\x11\x63lientInformation\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t\x12\x0f\n\x07n_class\x18\x02 \x01(\x05\x12\x12\n\nmodel_name\x18\x03 \x01(\t\" \n\rEmptyResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"O\n\x0eSelectedStates\x12\r\n\x05state\x18\x01 \x01(\x0c\x12\x0f\n\x07setting\x18\x02 \x01(\t\x12\x0f\n\x07weights\x18\x03 \x01(\x0c\x12\x0c\n\x04\x64rop\x18\x04 \x01(\x05\"q\n\x0bGlobalState\x12\r\n\x05state\x18\x01 \x01(\x0c\x12\x0c\n\x04loss\x18\x02 \x01(\x02\x12\x10\n\x08\x61\x63\x63uracy\x18\x03 \x01(\x02\x12\x0b\n\x03mae\x18\x04 \x01(\x02\x12\x0b\n\x03mse\x18\x05 \x01(\x02\x12\x0b\n\x03rse\x18\x06 \x01(\x02\x12\x0c\n\x04rmse\x18\x07 \x01(\x02\x32\xcf\x01\n\x0bgrpcService\x12<\n\tsendState\x12\x17.message.SelectedStates\x1a\x14.message.GlobalState\"\x00\x12@\n\x08valSetup\x12\x1a.message.clientInformation\x1a\x16.message.EmptyResponse\"\x00\x12@\n\x0egetGlobalModel\x12\x16.message.EmptyResponse\x1a\x14.message.GlobalState\"\x00\x62\x06proto3'
)




_CLIENTINFORMATION = _descriptor.Descriptor(
  name='clientInformation',
  full_name='message.clientInformation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='message.clientInformation.data', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='n_class', full_name='message.clientInformation.n_class', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_name', full_name='message.clientInformation.model_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=26,
  serialized_end=96,
)


_EMPTYRESPONSE = _descriptor.Descriptor(
  name='EmptyResponse',
  full_name='message.EmptyResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='message.EmptyResponse.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=98,
  serialized_end=130,
)


_SELECTEDSTATES = _descriptor.Descriptor(
  name='SelectedStates',
  full_name='message.SelectedStates',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='message.SelectedStates.state', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='setting', full_name='message.SelectedStates.setting', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='weights', full_name='message.SelectedStates.weights', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='drop', full_name='message.SelectedStates.drop', index=3,
      number=4, type=5, cpp_type=1, label=1,
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
  serialized_start=132,
  serialized_end=211,
)


_GLOBALSTATE = _descriptor.Descriptor(
  name='GlobalState',
  full_name='message.GlobalState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='message.GlobalState.state', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='loss', full_name='message.GlobalState.loss', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='accuracy', full_name='message.GlobalState.accuracy', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mae', full_name='message.GlobalState.mae', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mse', full_name='message.GlobalState.mse', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rse', full_name='message.GlobalState.rse', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rmse', full_name='message.GlobalState.rmse', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=213,
  serialized_end=326,
)

DESCRIPTOR.message_types_by_name['clientInformation'] = _CLIENTINFORMATION
DESCRIPTOR.message_types_by_name['EmptyResponse'] = _EMPTYRESPONSE
DESCRIPTOR.message_types_by_name['SelectedStates'] = _SELECTEDSTATES
DESCRIPTOR.message_types_by_name['GlobalState'] = _GLOBALSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

clientInformation = _reflection.GeneratedProtocolMessageType('clientInformation', (_message.Message,), {
  'DESCRIPTOR' : _CLIENTINFORMATION,
  '__module__' : 'message_pb2'
  # @@protoc_insertion_point(class_scope:message.clientInformation)
  })
_sym_db.RegisterMessage(clientInformation)

EmptyResponse = _reflection.GeneratedProtocolMessageType('EmptyResponse', (_message.Message,), {
  'DESCRIPTOR' : _EMPTYRESPONSE,
  '__module__' : 'message_pb2'
  # @@protoc_insertion_point(class_scope:message.EmptyResponse)
  })
_sym_db.RegisterMessage(EmptyResponse)

SelectedStates = _reflection.GeneratedProtocolMessageType('SelectedStates', (_message.Message,), {
  'DESCRIPTOR' : _SELECTEDSTATES,
  '__module__' : 'message_pb2'
  # @@protoc_insertion_point(class_scope:message.SelectedStates)
  })
_sym_db.RegisterMessage(SelectedStates)

GlobalState = _reflection.GeneratedProtocolMessageType('GlobalState', (_message.Message,), {
  'DESCRIPTOR' : _GLOBALSTATE,
  '__module__' : 'message_pb2'
  # @@protoc_insertion_point(class_scope:message.GlobalState)
  })
_sym_db.RegisterMessage(GlobalState)



_GRPCSERVICE = _descriptor.ServiceDescriptor(
  name='grpcService',
  full_name='message.grpcService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=329,
  serialized_end=536,
  methods=[
  _descriptor.MethodDescriptor(
    name='sendState',
    full_name='message.grpcService.sendState',
    index=0,
    containing_service=None,
    input_type=_SELECTEDSTATES,
    output_type=_GLOBALSTATE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='valSetup',
    full_name='message.grpcService.valSetup',
    index=1,
    containing_service=None,
    input_type=_CLIENTINFORMATION,
    output_type=_EMPTYRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='getGlobalModel',
    full_name='message.grpcService.getGlobalModel',
    index=2,
    containing_service=None,
    input_type=_EMPTYRESPONSE,
    output_type=_GLOBALSTATE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_GRPCSERVICE)

DESCRIPTOR.services_by_name['grpcService'] = _GRPCSERVICE

# @@protoc_insertion_point(module_scope)
