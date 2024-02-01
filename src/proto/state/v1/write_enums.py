import json

from typing import Dict


def write_enum(buf: str, datum: str, values: Dict[str, int]) -> str:
    buf += "enum " + datum.capitalize() + " {\n"
    for key, token in values.items():
        buf += f"\t_{key.upper()} = {token+1};\n"
    buf += "}\n\n"
    return buf


def main():
    with open("src/data.json", "r") as f:
        data = json.load(f)

    proto_buffer = """
syntax = "proto3";

package state.v1;

"""

    for datum, values in data.items():
        proto_buffer = write_enum(proto_buffer, datum, values)

    with open("src/proto/state/v1/enum.proto", "w") as f:
        f.write(proto_buffer)


if __name__ == "__main__":
    main()
