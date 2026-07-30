#!/usr/bin/env python3
import argparse
import math
import os
import struct

PLY_TYPE_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


class PlyHeader:
    def __init__(self):
        self.format = None
        self.vertex_count = None
        self.vertex_properties = []
        self.other_elements = []
        self.comments = []


def parse_header(fp):
    header = PlyHeader()
    current_element = None

    first = fp.readline().decode("ascii", errors="ignore").strip()
    if first != "ply":
        raise ValueError("Not a PLY file")

    while True:
        line_bytes = fp.readline()
        if not line_bytes:
            raise ValueError("Unexpected EOF while reading header")
        line = line_bytes.decode("ascii", errors="ignore").strip()

        if line.startswith("comment "):
            header.comments.append(line)
            continue

        if line.startswith("format "):
            header.format = line.split()[1]
            continue

        if line.startswith("element "):
            parts = line.split()
            current_element = parts[1]
            count = int(parts[2])
            if current_element == "vertex":
                header.vertex_count = count
            else:
                header.other_elements.append((current_element, count))
            continue

        if line.startswith("property "):
            parts = line.split()
            if parts[1] == "list":
                # list properties are ignored for vertex-only export
                continue
            if current_element == "vertex":
                dtype = parts[1]
                name = parts[2]
                header.vertex_properties.append((name, dtype))
            continue

        if line == "end_header":
            break

    if header.format != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {header.format}")
    if header.vertex_count is None:
        raise ValueError("Vertex element not found")

    return header


def build_struct(properties):
    fmt = "<" + "".join(PLY_TYPE_FORMATS[dtype] for _, dtype in properties)
    return struct.Struct(fmt)


def pick_indices(properties, names):
    index_by_name = {name: idx for idx, (name, _) in enumerate(properties)}
    indices = []
    for name in names:
        if name in index_by_name:
            indices.append(index_by_name[name])
    return indices


def main():
    parser = argparse.ArgumentParser(description="Downsample binary little-endian PLY point clouds.")
    parser.add_argument("--input", required=True, help="Input PLY file")
    parser.add_argument("--output", required=True, help="Output PLY file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-points", type=int, help="Approximate number of points to keep")
    group.add_argument("--stride", type=int, help="Keep every Nth point")
    group.add_argument("--fraction", type=float, help="Keep fraction of points (0-1)")
    args = parser.parse_args()

    if args.fraction is not None and (args.fraction <= 0 or args.fraction > 1):
        raise ValueError("fraction must be in (0, 1]")

    with open(args.input, "rb") as fp:
        header = parse_header(fp)
        vertex_struct = build_struct(header.vertex_properties)
        record_size = vertex_struct.size

        if args.stride:
            stride = max(args.stride, 1)
        elif args.target_points:
            stride = max(int(math.ceil(header.vertex_count / args.target_points)), 1)
        else:
            stride = max(int(math.floor(1 / args.fraction)), 1)

        output_count = (header.vertex_count + stride - 1) // stride

        color_indices = pick_indices(header.vertex_properties, ["red", "green", "blue", "r", "g", "b"])
        position_indices = pick_indices(header.vertex_properties, ["x", "y", "z"])
        if len(position_indices) != 3:
            raise ValueError("PLY must include x, y, z vertex properties")

        include_color = len(color_indices) == 3

        with open(args.output, "wb") as out:
            out.write(b"ply\n")
            out.write(b"format binary_little_endian 1.0\n")
            out.write(f"comment downsampled from {os.path.basename(args.input)}\n".encode("ascii"))
            out.write(f"element vertex {output_count}\n".encode("ascii"))
            out.write(b"property float x\n")
            out.write(b"property float y\n")
            out.write(b"property float z\n")
            if include_color:
                out.write(b"property uchar red\n")
                out.write(b"property uchar green\n")
                out.write(b"property uchar blue\n")
            out.write(b"end_header\n")

            for i in range(header.vertex_count):
                record = fp.read(record_size)
                if len(record) != record_size:
                    raise ValueError("Unexpected EOF while reading vertex data")

                if i % stride != 0:
                    continue

                values = vertex_struct.unpack(record)
                x = float(values[position_indices[0]])
                y = float(values[position_indices[1]])
                z = float(values[position_indices[2]])

                if include_color:
                    r = int(values[color_indices[0]])
                    g = int(values[color_indices[1]])
                    b = int(values[color_indices[2]])
                    out.write(struct.pack("<fffBBB", x, y, z, r, g, b))
                else:
                    out.write(struct.pack("<fff", x, y, z))


if __name__ == "__main__":
    main()
