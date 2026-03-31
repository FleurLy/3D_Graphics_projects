import numpy as np


def readply(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    vertices = []
    triangles = []

    nb_vertices = 0
    nb_faces = 0
    state = 0
    counter = 0

    # Simple ASCII PLY parser matching the TP file format.
    for line in lines:
        if state == 0:
            tokens = line.rstrip().split(" ")
            if tokens[0] == "element" and tokens[1] == "vertex":
                nb_vertices = int(tokens[2])
            if tokens[0] == "element" and tokens[1] == "face":
                nb_faces = int(tokens[2])
            if tokens[0] == "end_header":
                state = 1
                continue

        elif state == 1:
            tokens = line.split(" ")
            vertex = [float(v) for v in tokens]
            vertices.append(vertex)
            counter += 1
            if counter == nb_vertices:
                counter = 0
                state = 2
                continue

        elif state == 2:
            tokens = line.split(" ")
            tokens.pop(0)  # first value is number of indices in this face
            triangle = [int(v) for v in tokens]
            triangles.append(triangle)
            counter += 1
            if counter == nb_faces:
                break

    return np.array(vertices), np.array(triangles, dtype=int)
