import copy

import numpy as np
import pyvista


def read_vtk(in_file):
    """
    Read .vtk POLYDATA file

    in_file: string,  the filename
    Out: dictionary, 'vertices', 'faces', 'curv', 'sulc', ...
    """

    polydata = pyvista.read(in_file)

    n_faces = polydata.n_faces
    vertices = np.array(polydata.points)  # get vertices coordinate

    # only for triangles polygons data
    faces = np.array(polydata.GetPolys().GetData())  # get faces connectivity
    assert len(faces) / 4 == n_faces, "faces number is not consistent!"
    faces = np.reshape(faces, (n_faces, 4))

    data = {'vertices': vertices,
            'faces': faces
            }

    point_arrays = polydata.point_arrays
    for key, value in point_arrays.items():
        if value.dtype == 'uint32':
            data[key] = np.array(value).astype(np.int64)
        elif value.dtype == 'uint8':
            data[key] = np.array(value).astype(np.int32)
        else:
            data[key] = np.array(value)

    return data


def write_vtk(in_dic, file):
    """
    Write .vtk POLYDATA file

    in_dic: dictionary, vtk data
    file: string, output file name
    """
    assert 'vertices' in in_dic, "output vtk data does not have vertices!"
    assert 'faces' in in_dic, "output vtk data does not have faces!"

    data = copy.deepcopy(in_dic)

    vertices = data['vertices']
    faces = data['faces']
    surf = pyvista.PolyData(vertices, faces)

    del data['vertices']
    del data['faces']
    for key, value in data.items():
        surf.point_arrays[key] = value

    surf.save(file, binary=False)
