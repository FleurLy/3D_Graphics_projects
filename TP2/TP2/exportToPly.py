import numpy as np

def write_ply_file(verts, faces, ply_file_name, write_also_npz=False):
  verts_num = verts.shape[0]
  faces_num = faces.shape[0]
  with open(ply_file_name, 'w') as f:
      f.write('ply\n')
      f.write('format ascii 1.0  \n')
      f.write('element vertex ')
      f.write(str(verts_num))
      f.write('\n')
      f.write('property float x           { vertex contains float "x" coordinate }\n')
      f.write('property float y           { y coordinate is also a vertex property }\n')
      f.write('property float z           { z coordinate, too }\n')
      f.write('element face ')
      f.write(str(faces_num))
      f.write('\n')
      f.write('property list uchar int vertex_index\n')
      f.write('end_header                 { delimits the end of the header }\n')

      for i in range(verts_num):
          f.write(str(verts[i][0]))
          f.write(' ')
          f.write(str(verts[i][1]))
          f.write(' ')
          f.write(str(verts[i][2]))
          f.write('\n')


      for i in range(faces_num):
          f.write('3 ')
          f.write(str(faces[i][0]))
          f.write(' ')
          f.write(str(faces[i][1]))
          f.write(' ')
          f.write(str(faces[i][2]))
          f.write('\n')


