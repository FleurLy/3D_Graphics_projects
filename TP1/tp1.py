import numpy as np
import math as m
# vertices = np.array([
# [0.0,0.0,0.0], #first vertex
# [1.0,0.0,0.0], #second vertex
# [0.0,1.0,0.0], #third vertex
# [1.0,1.0,0.0], #fourth vertex
# [0.0,0.0,1.0], #fifth vertex
# [1.0,0.0,1.0], #sixth vertex
# [0.0,1.0,1.0], #seventh vertex
# [1.0,1.0,1.0], #eighth vertex
# ])

# triangles = np.array([
# [0,1,2] ,# create a triangle with the first, second, and third vertices
# [1,3,2], # create a triangle with the second, the third and the fourth vertices
# [1,3,5],
# [5,3,7],
# [0,1,4],
# [4,1,5],
# [4,5,6],
# [6,5,7],
# [2,3,6],
# [6,3,7],
# [0,2,4],
# [4,2,6]
# ], dtype=int)

# from exportToPly import write_ply_file
# write_ply_file(vertices,triangles, 'triangle.ply' )


#Part 2:

pointPerRing = 8
nbRing = 3

vertices = []
for i in range (nbRing) :
    for j in range (pointPerRing) :
        x= m.sin(j*np.pi/4) 
        y= m.cos(j*np.pi/4) 
        z= i - 1
        vertices.append([x,y,z])

vertices.append([0.0,0.0,1.0])
vertices.append([0.0,0.0,-1.0])
vertices = np.array(vertices)

triangles = []
for i in range (nbRing-1) :
    for j in range (pointPerRing) :
        triangles.append([j+i*pointPerRing,(j+1)%pointPerRing+i*pointPerRing,j+(i+1)*pointPerRing])
        triangles.append([(j+1)%pointPerRing+i*pointPerRing,(j+1)%pointPerRing+i*pointPerRing+pointPerRing,j+(i+1)*pointPerRing])

for j in range (pointPerRing) :
    triangles.append([25,j,(j+1)%pointPerRing])
    triangles.append([24,j+2*pointPerRing,(j+1)%pointPerRing+2*pointPerRing])
triangles = np.array(triangles)
from exportToPly import write_ply_file
write_ply_file(vertices,triangles, 'triangle.ply' )