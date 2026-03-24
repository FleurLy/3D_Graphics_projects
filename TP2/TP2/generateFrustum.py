from exportToPly import write_ply_file
from camera import Camera
from projection import Projection
import numpy as np


def generateFrustum(cam : Camera, proj: Projection ) :
  n = cam.lookAt * proj.nearPlane 
  f = cam.lookAt * proj.farPlane
  up = cam.up * np.tan(proj.fov/2.0)  / proj.aspectRatio
  right = cam.right * np.tan(proj.fov/2.0)

  v0= n - up * proj.nearPlane - right * proj.nearPlane + cam.position
  v1= n - up * proj.nearPlane + right * proj.nearPlane + cam.position
  v2= n + up * proj.nearPlane - right * proj.nearPlane + cam.position
  v3= n + up * proj.nearPlane + right * proj.nearPlane + cam.position

  v4= f - up * proj.farPlane - right * proj.farPlane + cam.position
  v5= f - up * proj.farPlane + right * proj.farPlane + cam.position
  v6= f + up * proj.farPlane - right * proj.farPlane + cam.position
  v7= f + up * proj.farPlane + right * proj.farPlane + cam.position

  vertices = np.array([
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7
  ])


  triangles = np.array([
    [1,0,2],
    [3,1,2],
    [4,5,6],
    [5,7,6],
    [0,1,4],
    [4,1,5],
    [2,6,3],
    [3,6,7],
    [0,6,2],
    [4,6,0],
    [1,3,7],
    [5,1,7]
  ], dtype=int)


  write_ply_file(vertices,triangles, 'frustum.ply' )


def generateFrustumCameraSpace(proj: Projection):
  n = np.array([0,0,1]) * proj.nearPlane 
  f = np.array([0,0,1]) * proj.farPlane
  up = np.array([0,1,0]) * np.tan(proj.fov/2.0)  / proj.aspectRatio
  right = np.array([1,0,0]) * np.tan(proj.fov/2.0)

  v0= n - up * proj.nearPlane - right * proj.nearPlane 
  v1= n - up * proj.nearPlane + right * proj.nearPlane
  v2= n + up * proj.nearPlane - right * proj.nearPlane
  v3= n + up * proj.nearPlane + right * proj.nearPlane

  v4= f - up * proj.farPlane - right * proj.farPlane
  v5= f - up * proj.farPlane + right * proj.farPlane
  v6= f + up * proj.farPlane - right * proj.farPlane
  v7= f + up * proj.farPlane + right * proj.farPlane

  vertices = np.array([
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7
  ])


  triangles = np.array([
    [1,0,2],
    [3,1,2],
    [4,5,6],
    [5,7,6],
    [0,1,4],
    [4,1,5],
    [2,6,3],
    [3,6,7],
    [0,6,2],
    [4,6,0],
    [1,3,7],
    [5,1,7]
  ], dtype=int)


  write_ply_file(vertices,triangles, 'frustumCameraSpace.ply' )


def generateProjectedFrustum():

  vertices = np.array([
    [-1,-1,0],
    [+1,-1,0],
    [-1,+1,0],
    [+1,+1,0],
    [-1,-1,1],
    [+1,-1,1],
    [-1,+1,1],
    [+1,+1,1],
  ])


  triangles = np.array([
    [1,0,2],
    [3,1,2],
    [4,5,6],
    [5,7,6],
    [0,1,4],
    [4,1,5],
    [2,6,3],
    [3,6,7],
    [0,6,2],
    [4,6,0],
    [1,3,7],
    [5,1,7]
  ], dtype=int)

  write_ply_file(vertices,triangles, 'frustumProjected.ply' )