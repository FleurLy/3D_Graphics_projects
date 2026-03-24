import numpy as np
import math as m

class Projection:
  def __init__(self, near,far,fov,aspectRatio) :
    self.nearPlane = near
    self.farPlane = far
    self.fov = fov
    self.aspectRatio = aspectRatio

  def getMatrix(self) :
    s = 1/m.tan(self.fov/2)
    matrix =  np.array([[s/self.aspectRatio, 0, 0, 0], [0, s, 0, 0], [0, 0, self.farPlane/(self.farPlane - self.nearPlane), -(self.farPlane*self.nearPlane)/(self.farPlane - self.nearPlane)], [0, 0, 1, 0]])
    return matrix
    #pass