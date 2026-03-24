import numpy as np

class Camera:
  def __init__(self, position, lookAt, up, right) :
    self.position = position
    self.lookAt = lookAt
    self.up = up
    self.right = right

  def getMatrix(self):
    matrix = np.array([[1,0,0,-self.position[0]],[0,1,0,-self.position[1]],[0,0,1,-self.position[2]],[0,0,0,1]])
    matrix1 = np.array([[self.right[0],self.right[1],self.right[2],0],[self.up[0],self.up[1],self.up[2],0],[self.lookAt[0],self.lookAt[1],self.lookAt[2],0],[0,0,0,1]])
    matrix = matrix1 @ matrix
    return matrix
    #pass 
