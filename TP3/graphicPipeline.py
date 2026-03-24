import numpy as np

class Fragment:
  def __init__(self, x : int, y : int, depth : float):
    self.x = x
    self.y = y
    self.depth = depth

def edgeSide(p, v0, v1) : 
  pass
  edgeside = (p[0] - v0[0])*(v1[1] - v0[1]) - (p[1] - v0[1])*(v1[0] - v0[0])
  return edgeside

class GraphicPipeline:
  def __init__ (self, width, height):
    self.width = width
    self.height = height
    self.depthBuffer = np.ones((height, width))

  def VertexShader(self, vertex, data) :
    outputVertex = np.zeros_like(vertex)
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]
    w = 1.0


    vec = np.array([[x],[y],[z],[w]])
    vec = np.matmul(data['projMatrix'],np.matmul(data['viewMatrix'],vec))


    outputVertex[0] = vec[0]/vec[3]
    outputVertex[1] = vec[1]/vec[3]
    outputVertex[2] = vec[2]/vec[3]

    return outputVertex


  def Rasterizer(self, v0, v1, v2) :
    fragments = []

    area = edgeSide(v0, v1, v2)
    if area == 0:
        return fragments

    for j in range(0, self.height) : 
      for i in range(0, self.width) :
        x = 2*(i+0.5)/self.width - 1
        y = 2*(j+0.5)/self.height - 1
        p = [x, y]
        lambda0 = edgeSide(p,v1,v2)/ area
        lambda1 = edgeSide(p,v2,v0) / area
        lambda2 = edgeSide(p,v0,v1) / area
        p1 = Fragment(i,j,lambda0*v0[2]+lambda1*v1[2]+lambda2*v2[2])
        if (edgeSide(p,v0,v1) > 0 and edgeSide(p,v1,v2) > 0 and edgeSide(p,v2,v0) > 0) or (edgeSide(p,v0,v1) < 0 and edgeSide(p,v1,v2) < 0 and edgeSide(p,v2,v0) < 0) :
            fragments.append(p1)
        
        #if inside 
          #emit a fragment
        #pass
    
    return fragments

  def draw(self, vertices, triangles, data):
    self.newVertices = np.zeros_like(vertices)

    for i in range(vertices.shape[0]) :
       self.newVertices[i] = self.VertexShader(vertices[i],data)


    fragments = []
    for t in triangles :
      #call the rasterizer the triangle t
      v0 = (self.newVertices[t[0]][0], self.newVertices[t[0]][1], self.newVertices[t[0]][2])
      v1 = (self.newVertices[t[1]][0], self.newVertices[t[1]][1], self.newVertices[t[1]][2])
      v2 = (self.newVertices[t[2]][0], self.newVertices[t[2]][1], self.newVertices[t[2]][2])
      fragments += self.Rasterizer(v0, v1, v2)
      #pass

    for f in fragments:
      #todo Process each fragment using the depth buffer
      i = f.x
      j = f.y
      if 0 <= i < self.width and 0 <= j < self.height:
        if f.depth < self.depthBuffer[j][i]:
          self.depthBuffer[j][i] = f.depth
      #pass