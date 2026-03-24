import numpy as np


class GraphicPipeline:

  def __init__ (self):
    self.newVertices = ()
    pass

  def VertexShader(self, vertex, data) :
    outputVertex = np.zeros_like(vertex)
    outputVertex = np.array([vertex[0], vertex[1], vertex[2], 1.0])
    outputVertex = data['viewMatrix'] @ outputVertex
    outputVertex = data['projMatrix'] @ outputVertex

    return outputVertex[:3]/outputVertex[3]



  def draw(self, vertices, triangles, data):
    self.newVertices = np.zeros_like(vertices)

    for i in range(vertices.shape[0]) :
      self.newVertices[i] = self.VertexShader(vertices[i],data)
      #pass
