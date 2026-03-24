import numpy as np


class Fragment:
    def __init__(self, x : int, y : int, depth : float, interpolated_data ):
        self.x = x
        self.y = y
        self.depth = depth
        self.interpolated_data = interpolated_data
        self.output = []

def edgeSide(p, v0, v1) : 
    return (p[0]-v0[0])*(v1[1]-v0[1]) - (p[1]-v0[1])*(v1[0]-v0[0])

def edgeSide3D(p,v0,v1) :
    return np.linalg.norm(np.cross(p[0:3]-v0[0:3],v1[0:3]-v0[0:3]))

class GraphicPipeline:
    def __init__ (self, width, height):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3))
        self.depthBuffer = np.ones((height, width))


    def VertexShader(self, vertex, data) :
        n = len(vertex)
        outputVertex = np.zeros(n+9)
        # for i in range(9):
        #     outputVertex.append(0)
        

        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        w = 1.0

        vec = np.array([[x],[y],[z],[w]])

        V = np.array(data['cameraPosition']).flatten() - np.array(vec[:3]).flatten()
        V = V / np.linalg.norm(V)
        L = np.array(data['lightPosition']).flatten() - np.array(vec[:3]).flatten()
        L = L / np.linalg.norm(L)
        N = np.array([vertex[3], vertex[4], vertex[5]])
        N = N / np.linalg.norm(N)

        vec = np.matmul(data['projMatrix'],np.matmul(data['viewMatrix'],vec))

        outputVertex[0] = vec[0]/vec[3]
        outputVertex[1] = vec[1]/vec[3]
        outputVertex[2] = vec[2]/vec[3]

        outputVertex[n:n+3] = N
        outputVertex[n+3:n+6] = L
        outputVertex[n+6:n+9] = V

        return outputVertex


    def Rasterizer(self, v0, v1, v2, data) :
        fragments = []

        #culling back face
        area = edgeSide(v0,v1,v2)
        area3D = edgeSide3D(v0,v1,v2)
        if area < 0 :
            return fragments
        
        
        #AABBox computation
        #compute vertex coordinates in screen space
        v0_image = np.array([0,0])
        v0_image[0] = (v0[0]+1.0)/2.0 * self.width 
        v0_image[1] = ((v0[1]+1.0)/2.0) * self.height 

        v1_image = np.array([0,0])
        v1_image[0] = (v1[0]+1.0)/2.0 * self.width 
        v1_image[1] = ((v1[1]+1.0)/2.0) * self.height 

        v2_image = np.array([0,0])
        v2_image[0] = (v2[0]+1.0)/2.0 * self.width 
        v2_image[1] = (v2[1]+1.0)/2.0 * self.height 

        #compute the two point forming the AABBox
        A = np.min(np.array([v0_image,v1_image,v2_image]), axis = 0)
        B = np.max(np.array([v0_image,v1_image,v2_image]), axis = 0)

        #cliping the bounding box with the borders of the image
        max_image = np.array([self.width-1,self.height-1])
        min_image = np.array([0.0,0.0])

        A  = np.max(np.array([A,min_image]),axis = 0)
        B  = np.min(np.array([B,max_image]),axis = 0)
        
        #cast bounding box to int
        A = A.astype(int)
        B = B.astype(int)
        #Compensate rounding of int cast
        B = B + 1

        #for each pixel in the bounding box
        for j in range(A[1], B[1]) : 
           for i in range(A[0], B[0]) :
                x = (i+0.5)/self.width * 2.0 - 1 
                y = (j+0.5)/self.height * 2.0 - 1

                p = np.array([x,y])
                
                area0 = edgeSide(p,v0,v1)
                area1 = edgeSide(p,v1,v2)
                area2 = edgeSide(p,v2,v0)

                #test if p is inside the triangle
                if (area0 >= 0 and area1 >= 0 and area2 >= 0) : 
                    
                    #Computing 2d barricentric coordinates
                    lambda0 = area1/area
                    lambda1 = area2/area
                    lambda2 = area0/area
                    
                    #one_over_z = lambda0 * 1/v0[2] + lambda1 * 1/v1[2] + lambda2 * 1/v2[2]
                    #z = 1/one_over_z
                    
                    z = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]

                    p = np.array([x,y,z])
                    
                    #Recomputing the barricentric coordinaties for vertex interpolation
                    
                    #interpolating Vertex data
                    
                    
                    #Emiting Fragment
                    interpolated_data = (
                                            lambda0 * v0 +
                                            lambda1 * v1 +
                                            lambda2 * v2
                                        )
                    fragments.append(Fragment(i,j,z, interpolated_data))

        return fragments
    
    def fragmentShader(self,fragment,data):
        color = np.array([1,1,1])

        N = fragment.interpolated_data[8:11]
        L = fragment.interpolated_data[11:14]
        V = fragment.interpolated_data[14:17]

        N = N / np.linalg.norm(N)
        L = L / np.linalg.norm(L)
        V = V / np.linalg.norm(V)

        diffuse = max(np.dot(L, N), 0)

        R = 2 * np.dot(L, N) * N - L
        specular = max(np.dot(R, V), 0)**32

        fragment.output = color * (0.5 + 0.9 * diffuse + 0.3 * specular)

        #pass

    def draw(self, vertices, triangles, data):
        #Calling vertex shader
        self.newVertices = np.zeros((vertices.shape[0], 17))

        for i in range(vertices.shape[0]) :
            self.newVertices[i] = self.VertexShader(vertices[i],data)
        
        fragments = []
        #Calling Rasterizer
        for i in triangles :
            fragments.extend(self.Rasterizer(self.newVertices[i[0]], self.newVertices[i[1]], self.newVertices[i[2]], data))
        
        for f in fragments:

            #TODO call fragmentShader on f
            self.fragmentShader(f,data)
            
            #depth test
            if self.depthBuffer[f.y][f.x] > f.depth : 
                self.depthBuffer[f.y][f.x] = f.depth
                self.image[f.y][f.x] = f.output
                
                # Store the fragment in the image
                
                
            

# from PIL import Image
# from numpy import asarray
# # Open the image form working directory
# image = asarray(Image.open('suzanne.png'))

# data = dict([
# ('viewMatrix',cam.getMatrix()),
# ('projMatrix',proj.getMatrix()),
# ('cameraPosition',position),
# ('lightPosition',lightPosition),
# ('texture', image),
# ])

# def sample(texture, u, v) :
#   u = int(u * texture.shape[0])
#   v = int((1-v) * texture.shape[1])
#   return texture[v,u] / 255.0
