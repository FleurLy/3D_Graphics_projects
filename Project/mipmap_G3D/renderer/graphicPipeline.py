import numpy as np
from mipmap import (build_mipmaps, sample_nearest, sample_bilinear,
                    sample_trilinear, sample_anisotropic,
                    compute_lod_accurate)



def edgeSide(p, v0, v1):

    return ((p[0]  - v0[0]) * (v1[1] - v0[1])
          - (p[1]  - v0[1]) * (v1[0] - v0[0]))


class Fragment:
    """
    Un Fragment = un pixel candidat emis par le rasterizer.

    Attributs added:
      lod               : LOD isotropique calcule
      dudx,dvdx,dudy,dvdy : derivees partielles UV (pour anisotropique)

    """

    def __init__(self, x, y, depth, interpolated_data, lod=0.0,
                 dudx=0.0, dvdx=0.0, dudy=0.0, dvdy=0.0):
        self.x                = x
        self.y                = y
        self.depth            = depth
        self.interpolated_data = interpolated_data
        self.lod              = lod
        self.dudx             = dudx
        self.dvdx             = dvdx
        self.dudy             = dudy
        self.dvdy             = dvdy
        self.output           = np.zeros(3, dtype=float)


class GraphicPipeline:
    """
    Pipeline graphique complet avec filtres de texture configurables.
    """

    def __init__(self, width, height,
                 filter_mode="trilinear",
                 downsample_filter="box"):
        """
        Parametres  added:
          
          filter_mode        : filtre de sampling ("nearest", "bilinear",
                               "trilinear", "anisotropic")
          downsample_filter  : filtre de downsampling pour la pyramide
                               ("box", "gaussian", "lanczos")
        """
        self.width             = width
        self.height            = height
        self.filter_mode       = filter_mode
        self.downsample_filter = downsample_filter

        self.image       = np.zeros((height, width, 3), dtype=float)
        self.depthBuffer = np.ones((height, width), dtype=float)
        self.mips        = None

   
   
    def VertexShader(self, vertex, data):
        """
        Transforme un sommet monde -> NDC et calcule N, V, L.

        Entree  (vertex, 8 valeurs) :
          [0:3]  position monde
          [3:6]  normale
          [6:8]  UV texture

        Sortie (18 valeurs) :
          [0:3]  position NDC
          [3:6]  normale monde
          [6:9]  vecteur V (vue)
          [9:12] vecteur L (lumiere)
          [12:14] UV
          [14:16] NDC (x, y) -> LOD
          [16:18] position ecran en pixels (px, py) -> LOD precis
        """
        outputVertex = np.zeros(19, dtype=float)

        vec  = np.array([vertex[0], vertex[1], vertex[2], 1.0])
        vec = np.matmul(data['projMatrix'],np.matmul(data['viewMatrix'],vec))
        w    = vec[3]

        outputVertex[0] = vec[0] / w
        outputVertex[1] = vec[1] / w
        outputVertex[2] = vec[2] / w

        outputVertex[3] = vertex[3] 
        outputVertex[4] = vertex[4]
        outputVertex[5] = vertex[5]

        outputVertex[6]  = data['cameraPosition'][0] - vertex[0]
        outputVertex[7]  = data['cameraPosition'][1] - vertex[1]
        outputVertex[8]  = data['cameraPosition'][2] - vertex[2]

        outputVertex[9]  = data['lightPosition'][0] - vertex[0]
        outputVertex[10] = data['lightPosition'][1] - vertex[1]
        outputVertex[11] = data['lightPosition'][2] - vertex[2]

        outputVertex[12] = vertex[6]   
        outputVertex[13] = vertex[7]   

        outputVertex[14] = outputVertex[0]      
        outputVertex[15] = outputVertex[1]      

        # Position ecran en pixels (pour derivees partielles precises)
        outputVertex[16] = (outputVertex[0] + 1.0) * 0.5 * self.width
        outputVertex[17] = (outputVertex[1] + 1.0) * 0.5 * self.height
        

        outputVertex[18] = w

        return outputVertex



    def Rasterizer(self, v0, v1, v2):

        fragments = []

        # culling backface
        area = edgeSide(v0, v1, v2)
        if area <= 0:
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


        # ---- Calcul LOD precis (4 derivees partielles) ----------
        u0, v_0 = v0[12], v0[13]
        u1, v_1 = v1[12], v1[13]
        u2, v_2 = v2[12], v2[13]

        dx_01 = v1_image[0] - v0_image[0]
        dy_01 = v1_image[1] - v0_image[1]

        dx_02 = v2_image[0] - v0_image[0]
        dy_02 = v2_image[1] - v0_image[1]

        du_01 = u1 - u0       
        dv_01 = v_1 - v_0

        du_02 = u2 - u0        
        dv_02 = v_2 - v_0

        # Resolution du systeme 2x2 pour dU/dx, dU/dy, dV/dx, dV/dy
        det = dx_01 * dy_02 - dx_02 * dy_01
        if abs(det) > 1e-10:
            dudx = (du_01 * dy_02 - du_02 * dy_01) / det
            dudy = (dx_01 * du_02 - dx_02 * du_01) / det
            dvdx = (dv_01 * dy_02 - dv_02 * dy_01) / det
            dvdy = (dx_01 * dv_02 - dx_02 * dv_01) / det
        else:
            dudx = dvdx = dudy = dvdy = 0.0

        if self.mips is not None:
            tex_size = max(self.mips[0].shape[0], self.mips[0].shape[1])
        else:
            tex_size = 1

        lod = compute_lod_accurate(dudx, dvdx, dudy, dvdy, tex_size)

        # AABBbox du triangle en pixels
        for j in range(A[1], B[1] + 1):
            for i in range(A[0], B[0] + 1):

                x = (i + 0.5) / self.width  * 2.0 - 1
                y = (j + 0.5) / self.height * 2.0 - 1
                p = np.array([x, y])

                area0 = edgeSide(p,v0,v1)
                area1 = edgeSide(p,v1,v2)
                area2 = edgeSide(p,v2,v0)

                
                if (area0 >= 0 and area1 >= 0 and area2 >= 0) : 
                    
                    #Computing 2d barricentric coordinates
                    lambda0 = area1/area
                    lambda1 = area2/area
                    lambda2 = area0/area


                    iz0 = 1.0 / v0[18] if abs(v0[18]) > 1e-8 else 1.0
                    iz1 = 1.0 / v1[18] if abs(v1[18]) > 1e-8 else 1.0
                    iz2 = 1.0 / v2[18] if abs(v2[18]) > 1e-8 else 1.0

                    denom = lambda0*iz0 + lambda1*iz1 + lambda2*iz2

                    if abs(denom) < 1e-12:
                        continue
                    z = z = (lambda0*iz0*v0[18] + lambda1*iz1*v1[18] + lambda2*iz2*v2[18]) / denom

                    w0 = (lambda0 * iz0) / denom
                    w1 = (lambda1 * iz1) / denom
                    w2 = (lambda2 * iz2) / denom

                    l = v0.shape[0]
                    #interpolating
                    interpolated_data = v0[3:l] * w0 + v1[3:l] * w1 + v2[3:l] * w2

                    #Emiting Fragment
                    fragments.append(Fragment(i, j, z, interpolated_data, lod,
                                              dudx, dvdx, dudy, dvdy))

        return fragments

    
    
    def fragmentShader(self, fragment, data):
        """
        Calcule la couleur finale : eclairage de Phong + texture.

        Le filtre de texture est selectionne via self.filter_mode :
          "nearest"     -> sample_nearest sur le niveau floor(LOD)
          "bilinear"    -> sample_bilinear sur le niveau floor(LOD)
          "trilinear"   -> sample_trilinear (standard)
          "anisotropic" -> sample_anisotropic (multi-tap directionnel)
        """
        N = fragment.interpolated_data[0:3]
        n_len = np.linalg.norm(N)
        if n_len < 1e-8:
            fragment.output = np.zeros(3); return
        N = N / n_len

        V = fragment.interpolated_data[3:6]
        v_len = np.linalg.norm(V)
        if v_len < 1e-8:
            fragment.output = np.zeros(3); return
        V = V / v_len

        L = fragment.interpolated_data[6:9]
        l_len = np.linalg.norm(L)
        if l_len < 1e-8:
            fragment.output = np.zeros(3); return
        L = L / l_len

        R        = 2.0 * np.dot(N, L) * N - L
        ambient  = 1.0
        diffuse  = max(np.dot(N, L), 0.0)
        specular = pow(max(np.dot(R, V), 0.0), 64)

        ka = 0.1 
        kd = 0.9
        ks = 0.3
        phong = ka * ambient + kd * diffuse + ks * specular
        phong = np.ceil(phong * 4 + 1) / 6.0 

        u = fragment.interpolated_data[9]
        v = fragment.interpolated_data[10]

        # echantillonnage de la texture avec le filtre selectionne
        fm = self.filter_mode

        if fm == "nearest":
            lvl = min(int(np.floor(fragment.lod)), len(self.mips) - 1)
            tex_color = sample_nearest(self.mips[lvl], u, v)

        elif fm == "bilinear":
            lvl = min(int(np.floor(fragment.lod)), len(self.mips) - 1)
            tex_color = sample_bilinear(self.mips[lvl], u, v)

        elif fm == "trilinear":
            tex_color = sample_trilinear(self.mips, u, v, fragment.lod)

        elif fm == "anisotropic":
            tex_color = sample_anisotropic(
                self.mips, u, v, fragment.lod,
                fragment.dudx, fragment.dvdx,
                fragment.dudy, fragment.dvdy,
                max_samples=8)
        else:
            tex_color = sample_trilinear(self.mips, u, v, fragment.lod)

        fragment.output = np.array([phong, phong, phong]) * tex_color

 
 
    def draw(self, vertices, triangles, data):
        """
        Execute le pipeline complet.

          1. Construction de la pyramide MIP (pré-calcul)
          2. Vertex Shader sur tous les sommets
          3. Rasterization de chaque triangle
          4. Fragment Shader + Depth Test sur chaque fragment
        """
        print(f"[INFO] Construction pyramide MIP "
              f"(downsample={self.downsample_filter})...")
        self.mips = build_mipmaps(data['texture'], self.downsample_filter)
        print(f"[INFO] {len(self.mips)} niveaux MIP generes "
              f"(de {self.mips[0].shape[1]}x{self.mips[0].shape[0]} "
              f"a {self.mips[-1].shape[1]}x{self.mips[-1].shape[0]})")
        print(f"[INFO] Filtre de sampling : {self.filter_mode}")

        nb_vertices = vertices.shape[0]
        self.newVertices = np.zeros((nb_vertices, 19), dtype=float)
        for i in range(nb_vertices):
            self.newVertices[i] = self.VertexShader(vertices[i], data)

        all_fragments = []
        for tri in triangles:
            v0 = self.newVertices[tri[0]]
            v1 = self.newVertices[tri[1]]
            v2 = self.newVertices[tri[2]]
            all_fragments.extend(self.Rasterizer(v0, v1, v2))


        for f in all_fragments:
            self.fragmentShader(f, data)
            if self.depthBuffer[f.y, f.x] > f.depth:
                self.depthBuffer[f.y, f.x] = f.depth
                self.image[f.y, f.x]       = f.output


#