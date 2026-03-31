import numpy as np


class Fragment:
    def __init__(self, x, y, depth, world_position, normal, uv, mip_level):
        self.x = x
        self.y = y
        self.depth = depth
        self.world_position = world_position
        self.normal = normal
        self.uv = uv
        self.mip_level = mip_level


def edge_side(p, v0, v1):
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])


def to_screen(ndc_xy, width, height):
    x = (ndc_xy[0] + 1.0) * 0.5 * (width - 1)
    y = (ndc_xy[1] + 1.0) * 0.5 * (height - 1)
    return np.array([x, y])


class GraphicPipeline:
    def __init__(self, width, height, filter_mode="nearest"):
        self.width = width
        self.height = height
        self.filter_mode = filter_mode

        self.image = np.zeros((height, width, 3), dtype=np.float64)
        self.depth_buffer = np.ones((height, width), dtype=np.float64)

        self.texture = None
        self.mipmaps = []

    def build_mipmaps(self, texture):
        mipmaps = [texture]
        current = texture

        # Build level i+1 by averaging 2x2 blocks from level i.
        while current.shape[0] > 1 or current.shape[1] > 1:
            h, w, _ = current.shape
            next_h = max(1, h // 2)
            next_w = max(1, w // 2)
            next_level = np.zeros((next_h, next_w, 3), dtype=np.float64)

            for y in range(next_h):
                for x in range(next_w):
                    y0 = min(2 * y, h - 1)
                    y1 = min(2 * y + 1, h - 1)
                    x0 = min(2 * x, w - 1)
                    x1 = min(2 * x + 1, w - 1)

                    block = np.array(
                        [
                            current[y0, x0],
                            current[y0, x1],
                            current[y1, x0],
                            current[y1, x1],
                        ]
                    )
                    next_level[y, x] = block.mean(axis=0)

            mipmaps.append(next_level)
            current = next_level

        return mipmaps

    def set_texture(self, texture):
        self.texture = texture
        self.mipmaps = self.build_mipmaps(texture)

    def vertex_shader(self, vertex, data):
        world_position = vertex[0:3]
        normal = vertex[3:6]
        uv = vertex[6:8]

        vec = np.array([world_position[0], world_position[1], world_position[2], 1.0])
        clip = data["proj_matrix"] @ (data["view_matrix"] @ vec)

        one_over_w = 1.0 / clip[3]
        ndc = clip[0:3] * one_over_w

        # We store attributes divided by w for perspective-correct interpolation.
        return {
            "ndc": ndc,
            "one_over_w": one_over_w,
            "pos_over_w": world_position * one_over_w,
            "normal_over_w": normal * one_over_w,
            "uv_over_w": uv * one_over_w,
            "uv": uv,
        }

    def estimate_triangle_lod(self, p0, p1, p2, uv0, uv1, uv2):
        tex_h, tex_w, _ = self.mipmaps[0].shape
        tex_size = np.array([tex_w, tex_h], dtype=np.float64)

        screen_edges = [
            np.linalg.norm(p1 - p0),
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p0 - p2),
        ]
        uv_edges = [
            np.linalg.norm((uv1 - uv0) * tex_size),
            np.linalg.norm((uv2 - uv1) * tex_size),
            np.linalg.norm((uv0 - uv2) * tex_size),
        ]

        rho = 0.0
        for i in range(3):
            rho = max(rho, uv_edges[i] / max(screen_edges[i], 1e-6))

        lod = np.log2(max(rho, 1e-6))
        return float(np.clip(lod, 0.0, len(self.mipmaps) - 1))

    def rasterizer(self, v0, v1, v2):
        fragments = []

        p0_ndc = v0["ndc"][0:2]
        p1_ndc = v1["ndc"][0:2]
        p2_ndc = v2["ndc"][0:2]

        area = edge_side(p2_ndc, p0_ndc, p1_ndc)
        if area <= 0.0:
            return fragments

        p0_screen = to_screen(p0_ndc, self.width, self.height)
        p1_screen = to_screen(p1_ndc, self.width, self.height)
        p2_screen = to_screen(p2_ndc, self.width, self.height)

        min_corner = np.floor(np.minimum(np.minimum(p0_screen, p1_screen), p2_screen)).astype(int)
        max_corner = np.ceil(np.maximum(np.maximum(p0_screen, p1_screen), p2_screen)).astype(int)

        min_corner = np.maximum(min_corner, np.array([0, 0]))
        max_corner = np.minimum(max_corner, np.array([self.width - 1, self.height - 1]))

        uv0 = v0["uv_over_w"] / v0["one_over_w"]
        uv1 = v1["uv_over_w"] / v1["one_over_w"]
        uv2 = v2["uv_over_w"] / v2["one_over_w"]
        triangle_lod = self.estimate_triangle_lod(p0_screen, p1_screen, p2_screen, uv0, uv1, uv2)

        for j in range(min_corner[1], max_corner[1] + 1):
            for i in range(min_corner[0], max_corner[0] + 1):
                x = (i + 0.5) / self.width * 2.0 - 1.0
                y = (j + 0.5) / self.height * 2.0 - 1.0
                p = np.array([x, y])

                area0 = edge_side(p, p0_ndc, p1_ndc)
                area1 = edge_side(p, p1_ndc, p2_ndc)
                area2 = edge_side(p, p2_ndc, p0_ndc)

                if area0 >= 0.0 and area1 >= 0.0 and area2 >= 0.0:
                    lambda0 = area1 / area
                    lambda1 = area2 / area
                    lambda2 = area0 / area

                    depth = (
                        lambda0 * v0["ndc"][2]
                        + lambda1 * v1["ndc"][2]
                        + lambda2 * v2["ndc"][2]
                    )

                    one_over_w = (
                        lambda0 * v0["one_over_w"]
                        + lambda1 * v1["one_over_w"]
                        + lambda2 * v2["one_over_w"]
                    )

                    world_position = (
                        lambda0 * v0["pos_over_w"]
                        + lambda1 * v1["pos_over_w"]
                        + lambda2 * v2["pos_over_w"]
                    ) / one_over_w

                    normal = (
                        lambda0 * v0["normal_over_w"]
                        + lambda1 * v1["normal_over_w"]
                        + lambda2 * v2["normal_over_w"]
                    ) / one_over_w

                    uv = (
                        lambda0 * v0["uv_over_w"]
                        + lambda1 * v1["uv_over_w"]
                        + lambda2 * v2["uv_over_w"]
                    ) / one_over_w

                    fragments.append(Fragment(i, j, depth, world_position, normal, uv, triangle_lod))

        return fragments

    def sample_nearest(self, texture, u, v):
        h, w, _ = texture.shape

        u = u - np.floor(u)
        v = v - np.floor(v)

        x = int(round(u * (w - 1)))
        y = int(round((1.0 - v) * (h - 1)))

        return texture[y, x]

    def sample_bilinear(self, texture, u, v):
        h, w, _ = texture.shape

        u = u - np.floor(u)
        v = v - np.floor(v)

        x = u * (w - 1)
        y = (1.0 - v) * (h - 1)

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        tx = x - x0
        ty = y - y0

        c00 = texture[y0, x0]
        c10 = texture[y0, x1]
        c01 = texture[y1, x0]
        c11 = texture[y1, x1]

        c0 = (1.0 - tx) * c00 + tx * c10
        c1 = (1.0 - tx) * c01 + tx * c11

        return (1.0 - ty) * c0 + ty * c1

    def sample_texture(self, u, v, mip_level):
        if self.filter_mode == "nearest":
            return self.sample_nearest(self.mipmaps[0], u, v)

        if self.filter_mode == "bilinear":
            return self.sample_bilinear(self.mipmaps[0], u, v)

        if self.filter_mode == "trilinear":
            l0 = int(np.floor(mip_level))
            l1 = min(l0 + 1, len(self.mipmaps) - 1)
            t = mip_level - l0

            c0 = self.sample_bilinear(self.mipmaps[l0], u, v)
            c1 = self.sample_bilinear(self.mipmaps[l1], u, v)
            return (1.0 - t) * c0 + t * c1

        return self.sample_nearest(self.mipmaps[0], u, v)

    def fragment_shader(self, fragment, data):
        n = fragment.normal / np.linalg.norm(fragment.normal)

        light_dir = data["light_position"] - fragment.world_position
        light_dir = light_dir / np.linalg.norm(light_dir)

        view_dir = data["camera_position"] - fragment.world_position
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Classic Blinn/Phong-style terms (kept simple for student project).
        diffuse = max(np.dot(light_dir, n), 0.0)
        reflect_dir = 2.0 * np.dot(light_dir, n) * n - light_dir
        specular = max(np.dot(reflect_dir, view_dir), 0.0) ** 32

        texture_color = self.sample_texture(fragment.uv[0], fragment.uv[1], fragment.mip_level)

        ambient = 0.2
        color = texture_color * (ambient + 0.8 * diffuse) + 0.15 * specular
        return np.clip(color, 0.0, 1.0)

    def draw(self, vertices, triangles, data):
        self.image.fill(0.0)
        self.depth_buffer.fill(1.0)

        self.set_texture(data["texture"])

        transformed = [self.vertex_shader(v, data) for v in vertices]

        for tri in triangles:
            v0 = transformed[tri[0]]
            v1 = transformed[tri[1]]
            v2 = transformed[tri[2]]

            for fragment in self.rasterizer(v0, v1, v2):
                if fragment.depth < self.depth_buffer[fragment.y, fragment.x]:
                    self.depth_buffer[fragment.y, fragment.x] = fragment.depth
                    self.image[fragment.y, fragment.x] = self.fragment_shader(fragment, data)
