# Project: Texture Mipmapping and Filtering

This folder contains a student-level implementation based on TP4.

Implemented methods:
- `nearest`: nearest-neighbor texture filtering (base texture only)
- `bilinear`: bilinear filtering (base texture only)
- `trilinear`: bilinear + mipmaps + interpolation between two mip levels

## Run

From repo root:

```bash
python3 Project/main.py
```

Outputs created in `Project/`:
- `render_nearest.png`
- `render_bilinear.png`
- `render_trilinear.png`
- `comparison_filters.png`

The script reuses mesh and texture from `TP4/TP4/suzanne.ply` and `TP4/TP4/suzanne.png`.
