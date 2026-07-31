# Deterministic operation catalog

The backend always executes the existing pipeline in this exact order:

1. saturation/color threshold
2. optional invert
3. morphological open
4. morphological close
5. signed dilation or erosion (`morph_pixels`)
6. 8-connected component filtering

- `open_kernel`: removes small foreground noise but may break thin lines.
- `close_kernel`: fills small concavities and short gaps while broadly preserving line width; excessive values can bridge separate devices or erase holes.
- positive `morph_pixels`: dilates and thickens all foreground edges.
- negative `morph_pixels`: erodes and thins all foreground edges.
- `min_component_area`: removes small connected foreground components.
- `region_mode=largest`: keeps only the largest foreground component.

Candidate parameters are backend-owned. Select only by candidate ID.
