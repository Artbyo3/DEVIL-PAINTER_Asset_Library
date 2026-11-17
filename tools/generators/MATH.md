# Mathematical Specifications — DEVIL-PAINTER Texture & Alpha Generators (All 81 Types)

Every generator in `tools/generators/types/` exposes `SPEC: GeneratorSpec` (`render_fn`, `height_fn`, `normal_source`, `normal_strength`, `tileable`, `tile_axes`). All periodic textures live natively on the unit 2-torus $\mathbb{T}^2 = [0, 1)^2$ via periodic grid wrapping (`% freq`, toroidal Voronoi offsets $\{-1,0,1\}^2$, circular convolution via `np.roll`), and all height fields feed `common.height_to_normal` (3×3 periodic Sobel central finite differences, OpenGL `+Y` up tangent-space normals).

---

## Shared Mathematical Primitives (`tools/generators/common.py`)

- **Toroidal Grid (`make_uv_grid`)**: $(u_j, v_i) = (j / S,\, i / S) \in [0,1)^2$ with `endpoint=False`, guaranteeing exact periodicity $f(u+1, v) = f(u, v+1) = f(u,v)$.
- **Periodic Quintic Gradient Noise (`periodic_gradient_perlin`)**: Toroidal lattice $(F_x \times F_y)$ with random unit gradient vectors $\mathbf{g}_{i,j} = (\cos \theta_{i,j}, \sin \theta_{i,j})$ indexed modulo $(F_x, F_y)$ and Perlin's $C^2$ quintic Hermite curve $s(t) = 6t^5 - 15t^4 + 10t^3$ (Perlin 2002, *Improving Noise*, ACM SIGGRAPH).
- **Periodic Multi-Octave fBm (`periodic_fbm`)**: Sum over $K$ octaves clamped to Nyquist-safe lattice frequencies $\le 128$ with persistence $\rho$: `fbm`, `turbulence` ($|2n-1|$), `billow` ($1 - |2n-1|$), `ridged` ($(1 - |2n-1|)^2$), and **Swiss** multifractal (derivative-advected ridged signal weighted by running signal energy $w_{k+1} = \mathrm{clip}(1.8 s_k, 0.1, 1.0)$; Musgrave / Gaffer derivative-weighted multifractal).
- **Periodic Worley / Voronoi & Lloyd CVT (`periodic_voronoi`, `periodic_lloyd_voronoi`)**: 9-neighbor toroidal cellular search under Euclidean ($\ell_2$), Manhattan ($\ell_1$), or Chebyshev ($\ell_\infty$) metrics returning normalized $(F_1, F_2 - F_1, \text{cell\_id}, \Delta x, \Delta y)$ (Worley 1996, SIGGRAPH). `periodic_lloyd_voronoi` iteratively relaxes seeds $\mathbf{p}_k \leftarrow 0.85\,\bar{\mathbf{x}}_k + 0.15\,\mathbf{p}_k$ toward cell centroids on a discrete toroidal grid (Lloyd 1982, IEEE Trans. Info. Theory).
- **Periodic Weave-Matrix Cloth Elevation (`woven_fabric_field`)**: Given a $(Q, P)$ binary draft matrix $M$ ($1=$ warp over, $0=$ weft over), computes continuous run lengths $L_{\text{warp}}, L_{\text{weft}}$ and phase offsets along each float so multi-pick floats form smooth continuous arches $\sin^{\alpha}(\pi s)$ over Peirce-inspired elliptical cross-sections $\sqrt{1 - (2\delta/r)^2}$ with helical 2-ply twist modulation (Peirce 1937, *J. Textile Inst.* 28:T45–T96).
- **Periodic Sparse Gabor Noise (`periodic_gabor_noise`)**: Toroidal sparse convolution $\sum_i a_i \exp(-\pi a^2 \|\mathbf{x}-\mathbf{x}_i\|^2) \cos(2\pi f_0 (\mathbf{x}-\mathbf{x}_i)\cdot(\cos\omega_0, \sin\omega_0) + \phi_i)$ for bandpass isotropic or anisotropic oriented textures (Lagae et al. 2009, ACM SIGGRAPH).
- **Periodic Curl Velocity Field (`periodic_curl_noise`)**: Divergence-free 2D advection $\mathbf{v} = (\partial \psi / \partial v,\, -\partial \psi / \partial u)$ normalized by its 95th percentile magnitude from a periodic scalar potential $\psi$ (Bridson et al. 2007, ACM SIGGRAPH).
- **Gray-Scott Reaction-Diffusion (`gray_scott_rd`)**: Explicit Euler solver on $\mathbb{T}^2$ using the 9-point isotropic finite-difference Laplacian ($0.20$ axial + $0.05$ diagonal $- 1.0$ center): $\partial_t U = D_u \nabla^2 U - UV^2 + F(1-U)$, $\partial_t V = D_v \nabla^2 V + UV^2 - (F+k)V$ with $D_u=0.16, D_v=0.08$, periodic padded bicubic upsampling (Pearson 1993, *Science* 261:189–192).
- **OpenGL Sobel Normal Converter (`height_to_normal`)**: Converts normalized $h \in [0,1]$ via periodic 3×3 Sobel operators with $+X$ right, $+Y$ up (OpenGL/Blender), returning $\mathbf{n} = \mathrm{normalize}(-s\,\partial_u h,\,-s\,\partial_v^{\uparrow} h,\,1)$ encoded to RGB $[0,255]$.

---

## Family 1: Cloth & Apparel Textures (17 Types — 17 Modified, 0 Unchanged)

### 1. `denim-twill` (`tools/generators/types/denim_twill.py`)
- **Construction**: Built with `woven_fabric_field` over a $4\times 4$ **3/1 twill weave matrix** $M_{j,i} = \mathbf{1}[(i + \mathrm{step}\cdot j)\bmod 4 < 3]$, switching between Z-twill ($\mathrm{step}=+1$, right-hand twill) on even variants and S-twill ($\mathrm{step}=-1$, left-hand twill) on odd variants. Ring-spun warp yarn slub `slub = periodic_fbm(u, v, 24, 3, 0.55, rng, freqy=4)` modulates `warp_rad = clip(0.88 + 0.14*(slub - 0.5), 0.78, 1.0)` with `threads = 24 + (variant % 5) * 4`, `float_arch=0.38`, `ply_twist=0.14`.
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{weave\_h} + 0.08\,\text{fuzz})$, shaded via `shade_cloth_surface`; albedo contrasts indigo warp floats (`warp_over`) against ecru weft dots exposed on 1 of 4 picks.
- **Source**: Peirce (1937) *The Geometry of Cloth Structure* (*J. Textile Inst.* 28:T45–T96); standard $3\times 1$ RHT/LHT denim weave specification.

### 2. `jersey-knit` (`tools/generators/types/jersey_knit.py`)
- **Construction**: Explicit periodic single-jersey weft-knitted V-loop yarn centerline geometry evaluated across $n_{\text{overlap}} \in \{-1,0,1\}$ vertical course neighbors on a `(wales, courses)` lattice (`wales = 14 + (variant % 5)*4`, `courses = round(wales * 1.35)`). Left and right legs follow $x_L(t) = 0.21 + 0.14\sin(0.85\pi t) + 0.04(1-\cos\pi t)$ and $x_R(t) = 1 - x_L(t)$ with depth $z_{\text{leg}}(t) = 0.52 + 0.48\sin(\pi t)$ and elliptical yarn cross-section $\sqrt{1-(d/r_{\text{yarn}})^2}$ ($r_{\text{yarn}}=0.205$) plus helical ply twist (`sin((5t \pm 4\delta)2\pi)`) and interlocking head/sinker crowns (`crown_y`).
- **Height Field (`native`)**: Upper envelope $\max_{\Delta y \in \{-1,0,1\}} \max(p_L z_{\text{leg}},\, p_R z_{\text{leg}},\, p_{\text{crown}})$ normalized plus `0.12 * fuzz`.
- **Source**: Chamberlain (1926) / Leaf & Glaskin (1955) *The Geometry of a Plain Knitted Loop* (*J. Textile Inst.* 46:T587–T605); Yuksel et al. (2012) *Stitch Meshes* (ACM SIGGRAPH).

### 3. `cable-knit` (`tools/generators/types/cable_knit.py`)
- **Construction**: Alternating vertical cable columns (`cols = 4 + (variant % 4)`, `period = 1/cols`) flanked by interlocking purl-wale troughs. Within each cable band ($|dx| < 0.38$), two intertwined multi-strand rope helices $s_1(v) = A\sin(2\pi k v)$, $s_2(v) = -A\sin(2\pi k v)$ cross smoothly over/under with crossing depth $z_1 = 0.65 + 0.35\cos(2\pi k v)$, $z_2 = 0.65 - 0.35\cos(2\pi k v)$, cylindrical profile $\sqrt{1 - (|dx - s_i|/w)^2}$, and secondary helical ply twist.
- **Height Field (`native`)**: $h = \mathrm{norm}(\max(\text{braid}\cdot\mathbf{1}_{\text{cable}},\, 0.48\,\text{purl}) + 0.10\,\text{fuzz})$.
- **Source**: Yuksel et al. (2012) *Stitch Meshes for Modeling Knitted Clothing with Yarn-level Detail* (ACM SIGGRAPH) & *first principles* helical rope crossover profile.

### 4. `tartan-plaid` (`tools/generators/types/tartan_plaid.py`)
- **Construction**: Symmetrical palindromic tartan sett generated by reflecting a 4-color half-sett vector `half` around its pivot (`full = half + half[-2:0:-1]`) of length `n_sett`. Evaluated on a $2\times 2$ balanced twill matrix (`[[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]]`) via `woven_fabric_field` with `threads` rounded to a multiple of `4` and `n_sett` (`repeats * n_sett`), assigning authentic warp vs. weft sett color on `warp_over`.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.85\,\text{weave\_h} + 0.15\,\text{fuzz})$ feeding `shade_cloth_surface`.
- **Source**: Scottish Register of Tartans palindromic sett symmetry standard; Peirce (1937) 2/2 twill weave geometry.

### 5. `houndstooth-wool` (`tools/generators/types/houndstooth_wool.py`)
- **Construction**: Authentic 8-thread yarn-dyed **4 dark / 4 light warp & weft color repeat** (`iu % 8 < 4`, `iv % 8 < 4`) woven over a $2\times 2$ balanced twill matrix (`twill_2x2`) via `woven_fabric_field` (`threads = repeats * 8`, `ply_twist=0.14`), producing the classic broken-check dogtooth optical silhouette exclusively from yarn intersection parity (`np.where(warp_over, warp_dark, weft_dark)`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.82\,\text{weave\_h} + 0.18\,\text{fuzz})$ shaded with woolen sheen (`sheen_amt=15.0`).
- **Source**: Traditional wool weaving color-and-weave theory (4-dark/4-light 2/2 twill); Peirce (1937) weave geometry.

### 6. `quilted-puffer` (`tools/generators/types/quilted_puffer.py`)
- **Construction**: Rotated periodic $45^\circ$ diamond grid ($d_1 = (u+v)\cdot S \bmod 1$, $d_2 = (u-v)\cdot S \bmod 1$, integer $S$ preserving $\mathbb{T}^2$ periodicity) with pneumatic cushion pillow $p = (\sin(\pi d_1)\sin(\pi d_2))^{0.52}$, seam-pinched tension folds radiating orthogonally from the baffle channels (`gather_folds * seam_groove`), and dashed lockstitch needle indentations along both diamond diagonals.
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{puffer\_pillow}\cdot(0.86 + 0.14\,\text{wrinkle}) + \text{gather\_folds}\cdot\text{seam\_groove} + 0.16\,\text{stitches})$.
- **Source**: *First principles* pneumatic cushion membrane equilibrium ($\sin^\alpha$ pillow dome) & ISO 4915 Class 301 lockstitch spacing.

### 7. `corduroy` (`tools/generators/types/corduroy.py`)
- **Construction**: Plain-weave ground backing (`woven_fabric_field`, `plain_m`, `wales*4`) overlaid by vertical velveteen cut-pile wales (`wales = 14 + (variant % 6) * 4`). Each wale cross-section follows a flat-topped superelliptical crown modulated by anisotropic cut-pile brush tufts (`periodic_gradient_perlin` and high-frequency `periodic_fbm` stretched along $u$ vs. $v$).
- **Height Field (`native`)**: $h = \mathrm{norm}(\max(\text{pile\_h},\, 0.24\,\text{base\_weave}) + 0.07\,\text{fuzz})$, shaded with plush velveteen sheen (`sheen_amt=22.0`).
- **Source**: *First principles* woven cut-pile wale geometry + Peirce (1937) ground weave.

### 8. `silk-brocade` (`tools/generators/types/silk_brocade.py`)
- **Construction**: Authentic **8-harness satin/sateen structural reversal** (`satin_8h` warp-faced matrix with step $+3 \bmod 8$ vs. its binary complement `sateen_8h` weft-faced matrix) computed with `woven_fabric_field` (`threads=64`, `float_arch=0.30`, `ply_twist=0.06`). Floral damask motifs (`motif`) modulate elevation, luster, and roughness (`0.20` glossy warp-float satin vs. `0.55` matte sateen ground).
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{motif}\cdot(0.38 + 0.62\,h_{\text{satin}}) + (1-\text{motif})\cdot(0.08 + 0.42\,h_{\text{sateen}}))$.
- **Source**: Peirce (1937) weave geometry; standard 8-end satin/sateen damask weave reversal structure.

### 9. `waffle-knit` (`tools/generators/types/waffle_knit.py`)
- **Construction**: Recessed thermal honeycomb cellular pockets bounded by raised cosine-arch rib ridges `1 - cos^2(...)` along $u$ and $v$ (`ridge = 0.55*max(wall_u, wall_v) + 0.45*avg`), composited over a sub-grid $1\times 1$ plain weave (`woven_fabric_field`, `cells*6`) representing yarn loops lining the pocket floor and walls.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.72\,\text{ridge} + 0.28\,\text{weave\_h})$.
- **Source**: *First principles* cellular thermal-pocket rib relief + Peirce (1937) yarn sub-weave.

### 10. `ripstop-nylon` (`tools/generators/types/ripstop_nylon.py`)
- **Construction**: $1\times 1$ plain-weave nylon base (`woven_fabric_field`, `sub=8` ends per grid box, `threads = cells * 8`) overlaid with thicker double-denier reinforcing warp/weft yarns at `iu % 8 == 0` and `iv % 8 == 0` (`rib_u`, `rib_v`) whose over/under crossing follows the parity of the underlying weave (`warp_over`).
- **Height Field (`native`)**: $h = \mathrm{norm}(\max(0.45\,\text{weave\_h},\, 0.95\,\text{ripstop\_grid}) + 0.04\,\text{slub})$.
- **Source**: MIL-DTL-44436 / standard 8-thread nylon ripstop plain-weave reinforcement grid + Peirce (1937).

### 11. `herringbone-weave` (`tools/generators/types/herringbone_weave.py`)
- **Construction**: Constructed directly as an exact periodic $(4, 2B)$ **broken-twill herringbone weave matrix** (`direction = +1` for $c < B$, `-1` for $c \ge B$ with $180^\circ$ clean phase break `break_shift` at chevron reversal) fed to `woven_fabric_field` (`threads = cols * band_w`, `float_arch=0.38`, `ply_twist=0.14`). Eliminates sawtooth breaks along diagonal zigzags.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.84\,\text{thread\_h} + 0.16\,\text{slub})$.
- **Source**: Peirce (1937) continuous float weave geometry; standard 2/2 broken-twill herringbone drafting matrix.

### 12. `rib-jersey-knit` (`tools/generators/types/rib_jersey_knit.py`)
- **Construction**: Periodic $1\times 1$, $2\times 2$, or $3\times 1$ alternating **face knit wales and recessed purl wales** (`period = 2, 4, 4`, `wales = m * period`). Knit columns evaluate the 3-neighbor interlocking Chamberlin-Leaf-Glaskin V-loop legs (`x_l, x_r, z_leg, twist`), while purl columns evaluate horizontal curved purl bumps (`purl_arc * purl_seg`) deep in the wale trough.
- **Height Field (`native`)**: $h = \mathrm{norm}(\mathrm{where}(\text{rib\_mask},\, \text{knit\_h},\, \text{purl\_h}) + 0.14\,\text{fuzz})$.
- **Source**: Chamberlain (1926) / Leaf & Glaskin (1955) knit loop geometry; Yuksel et al. (2012) *Stitch Meshes*.

### 13. `boucle-tweed` (`tools/generators/types/boucle-tweed.py`)
- **Construction**: Variable-diameter novelty slub yarns (`warp_radius = clip(0.72 + 0.34*slub_u, 0.55, 1.02)`) woven in $1\times 1$ plain weave via `woven_fabric_field` with warped thread centerlines (`tu_warp, tv_warp`), superposed with **1-iter Lloyd CVT curled boucle loop nubs** (`exp(-f1_nub^2 / 0.14) * nub_gate`) and wool neps/flecks.
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{weave\_h} + \text{boucle\_loop} + 0.18\,\text{fleck})$.
- **Source**: Peirce (1937) variable-radius yarn plain weave + Lloyd (1982) CVT boucle loop distribution (*first principles*).

### 14. `jacquard-geometric` (`tools/generators/types/jacquard-geometric.py`)
- **Construction**: Geometric diamond, interlocking circle, and chevron motifs (`motif`) rendered by alternating a **5-harness warp-faced satin** (`np.ones` float matrix, roughness $0.25$) inside the motif against a **5-harness weft-faced sateen** ground (roughness $0.62$) via `woven_fabric_field` (`threads=60`).
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{motif}\cdot(0.35 + 0.65\,h_{\text{satin}}) + (1-\text{motif})\cdot(0.08 + 0.42\,h_{\text{sateen}}))$.
- **Source**: Peirce (1937) 5-end satin/sateen jacquard reversal weave + *first principles* SDF geometric motifs.

### 15. `seersucker-shibori` (`tools/generators/types/seersucker-shibori.py`)
- **Construction**: Two textile modes over a $1\times 1$ `woven_fabric_field` plain weave: (a) **Seersucker** (`variant % 2 == 0`), where slack-tension warp bands (`0.5 + 0.5*tanh(6 sin(2\pi s_u))`) buckle into longitudinal gather ripples (`buckle`, `crinkle`); (b) **Arashi/Kumo Shibori** (`variant % 2 == 1`), where radial resist-dye rings (`cos(r\pi k + warp)`) accompany exponentially decaying compression pleat ridges (`pleats * exp(-0.9r)`).
- **Height Field (`native`)**: Seersucker $h = \mathrm{norm}(0.42\,\text{weave} + 0.58\,\text{pucker\_rel}\cdot(0.18 + 0.82\,\text{pucker\_band}))$; Shibori $h = \mathrm{norm}(0.56\,\text{linen} + 0.24\,\text{warp} + 0.20\,\text{pleats}\,e^{-0.9r})$.
- **Source**: Peirce (1937) plain weave + *first principles* differential-tension buckling & radial pleat decay.

### 16. `pebbled-leather` (`tools/generators/types/pebbled_leather.py`)
- **Construction**: **2-iteration periodic Lloyd CVT** (`periodic_lloyd_voronoi`, `lloyd_iters=2`) models tightly packed dermal collagen papillae (`pebble = sqrt(1 - 0.78*f1) * clip(4.5*edge, 0.16, 1.0)`), modulated by ridged hide flex creases and interstitial hair-follicle pores (`exp(-edge^2 / 0.008)`) localized along cell crevices.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.68\,\text{pebble} + 0.18\,\text{creases} + 0.14\,\text{micro} - 0.16\,\text{follicle\_pores})$.
- **Source**: Lloyd (1982) CVT cell packing & Worley (1996) cellular distance metrics; *first principles* leather follicle distribution.

### 17. `garment-wrinkles` (`tools/generators/types/garment-wrinkles.py`)
- **Construction**: Four tension-fold modes modulated by divergence-free **Bridson curl advection** (`periodic_curl_noise`) and Swiss multifractal secondary crumpling: pipe folds (`0.82 sin(\phi) + 0.18 sin(2\phi + 0.4)` asymmetric elastica buckle), pinched gathers (`amp_v` waist pinch envelope), randomized crush creases (curl-advected Voronoi edges + Swiss fBm), and diagonal bias drape.
- **Height Field (`native`)**: Normalized fold/drape/crease elevation $h \in [0,1]$ shared identically across normal/color/height outputs.
- **Source**: Bridson et al. (2007) curl-noise advection + *first principles* thin-shell elastica buckling harmonics.

---

## Family 2: Cloth, Seam & Lace Alphas (12 Types — 11 Modified, 1 Unchanged)

### 18. `double-topstitch-seam` (`tools/generators/types/double_topstitch_seam.py`)
- **Construction**: Flat-felled folded fabric step modeled by a $C^\infty$ **sigmoid profile** `(1 / (1 + exp(-18 x))) * 0.34`, localized transverse stitch puckering `exp(-x^2 / 0.04)`, parallel dual **ISO 4915 Class 301 lockstitch** thread dashes at $x = \pm 0.22$ (`sqrt(1 - d^2/r^2) * sin^0.55(\pi s_v) * twist`), and needle-penetration dimples (`hole_l, hole_r`) at stitch endpoints. Tiles vertically (`tile_axes="y"`).
- **Height Field (`native`)**: $h = \mathrm{clip}(\text{seam\_step} + \text{puckering} + 0.65\max(s_L, s_R) - \text{hole}_L - \text{hole}_R,\, 0, 1)$.
- **Source**: ISO 4915 Class 301 lockstitch seam geometry & *first principles* cylindrical thread/needle dimple relief.

### 19. `zipper-teeth-trim` (`tools/generators/types/zipper_teeth_trim.py`)
- **Construction**: Diagonal twill woven tape (`tape_mask * tape_twill`), twin cylindrical piping welt cords (`|x| = 0.165, r = 0.055`), and **IQ 2D rounded-box SDF** interlocking staggered zipper teeth (`_tooth_sdf` at $x = \pm 0.065$, phase-shifted $0.5$ pick between left and right tracks) with domed tooth crown bevels. Tiles vertically (`tile_axes="y"`).
- **Height Field (`native`)**: $h = \mathrm{clip}(\text{tape} + \text{cord} + 0.78\max(\text{left\_tooth}, \text{right\_tooth}),\, 0, 1)$.
- **Source**: Inigo Quilez 2D Rounded Box SDF (`iquilezles.org/articles/distfunctions2d/`) + *first principles* staggered zipper element profile.

### 20. `overlock-zigzag-seam` (`tools/generators/types/overlock_zigzag_seam.py`)
- **Construction**: **ISO 4915 Class 504 (3/4-thread overlock serger)** geometry (`tile_axes="y"`): upper and lower looper threads oscillate in antiphase $x_{\text{up}} = A\sin(2\pi s_v)$, $x_{\text{lo}} = -A\sin(2\pi s_v)$ with over/under Z-crossing depth $z = 0.80 \pm 0.20\cos(2\pi s_v)$, interlocking along the cut fabric edge purl bead (`x = +A`) and 1 or 2 straight needle-thread rails (`x = -A`, `-0.45A`).
- **Height Field (`native`)**: $h = \max(\text{upper\_looper},\, \text{lower\_looper},\, 0.92\,\text{purl},\, 0.88\,\text{rails})$.
- **Source**: ISO 4915:1991 Stitch Class 504 (3/4-thread overlock stitch topology) & *first principles* tubular yarn elevation.

### 21. `four-hole-button` (`tools/generators/types/four-hole-button.py`)
- **Construction**: Non-tileable (`tile_axes="none"`) radial circular disc (`btn_r`), raised perimeter rim (`exp(-((r-(btn_r-0.09))^2)/0.0045)`), dished inner basin (`recess`), 4 punched cylindrical eyelets at $(\pm 0.18, \pm 0.18)$, and **2D capsule SDF thread bars** (`_capsule_dist` in crossed X-stitch on even variants with over/under depth $0.85$ vs. $1.00$, or parallel = bars on odd variants). Corners strictly zero ($\Rightarrow$ `(128,128,255)` flat normals).
- **Height Field (`native`)**: $h = \mathrm{clip}(\text{disc}\cdot(0.64 + \text{rim} - \text{recess})\cdot(1 - 0.72\,\text{holes}) + \text{threads}\cdot\text{disc},\, 0, 1)$.
- **Source**: Inigo Quilez 2D Capsule SDF (`iquilezles.org/articles/distfunctions2d/`) & *first principles* button geometry.

### 22. `ribbed-cuff-trim` (`tools/generators/types/ribbed_cuff_trim.py`)
- **Construction**: Horizontal cuff band (`tile_axes="x"`) with alternating knit columns (`ix % 2 == 0`) evaluating 3-neighbor interlocking Chamberlain-Leaf-Glaskin V-loop legs (`knit_h`) and purl troughs (`purl_h`), gated by an anti-aliased horizontal band envelope (`band_mask`).
- **Height Field (`native`)**: $h = \mathrm{where}(\text{is\_knit},\, 0.35 + 0.65\,\text{knit\_h},\, 0.18 + \text{purl\_h}) \cdot \text{band\_mask}$.
- **Source**: Chamberlain (1926) / Leaf & Glaskin (1955) plain knitted loop geometry & *first principles* $1\times 1$ rib cuff construction.

### 23. `cross-stitch-seam` (`tools/generators/types/cross-stitch-seam.py`)
- **Construction**: Horizontal decorative embroidery band (`tile_axes="x"`) where each cell $(c_u, c_v) \in [-0.5, 0.5)^2$ renders two crossing diagonal **capsule SDF yarns** (`_seg_dist` from $(-0.34,-0.34)\to(0.34,0.34)$ and $(-0.34,0.34)\to(0.34,-0.34)$) with distinct lower/upper Z-elevation (`0.62` vs. `0.76`) so the top stitch arches cleanly over the bottom stitch, plus 2-ply twist ripple.
- **Height Field (`native`)**: $h = \max(s_1, s_2) \cdot \text{band}$.
- **Source**: Inigo Quilez 2D Segment/Capsule SDF & *first principles* overlapping embroidery cross-stitch topology.

### 24. `rivet-grommet` (`tools/generators/types/rivet_grommet.py`) — **confirmed, unchanged**
- **Construction**: Non-tileable (`tile_axes="none"`) annular beveled washer (`sin(pi * clip((r - inner_r)/(outer_r - inner_r), 0, 1))`), inner clinch prongs (`cos(k\theta)` Gaussian ring around `inner_r`), and outer radial leather compression puckers (`sin^2(m\theta)` Gaussian ring around `outer_r * 1.18`). Inspected and confirmed physically accurate as-is.
- **Height Field (`native`)**: $h = \text{ring}\cdot\text{bevel} + \text{prongs} + \text{puckers}$.
- **Source**: *First principles* annular washer bevel and radial stress-pucker harmonics (confirmed, unchanged).

### 25. `scalloped-lace-trim` (`tools/generators/types/scalloped_lace_trim.py`)
- **Construction**: Horizontal trim (`tile_axes="x"`) bounded by scalloped periodic lobes `0.32 + 0.18 |sin(\pi u \cdot \text{waves})|`, punched broderie-anglaise eyelet holes surrounded by ribbed satin embroidery stitches (`satin_rib = 0.78 + 0.22 cos(18\theta)` around `er = 0.145`), over a diagonal bobbinet tulle diamond lattice.
- **Height Field (`native`)**: $h = \text{base}\cdot\text{eyelet\_hole}\cdot\text{tulle} + \text{rim}\cdot\text{base}$.
- **Source**: *First principles* periodic scallop envelope, radial satin-stitch eyelet embroidery, and diagonal tulle mesh.

### 26. `lace-mesh-mask` (`tools/generators/types/lace_mesh_mask.py`)
- **Construction**: Four tileable opacity/alpha relief masks: (v0) Chantilly floral lace (5-petal rose curve `0.28 + 0.12 cos(5\theta)` + circular cordonnet border on diagonal tulle), (v1) diamond fishnet mesh (`min(d1, d2) < w` with cylindrical cord arch), (v2) staggered hexagonal jersey eyelet perforations on fine knit loops, and (v3) ripped denim tear (fBm threshold mask exposing frayed $3\times 1$ twill warp edge and bridged intact weft threads `weft_strands`).
- **Height Field (`derived`)**: Scalar opacity mask $h \in [0,1]$ returned directly as height per `SPEC.normal_source="derived"`.
- **Source**: *First principles* polar rose-curve lace, staggered hexagonal lattice SDF, and thresholded twill-warp/weft tear mask.

### 27. `smocked-gather` (`tools/generators/types/smocked_gather.py`)
- **Construction**: Elasticated shirred smocking cuff (`tile_axes="x"`): staggered honeycomb pleats `(u * cols + 0.5*(iy % 2))` producing diamond/ovoid fabric puffs (`cushion = (sin(\pi fu) sin(\pi fv))^{0.45}`) pinched by fine axial gather creases (`pleat_folds`) and stitched thread-tack dimples at diamond intersections (`tack_dimple`).
- **Height Field (`native`)**: $h = \mathrm{clip}(\text{cushion}\cdot\text{pleat\_folds} - \text{tack\_dimple},\, 0, 1)\cdot\text{mask}$.
- **Source**: *First principles* staggered honeycomb smocking gather cushion and needle-tack dimple geometry.

### 28. `belt-buckle` (`tools/generators/types/belt_buckle.py`)
- **Construction**: Non-tileable (`tile_axes="none"`) rounded-rectangle hardware frame via **IQ 2D Rounded Box SDF** (`sdf_out` outer chamfer, `sdf_in` cutout) yielding a smooth beveled cross-section `sin(\pi \cdot d / half_w)^{0.75}`, plus a cylindrical capsule prong pin resting on the center crossbar (`pin`, rotated slightly in variants) and flat `(128,128,255)` borders.
- **Height Field (`native`)**: $h = \max(\text{frame},\, \text{pin})$.
- **Source**: Inigo Quilez 2D Rounded Box & Capsule SDF (`iquilezles.org/articles/distfunctions2d/`).

### 29. `blanket-stitch-seam` (`tools/generators/types/blanket_stitch_seam.py`)
- **Construction**: Vertical edge-finish seam (`tile_axes="y"`): longitudinal spine cord (`x = -0.14`) and evenly spaced perpendicular loop bars (`_capsule_dist` from `-0.14` to `+0.18`) with cylindrical yarn profiles $\sqrt{1-d^2}$, slight helical twist (`0.90 + 0.10 sin(...)`), and a folded wool seam ridge behind the stitches.
- **Height Field (`native`)**: $h = \mathrm{clip}(\max(\text{spine},\, \text{cross\_bar}) + \text{seam\_ridge},\, 0, 1)$.
- **Source**: ISO 4915 Class 500 edge-finish blanket stitch geometry & IQ capsule yarn profile.

---

## Family 3: Surface Material Textures (22 Types — 20 Modified, 2 Unchanged)

### 30. `domain-warped-marble` (`tools/generators/types/domain_warped_marble.py`) — **confirmed, unchanged**
- **Construction**: **Inigo Quilez nested domain warping** $f(\mathbf{p} + 0.22\,\mathbf{r}(\mathbf{p} + 0.14\,\mathbf{q}(\mathbf{p})))$ with two stages of periodic fBm/ridged/turbulence offsets (`q_u, q_v` -> `r_u, r_v` -> `final`), followed by sinusoidal mineral vein interference `0.5 + 0.5 sin((final * k + q_u) 2\pi)`. Confirmed exact canonical IQ domain warp.
- **Height Field (`derived`)**: $h = \mathrm{norm}(0.75\,\text{color\_field} + 0.25\,r_u)$.
- **Source**: Inigo Quilez *Domain Warping* (`iquilezles.org/articles/warp/`) (confirmed, unchanged).

### 31. `swiss-erosion-rock` (`tools/generators/types/swiss_erosion_rock.py`) — **confirmed, unchanged**
- **Construction**: **Swiss ridged multifractal** (`periodic_fbm(..., mode="swiss", octaves=6)`) accumulating slope-derivative advection `dx_acc, dy_acc` and amplitude energy weighting, blended with sinusoidal tilted sedimentary bedding (`strata`) and high-frequency `turbulence` rock grit (`micro`). Confirmed authentic Swiss multifractal construction.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.72\,\text{swiss} + 0.16\,\text{strata} + 0.12\,\text{micro})$.
- **Source**: Musgrave / Gaffer derivative-weighted Swiss ridged multifractal (`common.periodic_fbm`, `mode="swiss"`) (confirmed, unchanged).

### 32. `voronoi-basalt-crackle` (`tools/generators/types/voronoi_basalt_crackle.py`)
- **Construction**: Upgraded to **2-iteration Periodic Lloyd CVT** (`periodic_lloyd_voronoi`, `lloyd_iters=2`) so cooling polygonal basalt columns anneal toward natural ~120° hexagonal columnar jointing (Goehring 2013), paired with per-column planar fracture tilt (`tx * rx_a + ty * ry_a`) and secondary finer shrinkage cracks (`edge_b`).
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{crack\_major}\cdot\text{crack\_minor}\cdot(0.72\,\text{tilt} + 0.28\,\text{micro}))$.
- **Source**: Goehring (2013) *Evolving fracture patterns: columnar joints, mud cracks and polygonal terrain* (*Phil. Trans. R. Soc. A*, arXiv:1211.6762); Lloyd (1982) CVT.

### 33. `hardwood-planks` (`tools/generators/types/hardwood_planks.py`)
- **Construction**: Grounded in **Liu & Marschner (2015/2016)** 3D wood grain modeling: cylindrical annual growth-ring phase uses an **asymmetric earlywood-to-latewood sawtooth/power curve** (`late = np.power(ring_frac, 2.2)` / sharp density drop at ring boundary) warped around elliptical **branch knots** (`knot_f1` localized distortion + smooth deflection) and **ring-porous earlywood vessels** (`vessels` micropits clustered in low-`ring_frac` springwood), mapped onto periodic staggered plank boards with chamfered tongue-and-groove bevels.
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{grain} \cdot \text{bevel})$ (where `grain = norm(0.64 rings + 0.18 fibers + 0.10*(1 - knot_influence) - 0.18 vessels)`).
- **Source**: Liu, Dong, Hašan & Marschner (2015/2016) *Simulating the Structure and Texture of Solid Wood* / *3D Wood Grain Texture Modeling* (arXiv:1511.04224).

### 34. `masonry-tiles` (`tools/generators/types/masonry_tiles.py`)
- **Construction**: Three exact architectural tessellations: (v0) **Exact periodic $2:1$ herringbone paver lattice** (`u_g, v_g` partitioned on a $4\times 4$ modular super-cell block so every brick lies seamlessly at $2:1$ length-to-width across tile edges); (v1) **BIA 1:3 US Modular running bond** (`cols = max(2, rows // 3)` with half-brick course stagger); (v2) **Archimedean $4.8.8$ truncated-square zellige mosaic** (octagons defined by dual $\ell_1/\ell_\infty$ chamfers paired with $45^\circ$-rotated interstitial diamond cabochons).
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{pillow\_dome}\cdot(0.82 + 0.18\,\text{micro}))$ inside smoothstep mortar joints.
- **Source**: BIA Technical Notes 10 (Modular Brick 1:3 ratio); Grünbaum & Shephard (1987) *Tilings and Patterns* (Archimedean $4.8.8$ tiling).

### 35. `damascus-carbon-metal` (`tools/generators/types/damascus_carbon_metal.py`)
- **Construction**: Four physically distinct metal/composite surfaces: (v0) **Bilayer folded acid-etched Damascus steel** (steep `tanh` differential acid etch between high-carbon martensite & nickel-rich austenite layers + fine anisotropic Gabor polish); (v1) **$2\times 2$ twill carbon fiber** via `woven_fabric_field` (`twill_22`, `float_arch=0.45`) with anisotropic directional tow micro-filaments (`periodic_gabor_noise` parallel to warp vs. weft); (v2) **Machined $45^\circ$ diamond frustum knurling** (`min(flat_top, pyramid)` truncated pyramidal teeth); (v3) **Anisotropic brushed cast metal** (`periodic_gabor_noise` at $\omega_0 = 0, 0.04$ rad + pitted casting pores).
- **Height Field (`native`)**: Layer-etch/weave/knurl-frustum/Gabor-scratch height field $h \in [0,1]$.
- **Source**: Peirce (1937) 2/2 twill carbon weave; Lagae et al. (2009) anisotropic Gabor brushing; *first principles* acid-etch step & diamond knurl frustum.

### 36. `scifi-greeble-panels` (`tools/generators/types/scifi_greeble_panels.py`)
- **Construction**: Two-level periodic hierarchical hull subdivision (`grid` macro cells split into $1\times 2$, $2\times 1$, or $2\times 2$ sub-panels via deterministic `np.int64` hashing) with chamfered **IQ Rounded Box SDF** panel borders (`seam`), hexagonal/circular recessed fastener bolts, and $45^\circ$ sawtooth cooling louvers (`role` dispatch).
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{seam}\cdot(0.78\,\mathrm{clip}(\text{feature}, 0, 1.2) + 0.22\,\text{wear}))$.
- **Source**: Inigo Quilez 2D Rounded Box SDF (`iquilezles.org/articles/distfunctions2d/`) & *first principles* hierarchical hull subdivision.

### 37. `biome-bark-foliage` (`tools/generators/types/biome_bark_foliage.py`)
- **Construction**: Three biome surfaces: (v0) **Anisotropic tree bark** combining vertically elongated **1-iter Lloyd CVT** ridges (`bark_u = 2.4 u`, `bark_v = 0.5 v`) with vertical fibrous phloem nap (`periodic_gabor_noise`, $\omega_0 = \pi/2$); (v1) **Asymmetric wind-blown saltation sand ripples** (gentle stoss / steep lee phase profile); (v2) **1-iter Lloyd CVT bryophyte moss cushions** blended with Swiss multifractal micro-fronds.
- **Height Field (`native`)**: $h \in [0,1]$ combining fissures/ripples/pillows with `micro`.
- **Source**: Lloyd (1982) CVT; Lagae et al. (2009) Gabor fiber nap; Bagnold (1941) / Anderson (1987) asymmetric ripples.

### 38. `crystalline-geode-agate` (`tools/generators/types/crystalline-geode-agate.py`)
- **Construction**: Two mineral habits: (v0) **Fortification agate** using geodesic `1-iter Lloyd CVT` cavity distance contours modulated by fBm warp into rhythmic **Liesegang precipitation bands** (`0.5 + 0.5 cos(2\pi k d + \text{warp})`); (v1–v3) **Trigonal/rhombohedral quartz geode crystals** built from 1-iter Lloyd CVT centers with explicit **3-plane pyramidal facet apex** $\max_{m=0,1,2}(\mathbf{r}\cdot\hat{\mathbf{e}}_m(\theta_c + 2\pi m/3))$ (`apex = clip(1 - tri_prism, 0, 1)`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.62(1-f_1) + 0.28\,\text{rings} + 0.10\,\text{micro})$ (agate) or $h = \mathrm{norm}(0.72\,\text{apex} + 0.18\,\text{facet\_tilt} + 0.10\,\text{edge})$ (quartz).
- **Source**: *First principles* Liesegang contour precipitation & trigonal $120^\circ$ crystal habit intersection planes.

### 39. `stylized-arcane-cel` (`tools/generators/types/stylized-arcane-cel.py`)
- **Construction**: Three stylized modes: (v0) **Tapered curved cel grass tufts** evaluated by swept quadratic-curve blade SDFs blended with 5-band cel-quantized fBm (`quant`); (v1) **Sharp optical water caustics** via dual Voronoi cell boundaries $c_k = (1 - \mathrm{edge}_k)^6$ concentrating bright cusps along refractive envelope intersections (Snell refraction focusing); (v2) **Engraved rune-carved Lloyd CVT dungeon slabs** (`lloyd_iters=1`) with incised segment SDF glyphs (`engraving`).
- **Height Field (`native`)**: Blade/caustic/incised-slab height $h \in [0,1]$.
- **Source**: Inigo Quilez Voronoi caustic cusps / segment SDF (`iquilezles.org/`); Lloyd (1982) CVT slab geometry.

### 40. `turing-reaction-diffusion` (`tools/generators/types/turing-reaction-diffusion.py`)
- **Construction**: Real numerical integration of the **Pearson (1993) Gray-Scott reaction-diffusion PDE** (`common.gray_scott_rd`, $D_u=0.16, D_v=0.08$, 420–620 steps at `sim_res=128..160` on $\mathbb{T}^2$) across four classic regime parameter pairs $(F, k)$: mitosis/spots `(0.0367, 0.0649)`, coral/labyrinth `(0.0545, 0.0620)`, worm/fingerprint `(0.0460, 0.0630)`, and Turing stripes `(0.0350, 0.0600)`.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.84\,\text{ridges}(V) + 0.16\,\text{micro})$.
- **Source**: Pearson (1993) *Complex Patterns in a Simple System*, *Science* 261(5118):189–192; Turing (1952) *Phil. Trans. R. Soc. B* 237:37–72.

### 41. `terrazzo` (`tools/generators/types/terrazzo.py`)
- **Construction**: Dual-scale embedded marble/quartz chips extracted from Voronoi cellular domains with **smoothstep anti-aliased boundaries** (`chip1`, `chip2`), internal mineral turbulence veining (`vein`), and an inlaid polished cementitious terrazzo binder (`base_noise`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.68 + 0.10\,\text{chip\_mask} + 0.14\,\text{base\_noise}\cdot(1 - 0.5\,\text{chip\_mask}))$.
- **Source**: Worley (1996) cellular fragments & *first principles* smoothstep aggregate chip polish relief.

### 42. `hex-mosaic` (`tools/generators/types/hex-mosaic.py`)
- **Construction**: Exact periodic **pointy-top regular hexagon tessellation** evaluated on an axial/staggered isometric lattice whose aspect ratio $\sqrt{3}$ snaps to nearest even-row periodic integer `(cols, rows)` (`rows ≈ round(cols / (sqrt(3)/2))`), yielding exact Euclidean distance to the nearest regular hexagon center (`f1`), edge distance to the adjacent hex (`edge = f2 - f1`), pillowed ceramic glaze dome (`pillow`), and recessed grout.
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{grout}\cdot(0.66 + 0.24\,\text{pillow}) + 0.10\,\text{glaze})$.
- **Source**: Exact $A_2$ regular hexagonal Voronoi lattice geometry (*first principles* / Conway & Sloane $A_2$ lattice).

### 43. `wind-dunes` (`tools/generators/types/wind-dunes.py`)
- **Construction**: **Bagnold (1941) aeolian dune profile** combining a long convex windward stoss slope ($\sim 10^\circ$–$15^\circ$) with a rounded brink and steep leeward avalanche **slipface at angle of repose ($\sim 34^\circ$)** (`dune`), superposed with secondary asymmetric saltation wind ripples that are **physically suppressed on the sheltered lee slipface** (`stoss_weight`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.84\,\text{dune} + 0.16\,\text{ripple\_asym}\cdot\text{stoss\_weight})$.
- **Source**: Bagnold (1941) *The Physics of Blown Sand and Desert Dunes*; Anderson (1987) *Sedimentology* 34:943–956.

### 44. `pcb-circuit` (`tools/generators/types/pcb-circuit.py`)
- **Construction**: Periodic **FR-4 woven fiberglass substrate** (`woven_fabric_field`, `plain_11`, `fg_n`) beneath a topologically connected periodic **octilinear ($0^\circ, 45^\circ, 90^\circ$) Manhattan+chamfer trace router** (`capsule SDF` copper traces linking toroidal grid pads and drilled solder vias `drill`). Solder mask vs. bare ENIG gold/copper trace roughness separation.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.76\,\text{mask} + 0.16\,\text{fiberglass}\cdot(1 - 0.6\,\text{mask}) - 0.28\,\text{drill})$.
- **Source**: IPC-2221/FR-4 fiberglass substrate weave (Peirce 1937) + *first principles* octilinear capsule SDF trace routing.

### 45. `stained-glass` (`tools/generators/types/stained-glass.py`)
- **Construction**: Cathedral glass panes generated by **1-iter Lloyd CVT** (`periodic_lloyd_voronoi`), bounded by raised **half-round cylindrical lead H-came cames** (`sqrt(1 - (edge/w)^2)`), with wavy hand-rolled cathedral glass surface ripple (`glass_ripple`, roughness `0.06` vs. matte lead came `0.60`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.68\,\text{lead} + 0.32\,(1-\text{lead\_t})\cdot\text{glass\_ripple})$.
- **Source**: Lloyd (1982) CVT pane tessellation & *first principles* semicircular lead H-came cross-section.

### 46. `hammered-metal` (`tools/generators/types/hammered-metal.py`)
- **Construction**: Repoussé/planished hammered metal formed by **2-iteration Periodic Lloyd CVT** (`lloyd_iters=2`) producing uniform planishing coverage of shallow **concave spherical-cap ball-peen dimples** (`f1^2` parabolic cup profile bounded by raised burnished cell rims `rim_burnish`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.86\,\text{dimple} + 0.14\,\text{micro})$.
- **Source**: Lloyd (1982) CVT + *first principles* spherical-indentor planishing cup geometry.

### 47. `dragon-scales` (`tools/generators/types/dragon-scales.py`)
- **Construction**: Multi-row overlapping hexagonal-staggered armor scales evaluated across prior anterior rows (`dr = 0..-2`) so each row cleanly imbricates/overlaps posterior rows with a pointed ogive boundary, **longitudinal dorsal keel ridge** (`exp(-su^2 / 0.012)`), radial striae, and posteriorly rising pitch tilt (`tilt`).
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{scale\_dome} + \text{keel} + 0.10\,\text{micro})$.
- **Source**: *First principles* staggered anterior-overlapping imbrication and dorsal keel geometry.

### 48. `fluted-travertine` (`tools/generators/types/fluted-travertine.py`)
- **Construction**: Classical architectural column **concave circular-arc flutes** `1 - 0.78 sqrt(1 - in_flute^2)` separated by crisp flat vertical arrises (`fu > channel_span`), intersected by horizontally elongated sedimentary bedding strata (`freqy=18`) and irregular dissolution **travertine vugs/pores** (`freqy=34`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.62\,\text{flute\_wave} + 0.26\,\text{strata} - 0.34\,\text{pores})$.
- **Source**: Classical Doric/Ionic semicircular flute & arris geometry (*first principles*) + sedimentary travertine dissolution porosity.

### 49. `cobblestone-paving` (`tools/generators/types/cobblestone-paving.py`)
- **Construction**: Sett/cobble street pavers formed by **2-iteration Periodic Lloyd CVT** (`lloyd_iters=2`) giving naturalistic interlocking rounded cobble domes (`smoothstep` stone perimeter bevel + convex wear dome) embedded in deep recessed sandy lime mortar joints.
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{stone\_mask}\cdot\text{cobble}\cdot(0.76 + 0.24\,\text{rock\_detail}))$.
- **Source**: Lloyd (1982) CVT stone packing + *first principles* smoothstep cobble dome and mortar bevel.

### 50. `iridescent-thinfilm` (`tools/generators/types/iridescent-thinfilm.py`)
- **Construction**: Physically based thin-film optical interference via the **Airy reflectance formula** $R(\lambda) = \frac{F \sin^2(\delta(\lambda)/2)}{1 + F \sin^2(\delta(\lambda)/2)}$ evaluated across RGB wavelengths $\lambda \in \{650, 532, 460\}\text{ nm}$ with film refractive index $n_{\text{film}}=1.45$, finesse $F$, and fBm-driven film thickness (optical path difference `opd`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.65\,f_2 + 0.35\,f_1)$ representing fluid thin-film surface undulations.
- **Source**: Airy (1833) / Born & Wolf *Principles of Optics* thin-film multi-beam interference equation.

### 51. `weathered-concrete-patina` (`tools/generators/types/weathered-concrete-patina.py`)
- **Construction**: Six variant-dispatched architectural weathering modes: (v0) aggregate concrete with **1-iter Lloyd CVT** craze cracks; (v1) peeling paint with curled delamination rim (`exp(-((base - 0.48)^2)/0.0012)`); (v2) hot-dip galvanized **zinc spangle** (1-iter Lloyd Voronoi grains with directional crystallographic dendrite lamellae `cos(\theta)`); (v3) asymmetric sedimentary rock ledges; (v4) tapered barrel clay **roof tiles** (`step`); and (v5) domain-warped copper/iron corrosion patina.
- **Height Field (`native`)**: Variant-specific physical relief $h \in [0,1]$.
- **Source**: Lloyd (1982) CVT (cracks & zinc grain tessellation) + *first principles* dendritic crystal orientation, delamination curl, and barrel-tile overlap.

---

## Family 4: Substance Utility, Mask & Noise Maps (4 Types — 4 Modified, 0 Unchanged)

### 52. `grunge-wear-mask` (`tools/generators/types/grunge_wear_mask.py`)
- **Construction**: Four tileable 2D weathering mask modes: (v0) anisotropic directional micro-scratches via multi-band **Lagae-Lefebvre sparse Gabor noise** (`periodic_gabor_noise`) mixed with blotch fBm; (v1) **Bridson divergence-free curl-advected gravity drip streaks** (`periodic_curl_noise`); (v2) smoothstep thresholded paint/rust peel patches; (v3) **1-iter Lloyd CVT** crazing/mud-crack edge fissures.
- **Height Field (`derived`)**: Scalar wear mask $\mathrm{norm}(0.80\,\text{raw} + 0.20\,\text{base}) \in [0,1]$ exposed directly as `generate_height`.
- **Source**: Lagae et al. (2009) SIGGRAPH Gabor noise; Bridson et al. (2007) SIGGRAPH curl-noise advection; Lloyd (1982) CVT.

### 53. `procedural-noise-map` (`tools/generators/types/procedural_noise_map.py`)
- **Construction**: Five canonical periodic mathematical noise fields: (v0) **Swiss ridged multifractal** (`periodic_fbm(..., mode="swiss")`); (v1) **true periodic Gabor noise** (`periodic_gabor_noise`, isotropic or bandpass oriented); (v2) **Worley cellular distance** (`periodic_voronoi` under Euclidean, Manhattan, or Chebyshev metrics); (v3) **resolution-independent Bridson curl-flow advection** (`periodic_curl_noise` warping fBm); (v4) **Pearson (1993) Gray-Scott reaction-diffusion** (`gray_scott_rd`, $F=0.042, k=0.060$).
- **Height Field (`derived`)**: Dimensionless periodic scalar field $h \in [0,1]$.
- **Source**: Musgrave Swiss multifractal; Lagae et al. (2009) Gabor noise; Worley (1996); Bridson et al. (2007); Pearson (1993).

### 54. `baking-utility-map` (`tools/generators/types/baking_utility_map.py`)
- **Construction**: Generates underlying surface height $h$ via `_base_surface_fields` (v0/v3 **1-iter Lloyd CVT** cobblestone/crackle, v1/v4 **Swiss multifractal** rock, v2 beveled panels) and computes Sobel tangent normals, cavity-aware ambient occlusion (`ao`), and high-pass curvature (`curv`) packed into ORM/curv/AO/tangent-flow maps (`periodic_curl_noise` tangent flow map on flow variants).
- **Height Field (`native`)**: Underlying physical height field $h$ baked by `_base_surface_fields`.
- **Source**: Lloyd (1982) CVT; Musgrave Swiss multifractal; Bridson et al. (2007) curl velocity flow field.

### 55. `matcap-sphere` (`tools/generators/types/matcap_sphere.py`)
- **Construction**: Non-tileable (`tile_axes="none"`) canonical unit hemisphere projection $s_z = \sqrt{\max(0, 1 - x^2 - y^2)}$ on $r \le 1$ with anti-aliased silhouette rim (`_sstep`), **normalized Blinn-Phong halfway vector** $\mathbf{H} = \mathrm{normalize}(\mathbf{L} + \mathbf{V})$ ($\mathbf{V}=(0,0,1)$) specular highlight $(n \cdot H)^p$, and physically based **Schlick Fresnel approximation** $F = 0.04 + 0.96(1 - s_z)^5$.
- **Height Field (`derived`)**: Spherical cap elevation $h = \mathbf{1}_{r \le 1}\,s_z$.
- **Source**: Blinn (1977) SIGGRAPH halfway specular model; Schlick (1994) *Comput. Graph. Forum* 13(3):233–246 Fresnel reflectance.

---

## Family 5: Sculpting & Mechanical Decal Alphas (2 Types — 2 Modified, 0 Unchanged)

### 56. `sculpt-stamp-alpha` (`tools/generators/types/sculpt_stamp_alpha.py`)
- **Construction**: Non-tileable (`tile_axes="none"`) high-contrast radial sculpting brush stamps framed by zeroed margins (`corners=(128,128,255)` normals): (v0) hexagonal bolt head (`max(|\mathbf{p}\cdot\hat{\mathbf{e}}_k|)` poly SDF) with recessed smoothstep screwdriver **slot SDF** (`slot_sdf`) and sinusoidal chamfer bevel; (v1) parabolic bowl impact crater centered accurately inside the stamp with radial ejecta rays; (v2) **1-iter Lloyd CVT** fractured stone chisel facets + Swiss multifractal; (v3) overlapping **TIG dime-stack weld beads** (curved arc of overlapping Gaussian/parabolic crescent beads inside falloff envelope); (v4) smoothstep concentric arcane sigil rings and spokes.
- **Height Field (`native`)**: Brush stamp height scalar $h \in [0,1]$ masked to zero at borders.
- **Source**: Inigo Quilez 2D Box/Polygon SDFs (`iquilezles.org/articles/distfunctions2d/`) + Lloyd (1982) CVT + *first principles* TIG weld stack & parabolic crater bowl.

### 57. `mechanical-decal-alpha` (`tools/generators/types/mechanical_decal_alpha.py`)
- **Construction**: Non-tileable (`tile_axes="none"`) hard-surface & organic brush decals (`corners=(128,128,255)`): (v0) involute-like **trapezoidal gear wheel** with smoothstep spoke window cutouts and centered hub; (v1) **IQ Rounded Box SDF** (`box_sdf`) framing parallel slanted cooling louvers; (v2) multi-claw tapered gouge scratches inside radial falloff; (v3) radial Voronoi glass/armor shatter; (v4) rose-curve mechanical rosette; (v5) **1-iter Lloyd CVT** armored scale stamp; (v6) hammer dents; (v7) anti-aliased `smoothstep` optical targeting reticle and chevrons.
- **Height Field (`native`)**: Decal relief field $h \in [0,1]$ multiplied by smooth radial falloff envelope.
- **Source**: Inigo Quilez 2D Rounded Box & Polar SDFs (`iquilezles.org/articles/distfunctions2d/`) + Lloyd (1982) CVT + *first principles*.

---

## Family 6: Military Camouflage Textures (9 Types — 9 Modified, 0 Unchanged)

### 58. `woodland-camo` (`tools/generators/types/woodland_camo.py`)
- **Construction**: **U.S. M81 Woodland (`MIL-DTL-44436D`)** target color coverage proportions enforced exactly via **hierarchical conditional quantile thresholding** (`np.quantile`) over domain-warped (`wx, wy`) periodic anisotropic fBm fields (`freqy > freqx`): `16%` Black (`f_black >= q_k`), `34%` Brown (`f_brown >= q_b` over free complement), `30%` Green (`f_green >= q_g`), `20%` Tan background. Evaluated over a $2\times 2$ twill weave via `woven_fabric_field` (`threads` multiple of `4`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.76\,\text{weave} + 0.24\,\text{slub})$ shaded via `shade_cloth_surface`.
- **Source**: U.S. Military Specification **MIL-DTL-44436D** (M81 Woodland area ratios 34/30/20/16%) & Peirce (1937) 2/2 twill.

### 59. `desert-camo` (`tools/generators/types/desert_camo.py`)
- **Construction**: **U.S. 3-Color Desert Camouflage (`DCU`, Drawing 2-1-2240)** broad sweeping horizontal terrain patches enforced via hierarchical conditional quantiles (`16%` Earth Brown, `34%` Medium Tan, `50%` Light Cream/Khaki) on warped low-frequency anisotropic fBm (`freqy >= freqx`), woven on a $2\times 2$ twill (`threads` multiple of `4`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.75\,\text{weave} + 0.25\,\text{slub})$.
- **Source**: U.S. Army Natick Drawing **2-1-2240** (3-Color Desert Camouflage Pattern 50/34/16% coverage) & Peirce (1937) 2/2 twill.

### 60. `chocolate-chip-camo` (`tools/generators/types/chocolate_chip_camo.py`)
- **Construction**: **U.S. 6-Color Desert Battle Dress Uniform (`DBDU`, Drawing 2-1-1884)**: four broad desert strata (`14%` Brown, `30%` Medium Tan, `50%` Pale Sand, `6%` Olive/Sandstone) quantized hierarchically, overlaid by clustered **1-iter Lloyd CVT** angular dark "chocolate chip" shadow pebbles (`f1`) paired with directional sunward white highlight crescents (`cx_s, cy_s` offset). Woven on `woven_fabric_field` $2\times 2$ twill (`threads` multiple of `4`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.75\,\text{weave} + 0.25\,\text{slub})$.
- **Source**: U.S. Army Drawing **2-1-1884** (6-Color DBDU pebble-and-highlight geometry); Lloyd (1982) CVT; Peirce (1937).

### 61. `digital-woodland-camo` (`tools/generators/types/digital_woodland_camo.py`)
- **Construction**: **MARPAT Woodland (`US6805957B1`)**: orthogonal optel discretization across two integer grid scales (`2x2` coarse clusters `optels = 64` + `1x1` fine boundary breakup pixels `128`) driven by warped multi-scale fBm fields and enforced by conditional quantile thresholds (`47%` Woodland Green, `30%` Coyote Tan/Khaki, `18%` Coyote Brown, `5%` Black). Cloth relief uses **1/1 8-thread NYCO ripstop** via `woven_fabric_field` (`plain_11`, `threads` multiple of `8`) + double-thread `rip_grid`.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.68\,\text{weave} + 0.16\,\text{rip\_grid} + 0.16\,\text{slub})$.
- **Source**: U.S. Patent **US6805957B1** (MARPAT dual-scale digital optel camouflage ratios 47/30/18/5%) & NYCO 1/1 ripstop weave.

### 62. `digital-desert-camo` (`tools/generators/types/digital_desert_camo.py`)
- **Construction**: **MARPAT Desert (`US6805957B1`)**: identical dual-scale `2x2 + 1x1` optel spatial architecture (`optels = 64` / `128` grid) with horizontal desert stratification and conditional quantiles enforcing **47% Light Coyote Tan, 30% Urban Tan/Khaki, 18% Coyote Brown, 5% Light Olive/Sand** over 1/1 8-thread NYCO ripstop `woven_fabric_field`.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.68\,\text{weave} + 0.16\,\text{rip\_grid} + 0.16\,\text{slub})$.
- **Source**: U.S. Patent **US6805957B1** (Desert MARPAT 47/30/18/5% optel geometry) & NYCO 1/1 ripstop weave.

### 63. `tiger-stripe-camo` (`tools/generators/types/tiger_stripe_camo.py`)
- **Construction**: Horizontal elongated brush-stroke stripes modulated along warp direction using horizontal **Lagae-Lefebvre anisotropic Gabor bristle noise** (`omega0 = 0.0`, `bristle` / `bristle_fine`) mixed into low-`freqx`/high-`freqy` fBm lobes, quantized via conditional quantiles (`30%` Black, `22%` Dark Olive, `10%` Khaki/Tan, `38%` Moss Green base) on $2\times 2$ twill `woven_fabric_field`.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.75\,\text{weave} + 0.25\,\text{slub})$.
- **Source**: Vietnam-era Gold/Tadpole Tiger Stripe textile geometry; Lagae et al. (2009) directional Gabor brush bristles; Peirce (1937).

### 64. `flecktarn-camo` (`tools/generators/types/flecktarn_camo.py`)
- **Construction**: **Bundeswehr 5-Color Flecktarn (`TL 8305-0290`)**: clustered stippled flecks generated from **1-iter Lloyd CVT** spot kernels (`f1`) modulated by regional density clustering (`cluster`) and edge wobble, strictly matching spec coverage percentages via conditional quantiles (**10% Black, 20% Red-Brown, 35% Moss Green, 15% Gray-Green/Sage, 20% Light Olive**) over $2\times 2$ twill `woven_fabric_field`.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.75\,\text{weave} + 0.25\,\text{slub})$.
- **Source**: German Bundeswehr Technical Standard **TL 8305-0290** (5-color Flecktarn ratios 35/20/20/15/10%); Lloyd (1982) CVT; Peirce (1937).

### 65. `multicam-camo` (`tools/generators/types/multicam_camo.py`)
- **Construction**: **Crye MultiCam®** dual-layer pattern: continuous multi-stop low-frequency tan/olive/cream **gradient background** (`grad`, `grad_b`), organic smoothstep-feathered medium blotches (`b_sage, b_brown, b_cream`), and clustered dark-brown/cream **Voronoi micro-twigs** (`f1_l, f1_d`) over a **1/1 8-thread NYCO ripstop** substrate (`woven_fabric_field`, `plain_11`, `threads` multiple of `8`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.68\,\text{weave} + 0.16\,\text{rip\_grid} + 0.16\,\text{slub})$.
- **Source**: Crye Precision MultiCam® optical continuous-gradient + micro-twig pattern principles; Peirce (1937) 1/1 ripstop.

### 66. `urban-camo` (`tools/generators/types/urban_camo.py`)
- **Construction**: M81-derived urban greyscale/slate geometry enforcing **MIL-DTL-44436D** conditional area fractions (`16%` Charcoal Black, `34%` Dark Slate, `30%` Mid Concrete Gray, `20%` Pale Ash) over anisotropic domain-warped fBm fields on a $2\times 2$ twill substrate (`woven_fabric_field`, `threads` multiple of `4`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.76\,\text{weave} + 0.24\,\text{slub})$.
- **Source**: U.S. Spec **MIL-DTL-44436D** M81 pattern geometry adapted to urban monochrome stops & Peirce (1937) 2/2 twill.

---

## Family 7: Environment Surfaces — Rocks, Ground & Walls (15 Types — 15 Modified, 0 Unchanged)

### 67. `granite-rock` (`tools/generators/types/granite_rock.py`)
- **Construction**: Two-scale interlocking crystal network via **1-iter Lloyd CVT** (`coarse`, `fine`) whose mineral phase fractions enforce **IUGS QAPF Field 3 Granite modal bounds**: `12%` dark Biotite mica (`cid < 0.12`), `25%` Plagioclase (`0.12..0.37`), `35%` K-Feldspar (`0.37..0.72`), and `28%` low-roughness vitreous Quartz (`cid >= 0.72`, `rough=0.22`). Each grain displays a randomized planar crystallographic cleavage tilt (`cleavage`) intersected by Swiss multifractal rock fractures.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.42\,\text{fracture} + 0.34\,\text{cleavage}\cdot(1-0.5 f_1) + 0.24\,\mathrm{clip}(3.5\,\text{edge}, 0, 1))$.
- **Source**: IUGS / Streckeisen QAPF Field 3 Granitoid Modal Classification; Lloyd (1982) CVT interlocking crystal packing.

### 68. `sandstone-strata` (`tools/generators/types/sandstone_strata.py`)
- **Construction**: Sedimentary **cross-bedding coset geometry**: periodic horizontal bounding-surface grooves (`n_sets`, `bed_seam`) truncate alternating-dip inclined foreset laminae (`set_dir = ±1`, `foreset_phase`), each shaped with an asymmetric differential-cementation ledge profile (gradual ramp $0.78$ / crisp undercut $0.22$) plus diagenetic **Liesegang iron-oxide staining bands** (`liesegang`) and quartz sand grit.
- **Height Field (`native`)**: $h = \mathrm{norm}((0.60\,\text{ledge} + 0.24\,\text{strata\_sub} + 0.16\,\text{grit})\cdot\text{bed\_seam})$.
- **Source**: *First principles* sedimentary cross-bedding coset truncation & differential cementation profile + Liesegang banding.

### 69. `slate-rock` (`tools/generators/types/slate_rock.py`)
- **Construction**: Metamorphic stepped slaty cleavage sheets (`coord = (v * sheets + tilt * sin(2\pi u) + 0.45*(warp - 0.5)) % sheets`) with flat stepped terraces (`plate_height`), ridged cleavage fissures (`crack_mask`), and aligned **anisotropic Gabor phyllosilicate mica foliation** (`periodic_gabor_noise`, `omega0 = 0.06 * tilt`) following the sheet dip.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.64\,\text{plate\_height} + 0.14\,\text{plate\_id} - 0.34\,\text{crack\_mask} + 0.12\,\text{foliation} + 0.06\,\text{fine\_grain})$.
- **Source**: Lagae et al. (2009) anisotropic Gabor foliation grain + *first principles* stepped metamorphic cleavage geometry.

### 70. `limestone` (`tools/generators/types/limestone.py`)
- **Construction**: Pale calcareous carbonate matrix (`cloud`, `chalk`) incised by **1-iter Lloyd CVT karst stylolite dissolution joints** (`joints = exp(-16 edge_j)`), smoothstep-beveled dissolution vug pits (`pits`), and bio-clast shell/fossil fragment specks (`fossils`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.80 + 0.16\,\text{cloud} + 0.10\,\text{chalk} - 0.50\,\text{pits} - 0.28\,\text{joints} - 0.12\,\text{fossils})$.
- **Source**: Lloyd (1982) CVT stylolite jointing + *first principles* smoothstep carbonate dissolution vug morphology.

### 71. `river-pebbles` (`tools/generators/types/river_pebbles.py`)
- **Construction**: Water-worn streambed cobbles modeled via **2-iteration Lloyd CVT** (`lloyd_iters=2`) primary rounded ovoid domes (`pebble_mask` via smoothstep, convex wear `dome`) interlocked with **1-iter Lloyd CVT secondary interstitial pebbles** (`sub_dome`) nestled in damp sandy streambed gaps (`sand_grit`).
- **Height Field (`native`)**: $h = \mathrm{norm}(\max(\text{pebble\_mask}\cdot(0.26 + 0.74\,\text{dome}),\, \text{sub\_dome}) + (1-\max(\text{pebble\_mask}, t_{\text{sub}}))\cdot 0.08\,\text{sand\_grit})$.
- **Source**: Lloyd (1982) CVT stone size-sorting & *first principles* fluvial abrasion ovoid dome relief.

### 72. `gravel` (`tools/generators/types/gravel.py`)
- **Construction**: Dual-scale angular crushed aggregate using non-Euclidean **Chebyshev ($\ell_\infty$)** and **Manhattan ($\ell_1$)** periodic Voronoi cells (`cells_a`, `cells_b`), where each angular chip receives an independent tilted **planar fracture cleavage facet** (`tilt_a`, `tilt_b` dotted with radial vectors `rxa, rya`) composited with max-height competition (`use_a`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.84\,\text{stone\_h} + 0.16\,\text{grit})$.
- **Source**: Worley (1996) $\ell_1/\ell_\infty$ cellular polyhedra + *first principles* angular crushed-rock cleavage facet planes.

### 73. `grass-field` (`tools/generators/types/grass_field.py`)
- **Construction**: Anisotropic turf bladelets generated from three wind-lay azimuths (`omega0 = 0.35, 1.45, 2.25` rad) using **Lagae-Lefebvre sparse Gabor wavelets** (`periodic_gabor_noise` with compact Gaussian envelopes eliminating global periodic grid stripes), modulated by clump fBm (`clump`, `density`) and curl/fBm leaf-lay warp (`wu, wv`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.58\,\text{blades} + 0.26\,\text{clump} + 0.16\,\text{density})$.
- **Source**: Lagae et al. (2009) *Procedural Noise using Sparse Gabor Convolution* (ACM SIGGRAPH) multi-angle anisotropic canopy.

### 74. `grass-dirt-patch` (`tools/generators/types/grass_dirt_patch.py`)
- **Construction**: Dual-angle **Lagae-Lefebvre Gabor grass bladelets** (`omega0 = 0.42, 1.78` rad) thinning organically across a smoothstep macro/edge coverage transition (`grass_mask = smoothstep(grass_zone, 0.44, 0.58)`) into compacted earth containing **1-iter Lloyd CVT** smoothstep-masked stones (`small_stones`) and loam grit.
- **Height Field (`native`)**: $h = \mathrm{norm}(h_{\text{dirt}}(1 - \text{grass\_mask}) + h_{\text{grass}}\,\text{grass\_mask})$.
- **Source**: Lagae et al. (2009) Gabor bladelets + Lloyd (1982) CVT soil stones + *first principles* smoothstep ecotone blending.

### 75. `moss-ground` (`tools/generators/types/moss_ground.py`)
- **Construction**: Cushion bryophyte colonies shaped by **2-iteration Periodic Lloyd CVT** (`lloyd_iters=2`) modeling radial vegetative outward competition into regular domed pillows (`cushion = sqrt(smoothstep(edge / 0.38))`), coated with velvety high-frequency isotropic **Lagae-Lefebvre Gabor micro-tufts** (`tuft_gabor`) and fine fBm phyllid fuzz.
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{cushion}\cdot(0.62 + 0.16\,\text{velvet} + 0.14\,\text{tuft\_gabor} + 0.08\,\text{micro\_fuzz}))$.
- **Source**: Lloyd (1982) CVT colony packing & Lagae et al. (2009) isotropic Gabor bryophyte tuft texture.

### 76. `dirt-soil` (`tools/generators/types/dirt_soil.py`)
- **Construction**: Multi-scale loam soil clods (`clods`, `crumb`, `grit`), **1-iter Lloyd CVT** embedded pebbles masked with $C^1$ smoothstep domes (`stone_mask`, `stone_dome`), and shallow desiccated crust hairline micro-fissures (`micro_crack`) from secondary 1-iter Lloyd CVT edges.
- **Height Field (`native`)**: $h = \mathrm{norm}((0.44\,\text{clods} + 0.30\,\text{crumb} + 0.10\,\text{grit})\cdot(0.86 + 0.14\,\text{micro\_crack}) + 0.35\,\text{stone\_dome})$.
- **Source**: Lloyd (1982) CVT pebble/fissure distribution & *first principles* multi-octave soil aggregate hierarchy.

### 77. `cracked-dry-mud` (`tools/generators/types/cracked_dry_mud.py`)
- **Construction**: Grounded in **Goehring et al. (2010) & Goehring (2013)**: repeated wetting-drying cycles anneal desiccation cracks from $90^\circ$ T-junctions toward a **Centroidal Voronoi Tessellation (~120° Y-junctions)**, modeled via **2-iteration Periodic Lloyd CVT** (`lloyd_iters=2`) plus hierarchical secondary cracks (`edge_s`). Smoothstep fissure walls frame upward plate-edge **differential desiccation curl** (`cup_curl = t_curl^2 * 0.22 * main_crack`).
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{plate\_mask}\cdot(0.64 + 0.14\,\text{silt} + \text{cup\_curl}))$.
- **Source**: Goehring, Conroy, Akhter, Clegg & Routh (2010) *Soft Matter*; Goehring (2013) *Phil. Trans. R. Soc. A* 371:20120353 (`arXiv:1211.6762`).

### 78. `sand` (`tools/generators/types/sand.py`)
- **Construction**: Grounded in **Bagnold (1941), Sharp (1963), & Anderson (1987)** aeolian impact-ripple mechanics: $C^1$ piecewise cosine **asymmetric saltation ripple** with gentle windward **stoss (`72%`)** and steeper leeward slope (`28%`, **Ripple Symmetry Index $\mathrm{RSI} \approx 2.6$**), superposed with broad floor undulation and **Sharp (1963) crest-sorted coarse mineral/quartz sparkle** (`crest_sort * grain_b`).
- **Height Field (`native`)**: $h = \mathrm{norm}(0.46\,\text{fine\_ripple} + 0.16\,\text{broad\_undulation} + 0.26\,\text{grain\_a} + 0.12\,\text{grain\_b}\cdot\text{crest\_sort})$.
- **Source**: Bagnold (1941) *The Physics of Blown Sand*; Anderson (1987) *Sedimentology* 34:943–956; Sharp (1963) *J. Geol.* 71:617–636.

### 79. `red-brick-wall` (`tools/generators/types/red_brick_wall.py`)
- **Construction**: Grounded in **Brick Industry Association (BIA) Technical Notes 10 (*Dimensioning and Estimating Brick Masonry*)**: standard U.S. Modular clay brick ($4 \times 2\tfrac{2}{3} \times 8\text{ in.}$ nominal, exact **$3:1$ exposed face ratio** via `cols = rows // 3` for `rows in [9, 12, 15]`) in half-staggered running bond with **$\tfrac{3}{8}\text{ in.}$ recessed mortar joint proportion (`14%` course pitch, half-margin `0.070`)**, smoothstep brick edge wear, and anisotropic horizontal **wire-cut extrusion Gabor striations** (`omega0 = 0.0`).
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{brick\_mask}\cdot\text{brick\_face} + (1 - \text{brick\_mask})\cdot\text{mortar\_h})$.
- **Source**: BIA Technical Notes 10 (`gobrick.com`); Lagae et al. (2009) Gabor wire-cut extrusion striations.

### 80. `stone-wall` (`tools/generators/types/stone_wall.py`)
- **Construction**: Aspect-scaled coursed fieldstone masonry (`_periodic_rect_voronoi`, `nx, ny`) with smoothstep-beveled stone perimeters (`stone_mask`), convex rock cushion domes (`stone_dome`), and **randomized per-stone planar rock cleavage tilt** (`cleavage = tx * fx + ty * fy`) overlaid by Swiss multifractal fracture grain.
- **Height Field (`native`)**: $h = \mathrm{norm}(\text{stone\_mask}\cdot\text{stone\_dome} + (1 - \text{stone\_mask})\cdot 0.12\,\text{grit})$.
- **Source**: Worley (1996) aspect-scaled cellular courses + *first principles* planar cleavage facet tilt & smoothstep mortar bevel.

### 81. `stucco-plaster-wall` (`tools/generators/types/stucco_plaster_wall.py`)
- **Construction**: Replaces cellular straight ridges with physically accurate **overlapping circular steel trowel-blade arc sweeps**: $N = 5 \cdot \text{trowel\_cells}$ toroidal arc passes centered at $(c_x, c_y)$ with blade radius $R \in [0.14, 0.28]$, directional sweep angular envelope `(0.5 + 0.5 cos(\theta - \theta_0))^{2.2}`, Gaussian blade-edge bead `exp(-((r - R)^2) / 0.00055)`, broad skim-coat swell, fBm lime wash, and fine silica sand float.
- **Height Field (`native`)**: $h = \mathrm{norm}(0.42\,\text{lime\_wash} + 0.34\,\text{trowel\_ridge} + 0.16\,\text{skim\_swell} + 0.08\,\text{sand\_float})$.
- **Source**: *First principles* curved circular trowel-blade sweep ridge kinematics + lime-wash/sand-float fBm hierarchy.
