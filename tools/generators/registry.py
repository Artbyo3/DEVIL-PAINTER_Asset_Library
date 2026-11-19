"""Single canonical registry of all procedural texture and alpha generators.

Adding a future generator type requires ONLY:
  1. Dropping one new module into ``tools/generators/types/<module_name>.py``
     exposing ``SPEC: GeneratorSpec``.
  2. Adding ``"<module_name>"`` as one line in ``REGISTERED_MODULES`` below.
Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import importlib

try:
    from .interface import GeneratorSpec
except ImportError:
    from interface import GeneratorSpec  # type: ignore[no-redef]

# One line per generator module in tools/generators/types/.
REGISTERED_MODULES: tuple[str, ...] = (
    # --- Cloth & Apparel Textures (17: 17 NATIVE) ---
    "denim_twill",                # NATIVE: twill weave + yarn slub height
    "jersey_knit",                # NATIVE: interlocking V-loop knit height
    "cable_knit",                 # NATIVE: braided cable twist + rib height
    "tartan_plaid",               # NATIVE: intersecting warp/weft twill height
    "houndstooth_wool",           # NATIVE: wool twill weave + fuzz height
    "quilted_puffer",             # NATIVE: quilted pillow + seam + stitch height
    "corduroy",                   # NATIVE: vertical wale pile height
    "silk_brocade",               # NATIVE: damask motif + satin weave height
    "waffle_knit",                # NATIVE: honeycomb waffle pocket height
    "ripstop_nylon",              # NATIVE: ripstop grid + micro weave height
    "herringbone_weave",          # NATIVE: chevron twill thread height
    "rib_jersey_knit",            # NATIVE: rib column + knit loop height
    "boucle_tweed",               # NATIVE: slubbed boucle yarn + weave height
    "jacquard_geometric",         # NATIVE: jacquard motif float height
    "seersucker_shibori",         # NATIVE: seersucker pucker + weave height
    "pebbled_leather",            # NATIVE: Voronoi pebble grain + pore height
    "leather_suede",              # NATIVE: 12 apparel leather & suede variants + calibrated roughness
    "knits_fleece",               # NATIVE: 12 apparel knit & fleece variants + calibrated roughness
    "garment_wrinkles",           # NATIVE: tension fold, pleat, crumple height
    # --- Cloth, Seam & Lace Alphas (12: 11 NATIVE, 1 DERIVED) ---
    "double_topstitch_seam",      # NATIVE: folded seam step + pucker + stitches
    "zipper_teeth_trim",          # NATIVE: interlocking teeth + cord bead + tape
    "overlock_zigzag_seam",       # NATIVE: zigzag serger thread + border rails
    "four_hole_button",           # NATIVE: beveled dish + rim + holes + X-stitch
    "ribbed_cuff_trim",           # NATIVE: ribbed knit cuff wave relief
    "cross_stitch_seam",          # NATIVE: diagonal X-stitch thread relief
    "rivet_grommet",              # NATIVE: beveled grommet ring + prongs + puckers
    "scalloped_lace_trim",        # NATIVE: scalloped rim + eyelet embroidery relief
    "lace_mesh_mask",             # DERIVED: cutout/opacity/tear mask treated as relief
    "smocked_gather",             # NATIVE: pleated smocked fabric gather relief
    "belt_buckle",                # NATIVE: beveled buckle frame + pin relief
    "blanket_stitch_seam",        # NATIVE: blanket stitch spine + bar + ridge
    # --- Consolidated Surface Material Textures (11: 10 NATIVE, 1 DERIVED) ---
    "domain_warped_marble",       # DERIVED: domain-warped mineral vein scalar field
    "swiss_erosion_rock",         # NATIVE: Swiss ridged erosion elevation
    "voronoi_basalt_crackle",     # NATIVE: dual-scale basalt column + fissure height
    "hardwood_planks",            # NATIVE: plank bevel + wood grain + pore height
    "masonry_tiles",              # NATIVE: beveled stone/brick + recessed mortar height
    "damascus_carbon_metal",      # NATIVE: etched steel / carbon weave relief
    "scifi_greeble_panels",       # NATIVE: recessed panel seams + bevel + bolt height
    "biome_bark_foliage",         # NATIVE: bark fissures / scale relief height
    "crystalline_geode_agate",    # NATIVE: crystalline facet + growth band height
    "stylized_arcane_cel",        # NATIVE: stylized blade / caustic / slab height
    "turing_reaction_diffusion",  # NATIVE: coral/labyrinth Turing ridge height
    # --- Specialized & Curated Surface Textures (11: 11 NATIVE) ---
    "terrazzo",                   # NATIVE: raised aggregate stone chips in binder
    "hex_mosaic",                 # NATIVE: beveled hex tiles + recessed grout
    "wind_dunes",                 # NATIVE: asymmetric dune slip-face + ripples
    "pcb_circuit",                # NATIVE: raised copper traces, pads, via rings
    "stained_glass",              # NATIVE: raised lead came borders + rippled glass
    "hammered_metal",             # NATIVE: planished concave Voronoi hammer dimples
    "dragon_scales",              # NATIVE: overlapping keeled scales + sub-ridges
    "fluted_travertine",          # NATIVE: fluted stone ribs + travertine pores
    "cobblestone_paving",         # NATIVE: rounded cobble domes + recessed mortar
    "iridescent_thinfilm",        # NATIVE: thin-film fluid ripple height
    "weathered_concrete_patina",  # NATIVE: cracks, paint peel, spangle, strata, tiles
    # --- Substance Utility, Mask & Noise Maps (4: 1 NATIVE, 3 DERIVED) ---
    "grunge_wear_mask",           # DERIVED: 2D scratch/drip/peel/crack mask scalar
    "procedural_noise_map",       # DERIVED: dimensionless mathematical noise scalar
    "baking_utility_map",         # NATIVE: underlying surface height h from _base_surface_fields
    "matcap_sphere",              # DERIVED: hemisphere z = sqrt(max(0, 1 - r^2)) mask
    # --- Sculpting & Mechanical Decal Alphas (2: 2 NATIVE) ---
    "sculpt_stamp_alpha",         # NATIVE: hex bolt, crater, chisel, weld, sigil height
    "mechanical_decal_alpha",     # NATIVE: gear, louver, claw, shatter, rosette, scale, dent height
    # --- Military Camouflage Textures (9: 9 NATIVE) ---
    "woodland_camo",              # NATIVE: woven twill + yarn slub cloth height
    "desert_camo",                # NATIVE: woven twill + yarn slub cloth height
    "chocolate_chip_camo",        # NATIVE: woven twill + yarn slub cloth height
    "digital_woodland_camo",      # NATIVE: MARPAT ripstop/twill + slub cloth height
    "digital_desert_camo",        # NATIVE: MARPAT ripstop/twill + slub cloth height
    "tiger_stripe_camo",          # NATIVE: woven twill + yarn slub cloth height
    "flecktarn_camo",             # NATIVE: woven twill + yarn slub cloth height
    "multicam_camo",              # NATIVE: woven twill + yarn slub cloth height
    "urban_camo",                 # NATIVE: woven twill + yarn slub cloth height
    # --- Environment Surfaces: Rocks, Ground & Walls (15: 15 NATIVE) ---
    "granite_rock",               # NATIVE: interlocking mineral crystals + micro-fissures
    "sandstone_strata",           # NATIVE: eroded sedimentary bedding ledges
    "slate_rock",                 # NATIVE: stepped metamorphic cleavage sheets
    "limestone",                  # NATIVE: weathered blocky joints + dissolution pits
    "river_pebbles",              # NATIVE: packed rounded pebble domes + sand bed
    "gravel",                     # NATIVE: dense angular crushed rock chips
    "grass_field",                # NATIVE: layered grass blade ridge canopy
    "grass_dirt_patch",           # NATIVE: turf elevation transitioning to bare soil
    "moss_ground",                # NATIVE: pillowy moss cushions + tufted canopy
    "dirt_soil",                  # NATIVE: clumped soil aggregates + granular grit
    "cracked_dry_mud",            # NATIVE: sunken desiccation cracks + curled clay plates
    "sand",                       # NATIVE: periodic wind-blown sand ripple field
    "red_brick_wall",             # NATIVE: running-bond bricks + recessed mortar joints
    "stone_wall",                 # NATIVE: coursed rough-hewn stone blocks + mortar
    "stucco_plaster_wall",        # NATIVE: sweeping trowel swells + fine sand float
)


def _load_registry() -> dict[str, GeneratorSpec]:
    specs: dict[str, GeneratorSpec] = {}
    pkg_base = f"{__package__}.types" if __package__ else "types"
    for mod_name in REGISTERED_MODULES:
        module = importlib.import_module(f".{mod_name}", package=pkg_base)
        spec = getattr(module, "SPEC", None)
        if not isinstance(spec, GeneratorSpec):
            raise TypeError(f"Module {mod_name!r} does not expose a valid SPEC: GeneratorSpec")
        if spec.name in specs:
            raise ValueError(f"Duplicate generator name registered: {spec.name!r}")
        specs[spec.name] = spec
    return specs


GENERATORS: dict[str, GeneratorSpec] = _load_registry()
NORMAL_SOURCES: dict[str, str] = {name: spec.normal_source.upper() for name, spec in GENERATORS.items()}


def get_generator(name: str) -> GeneratorSpec:
    key = name.strip().lower().replace("_", "-")
    if key in GENERATORS:
        return GENERATORS[key]
    raise KeyError(f"Unknown generator type {name!r}. Run with --list to see all {len(GENERATORS)} registered types.")


def list_generators(kind: str | None = None) -> list[GeneratorSpec]:
    items = list(GENERATORS.values())
    if kind is not None:
        k = kind.strip().lower()
        items = [s for s in items if s.kind == k]
    return items


def normal_source_counts() -> dict[str, int]:
    native = sum(1 for spec in GENERATORS.values() if spec.normal_source == "native")
    derived = sum(1 for spec in GENERATORS.values() if spec.normal_source == "derived")
    return {"native": native, "derived": derived, "total": len(GENERATORS)}
