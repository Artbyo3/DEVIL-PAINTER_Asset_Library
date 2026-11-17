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
    # --- Cloth & Apparel Textures (17) ---
    "denim_twill",
    "jersey_knit",
    "cable_knit",
    "tartan_plaid",
    "houndstooth_wool",
    "quilted_puffer",
    "corduroy",
    "silk_brocade",
    "waffle_knit",
    "ripstop_nylon",
    "herringbone_weave",
    "rib_jersey_knit",
    "boucle_tweed",
    "jacquard_geometric",
    "seersucker_shibori",
    "pebbled_leather",
    "garment_wrinkles",
    # --- Cloth, Seam & Lace Alphas (12) ---
    "double_topstitch_seam",
    "zipper_teeth_trim",
    "overlock_zigzag_seam",
    "four_hole_button",
    "ribbed_cuff_trim",
    "cross_stitch_seam",
    "rivet_grommet",
    "scalloped_lace_trim",
    "lace_mesh_mask",
    "smocked_gather",
    "belt_buckle",
    "blanket_stitch_seam",
    # --- Consolidated Surface Material Textures (11) ---
    "domain_warped_marble",
    "swiss_erosion_rock",
    "voronoi_basalt_crackle",
    "hardwood_planks",
    "masonry_tiles",
    "damascus_carbon_metal",
    "scifi_greeble_panels",
    "biome_bark_foliage",
    "crystalline_geode_agate",
    "stylized_arcane_cel",
    "turing_reaction_diffusion",
    # --- Specialized & Curated Surface Textures (11) ---
    "terrazzo",
    "hex_mosaic",
    "wind_dunes",
    "pcb_circuit",
    "stained_glass",
    "hammered_metal",
    "dragon_scales",
    "fluted_travertine",
    "cobblestone_paving",
    "iridescent_thinfilm",
    "weathered_concrete_patina",
    # --- Substance Utility, Mask & Noise Maps (4) ---
    "grunge_wear_mask",
    "procedural_noise_map",
    "baking_utility_map",
    "matcap_sphere",
    # --- Sculpting & Mechanical Decal Alphas (2) ---
    "sculpt_stamp_alpha",
    "mechanical_decal_alpha",
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
