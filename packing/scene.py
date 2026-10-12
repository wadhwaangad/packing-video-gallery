"""Apartment, furniture, suitcase and packed items: one source for physics and rendering.

World frame: z up, metres. The apartment interior spans x 0..7, y 0..6.4.
Poly Haven glTF assets are y-up; Blender's importer converts them to z-up with
the asset front facing -y. `yaw` below rotates that front about +z.
"""
from dataclasses import dataclass, field
import numpy as np

WALL_T, WALL_H = 0.10, 2.60


@dataclass
class Box:
    name: str
    center: tuple
    size: tuple            # full extents
    material: str
    collide: bool = True
    yaw: float = 0.0


@dataclass
class Asset:
    name: str
    asset: str
    pos: tuple             # floor point under the asset origin
    yaw: float
    collision: list = field(default_factory=list)   # world-frame Boxes


@dataclass
class Item:
    name: str
    label: str
    kind: str              # garment | bag | asset
    size: tuple            # collision box full extents (local x, y, z)
    mass: float
    material: str
    start: tuple           # centre position on its source surface
    start_yaw: float
    room: str
    asset: str = ""
    tint: tuple = (1, 1, 1)
    soft: bool = False
    mesh_offset: tuple = (0, 0, 0)   # asset origin in the item body frame
    mesh_objects: tuple = ()         # asset objects to render (empty = all)


# ---------------------------------------------------------------- rooms
ROOMS = {
    "bedroom": dict(x=(0.0, 3.8), y=(0.0, 3.4), floor="oak_wood_planks"),
    "bathroom": dict(x=(3.8, 5.6), y=(0.0, 2.2), floor="long_white_tiles"),
    "kitchen": dict(x=(5.6, 7.0), y=(0.0, 2.2), floor="long_white_tiles"),
    "hall": dict(x=(3.8, 7.0), y=(2.2, 3.4), floor="oak_wood_planks"),
    "living": dict(x=(0.0, 7.0), y=(3.4, 6.4), floor="oak_wood_planks"),
}


def _wall(name, a, b, gaps=()):
    """Axis-aligned wall from a to b (2D) with door/window gaps [(s0, s1, z0, z1)]."""
    a, b = np.array(a, float), np.array(b, float)
    horizontal = abs(b[1] - a[1]) < 1e-9
    length = np.linalg.norm(b - a)
    pieces = []
    cuts = sorted(gaps)
    s = 0.0
    def seg(s0, s1, z0, z1, tag):
        if s1 - s0 < 1e-6 or z1 - z0 < 1e-6:
            return
        mid = a + (b - a) * ((s0 + s1) / 2 / length)
        size = (s1 - s0, WALL_T, z1 - z0) if horizontal else (WALL_T, s1 - s0, z1 - z0)
        pieces.append(Box(f"{name}_{tag}", (mid[0], mid[1], (z0 + z1) / 2), size, "wall"))
    for i, (g0, g1, z0, z1) in enumerate(cuts):
        seg(s, g0, 0, WALL_H, f"s{i}")
        seg(g0, g1, 0, z0, f"below{i}")
        seg(g0, g1, z1, WALL_H, f"above{i}")
        s = g1
    seg(s, length, 0, WALL_H, "end")
    return pieces


DOOR = (0.0, 2.05)
WALLS = (
    _wall("wall_south", (0, 0), (7, 0), [(1.2, 2.6, 0.85, 2.15), (6.0, 6.7, 1.0, 2.0)])
    + _wall("wall_north", (0, 6.4), (7, 6.4), [(1.0, 3.2, 0.7, 2.25), (4.6, 6.2, 0.7, 2.25)])
    + _wall("wall_west", (0, 0), (0, 6.4))
    + _wall("wall_east", (7, 0), (7, 6.4))
    # bedroom/living partition with the bedroom door at x 2.75..3.65
    + _wall("wall_bed_north", (0, 3.4), (3.8, 3.4), [(2.75, 3.65, *DOOR)])
    # bedroom/bathroom partition, full height
    + _wall("wall_bed_east", (3.8, 0), (3.8, 3.4))
    # bathroom north wall with door at x 4.15..4.95
    + _wall("wall_bath_north", (3.8, 2.2), (5.6, 2.2), [(0.35, 1.15, *DOOR)])
    + _wall("wall_bath_east", (5.6, 0), (5.6, 2.2))
)
WINDOWS = [  # (wall, s0, s1, z0, z1) for glazing in the render
    ("wall_south", 1.2, 2.6, 0.85, 2.15), ("wall_south", 6.0, 6.7, 1.0, 2.0),
    ("wall_north", 1.0, 3.2, 0.7, 2.25), ("wall_north", 4.6, 6.2, 0.7, 2.25),
]

TABLE_H = 0.39
# ---------------------------------------------------------------- furniture
BED = dict(x=(0.05, 2.15), y=(0.95, 2.55), frame_h=0.30, mattress_top=0.56)
PROCEDURAL = [
    # bed frame and mattress (mattress is compliant in physics)
    Box("bed_frame", (1.10, 1.75, 0.15), (2.10, 1.60, 0.30), "bed_frame"),
    Box("mattress", (1.11, 1.75, 0.43), (2.04, 1.54, 0.26), "mattress"),
    Box("headboard", (0.08, 1.75, 0.60), (0.06, 1.66, 1.20), "bed_frame"),
    # bathroom vanity along the south wall
    Box("vanity", (4.70, 0.29, 0.42), (1.10, 0.48, 0.84), "vanity"),
    Box("vanity_top", (4.70, 0.29, 0.855), (1.14, 0.52, 0.03), "counter"),
    Box("towel_shelf", (5.39, 1.15, 0.95), (0.32, 0.60, 0.03), "vanity"),
    Box("towel_shelf_lower", (5.39, 1.15, 0.45), (0.32, 0.60, 0.03), "vanity"),
    Box("towel_shelf_side", (5.39, 1.15, 0.48), (0.30, 0.03, 0.96), "vanity", collide=False),
    # kitchen counter along the east wall
    Box("kitchen_base", (6.68, 1.10, 0.43), (0.60, 2.00, 0.86), "vanity"),
    Box("kitchen_top", (6.67, 1.10, 0.885), (0.64, 2.04, 0.03), "counter"),
    # rug in front of the sofa (visual only)
    Box("rug", (2.10, 4.95, 0.004), (2.4, 1.7, 0.008), "rug", collide=False),
    Box("bedroom_rug", (2.75, 1.75, 0.004), (1.0, 1.6, 0.008), "rug", collide=False),
]


def _asset_box(name, center_xy, footprint, height, yaw=0.0):
    sx, sy = footprint
    if abs(np.sin(yaw)) > 0.7:
        sx, sy = sy, sx
    return Box(name + "_coll", (center_xy[0], center_xy[1], height / 2), (sx, sy, height), "none")


ASSETS = [
    # Dresser against the east bedroom wall, front faces -x. Height .545, width .858, depth .442
    Asset("dresser", "vintage_wooden_drawer_01", (3.53, 1.20, 0), -np.pi / 2,
          [_asset_box("dresser", (3.53, 1.20), (0.858, 0.442), 0.545, -np.pi / 2)]),
    Asset("nightstand", "ClassicNightstand_01", (0.36, 2.95, 0), np.pi,
          [_asset_box("nightstand", (0.36, 2.95), (0.568, 0.424), 0.70)]),
    Asset("bed_plant", "potted_plant_02", (0.45, 0.42, 0), 0.0, []),
    # Living room: sofa against the north wall faces -y
    Asset("sofa", "sofa_03", (2.10, 5.90, 0), 0.0,
          [_asset_box("sofa", (2.10, 5.90), (2.731, 0.925), 0.55)]),
    Asset("pillows", "throw_pillows_01", (2.10, 5.95, 0.42), 0.0, []),
    Asset("coffee_table", "modern_coffee_table_01", (2.10, 4.85, 0), np.pi / 2,
          [_asset_box("coffee_table", (2.10, 4.85), (0.6, 1.202), 0.39, np.pi / 2)]),
    Asset("bookshelf", "wooden_bookshelf_worn", (0.33, 4.90, 0), -np.pi / 2,
          [_asset_box("bookshelf", (0.33, 4.90), (1.374, 0.581), 2.06, -np.pi / 2)]),
    Asset("tv_stand", "side_table_01", (2.10, 3.75, 0), np.pi,
          [_asset_box("tv_stand", (2.10, 3.75), (0.55, 0.45), 0.55)]),
    Asset("television", "Television_01", (2.10, 3.75, 0.549), np.pi, []),
    Asset("living_plant", "potted_plant_04", (6.60, 6.05, 0), 0.0, []),
    Asset("alarm_clock", "alarm_clock_01", (0.30, 2.95, 0.70), np.pi * 0.9, []),
    Asset("open_binder", "binder_notebook", (2.38, 5.0, TABLE_H), 0.25, []),
]

# ---------------------------------------------------------------- suitcase
# Medium hard-shell spinner, 66 x 44 x 26 cm. Local frame: x across the width,
# y along the length (+y = top end with the carry handle, -y = wheel end),
# z out of the back panel. The lid hinges on the -x long edge.
SUITCASE = dict(
    outer=(0.44, 0.66), base_depth=0.155, lid_depth=0.105, shell=0.006,
    base_mass=2.6, lid_mass=1.2, hinge_side=-1,
    # Lying open on the bed. yaw=-pi/2 maps local -x (hinge side) to world +y, so the
    # lid lies open towards the bed's far side, the operator packs from y~0.6, and the
    # top handle (local +y) faces the foot of the bed.
    pos=(1.55, 1.21, BED["mattress_top"]), yaw=-np.pi / 2,
    color=(0.03, 0.13, 0.20),
)

# ---------------------------------------------------------------- items
DRESSER_TOP = 0.545
COUNTER = 0.87
SHELF = 0.965
TABLE = 0.39
KITCHEN = 0.90

ITEMS = [
    Item("jeans", "folded jeans", "garment", (0.32, 0.25, 0.05), 0.70, "denim_fabric",
         (3.52, 0.98, DRESSER_TOP + 0.025), np.pi / 2, "bedroom", soft=True),
    Item("tshirt_white", "white T-shirt", "garment", (0.30, 0.23, 0.03), 0.20, "cotton_jersey",
         (3.52, 1.42, DRESSER_TOP + 0.015), np.pi / 2 + 0.05, "bedroom", tint=(0.92, 0.92, 0.9), soft=True),
    Item("tshirt_navy", "navy T-shirt", "garment", (0.30, 0.23, 0.03), 0.20, "cotton_jersey",
         (3.52, 1.42, DRESSER_TOP + 0.045), np.pi / 2 - 0.04, "bedroom", tint=(0.10, 0.14, 0.30), soft=True),
    Item("sweater", "knit sweater", "garment", (0.32, 0.25, 0.055), 0.50, "knitted_fleece",
         (0.72, 1.17, BED["mattress_top"] + 0.0275), 0.04, "bedroom", tint=(0.80, 0.72, 0.60), soft=True),
    Item("towel", "folded towel", "garment", (0.30, 0.20, 0.045), 0.45, "waffle_pique_cotton",
         (5.39, 1.15, SHELF + 0.0225), np.pi / 2, "bathroom", tint=(0.95, 0.95, 0.93), soft=True),
    Item("toiletry_bag", "toiletry bag", "bag", (0.23, 0.14, 0.09), 0.65, "toiletry",
         (4.55, 0.30, COUNTER + 0.045), 0.0, "bathroom", tint=(0.12, 0.25, 0.20)),
    Item("bottle", "water bottle", "bottle", (0.072, 0.072, 0.23), 0.55, "bottle",
         (6.62, 0.75, KITCHEN + 0.115), 0.0, "kitchen", tint=(0.36, 0.45, 0.38)),
    Item("notebook", "leather notebook", "asset", (0.17, 0.20, 0.025), 0.35, "asset",
         (1.88, 4.78, TABLE + 0.0125), 0.0, "living", asset="binder_notebook",
         mesh_offset=(-0.342, 0, -0.0105), mesh_objects=("binder_notebook_closed",)),
    Item("camera", "camera", "asset", (0.148, 0.035, 0.078), 0.62, "asset",
         (2.32, 4.80, TABLE + 0.039), 0.0, "living", asset="Camera_01",
         mesh_offset=(0, -0.0015, -0.039), mesh_objects=("Camera_01",)),
]

# Final placement in suitcase-base local coordinates (item centre, yaw about local z).
# Inner base: x +-0.214, y +-0.324, z 0.006..0.155; the closed lid adds 0.099.
# The garment stack (+ notebook) lies on the hinge side with its long side along the
# suitcase length, so both hands squeeze it from the operator's left and right. The
# strip on the operator's side holds the toiletry bag (camera on top) and the water
# bottle standing upright; hands never reach across a tall pile.
STACK_X, STRIP_X = -0.080, 0.130
PACK_SLOTS = {
    "jeans":        dict(pos=(STACK_X, 0.0, 0.006 + 0.025), yaw=np.pi / 2),
    "tshirts":      dict(pos=(STACK_X, 0.0, 0.056 + 0.030), yaw=np.pi / 2),
    "sweater":      dict(pos=(STACK_X, 0.0, 0.116 + 0.0275), yaw=np.pi / 2),
    "towel":        dict(pos=(STACK_X, 0.0, 0.171 + 0.0225), yaw=np.pi / 2),
    "notebook":     dict(pos=(STACK_X, 0.0, 0.216 + 0.0125), yaw=np.pi / 2),
    "toiletry_bag": dict(pos=(STRIP_X, 0.130, 0.006 + 0.045), yaw=np.pi / 2),
    "camera":       dict(pos=(STRIP_X, 0.130, 0.096 + 0.039), yaw=np.pi / 2),
    "bottle":       dict(pos=(STRIP_X + 0.01, -0.200, 0.006 + 0.115), yaw=0.0),
}
# Trips: (destination room, items carried together, grasp)
TRIPS = [
    ("bedroom", ["jeans"], "squeeze"),
    ("bedroom", ["tshirt_white", "tshirt_navy"], "squeeze"),
    ("bedroom", ["sweater"], "squeeze"),
    ("bathroom", ["towel"], "squeeze"),
    ("bathroom", ["toiletry_bag"], "squeeze"),
    ("kitchen", ["bottle"], "wrap"),
    ("living", ["notebook"], "squeeze"),
    ("living", ["camera"], "squeeze"),
]


def all_static_boxes():
    boxes = list(WALLS) + [b for b in PROCEDURAL]
    for a in ASSETS:
        boxes += a.collision
    return boxes


def item(name):
    return next(i for i in ITEMS if i.name == name)
