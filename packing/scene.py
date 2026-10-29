"""Apartment, furniture, suitcase and packed items: one source for physics and rendering.

World frame: z up, metres. The apartment interior spans x 0..7, y 0..6.4.
Poly Haven glTF assets are y-up; Blender's importer converts them to z-up with
the asset front facing -y. `yaw` below rotates that front about +z.
"""
from dataclasses import dataclass, field
from functools import lru_cache
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
    scale: float = 1.0     # uniform scale of the model (its collision boxes are already in metres)


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
BED = dict(x=(0.05, 2.15), y=(0.95, 2.55), frame_h=0.26, mattress_top=0.50)   # low platform bed
PROCEDURAL = [
    # bed frame and mattress (mattress is compliant in physics)
    Box("bed_frame", (1.10, 1.75, 0.13), (2.10, 1.60, 0.26), "bed_frame"),
    Box("mattress", (1.11, 1.75, 0.38), (2.04, 1.54, 0.24), "mattress"),
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
    # Dresser against the east bedroom wall, front faces -x. Height .545, width .858, depth .442:
    # the model is 1.715 x 1.091 m, so it is drawn at half size to match (scale).
    Asset("dresser", "vintage_wooden_drawer_01", (3.53, 1.20, 0), -np.pi / 2,
          [_asset_box("dresser", (3.53, 1.20), (0.858, 0.442), 0.545, -np.pi / 2)], scale=0.5),
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
    Asset("alarm_clock", "alarm_clock_01", (0.17, 3.06, 0.70), np.pi * 0.9, []),
    Asset("open_binder", "binder_notebook", (2.38, 5.0, TABLE_H), 0.25, []),
]

# ---------------------------------------------------------------- suitcase
# The case is a scanned black spinner (BlenderKit "Black suitcase", royalty-free),
# scaled 1.4x into a large check-in case and cut at its zip seam into base and lid
# (prepare_assets.py). The physics shell below uses the scan's measured extents.
# Local frame: x across the width, y along the length (+y = top end with the trolley
# handle, -y = wheel end), z out of the back panel. The lid hinges on the -x long edge.
SUITCASE_SCAN = dict(
    asset="ebbe0d72-f5b0-4875-b91e-72457dbfbe09", name="Black suitcase", author="L'ANE", scale=1.4,
    x_centre=-0.002, z_bottom=0.062, z_top=0.560,      # shell extents in scan units (z up)
    back_y=0.142, seam_y=0.035, front_y=-0.137,          # back panel, zip seam, front panel (scan y)
    wheel_seam_y=0.020,                                  # casters split between the halves here
    trolley_drop=0.045,                                  # telescopic handle lowered to its first stop (scan units)
    trolley_shift=0.0136,                                # ... and moved off the back panel, so fingers fit under the bar
)


def _scan_dims():
    c = SUITCASE_SCAN; s = c["scale"]
    W = 0.362 * s
    L = (c["z_top"] - c["z_bottom"]) * s
    return round(W, 4), round(L, 4), round((c["back_y"] - c["seam_y"]) * s, 4), round((c["seam_y"] - c["front_y"]) * s, 4)


_W, _L, _D, _DL = _scan_dims()
SUITCASE = dict(
    outer=(_W, _L), base_depth=_D, lid_depth=_DL,
    # The fabric liner sits 12 mm inside the scanned shell; the physics walls and floors
    # are the liner surfaces.
    shell=0.012, floor=0.018, lid_floor=0.040,     # the lid liner clears the scan's front-pocket hardware
    # Fabric hinge: the axis sits at mid-depth so both halves lie flat when open. The lid
    # frame has its origin on the hinge axis; the lid rim is at lid z = base_depth - hinge_h.
    hinge_h=round((_D + _DL) / 2, 4),
    base_mass=3.0, lid_mass=1.6, hinge_side=-1,
    # Trolley bar (retracted): axis along x, centre (y, z) in the base frame, radius, half length.
    bar=dict(x=-0.012, y=_L / 2 + 0.0841, z=0.060, r=0.016, half=0.088),
    # Spinner casters: sphere radius and centres (x, y, z) in the base / lid frames.
    wheel_r=0.035, wheel_x=0.21, wheel_y=-_L / 2 - 0.052, wheel_z_base=0.094, wheel_z_lid=0.091,
    # Lying open on the bed. yaw=-pi/2 maps local -x (hinge side) to world +y, so the
    # lid lies open towards the bed's far side, the person packs from y~0.64, and the
    # trolley handle (local +y) faces the foot of the bed.
    pos=(1.55, 1.26, BED["mattress_top"]), yaw=-np.pi / 2,
    color=(0.03, 0.03, 0.035),
)
SUIT_W, SUIT_L = SUITCASE["outer"]
INNER_H = SUITCASE["base_depth"] + SUITCASE["lid_depth"] - SUITCASE["floor"] - SUITCASE["lid_floor"]


# ---------------------------------------------------------------- stations
@dataclass
class Station:
    """A place the person stands at and acts on. `normal` is the direction they face;
    items there are picked with their short side (the squeeze axis) across it. `access`
    is the walking-graph node the station is reached from; `crouch` is the pelvis height
    while reaching."""
    name: str
    label: str
    room: str
    normal: tuple
    standoff: float
    crouch: float
    top: float = 0.0
    access: str = ""
    stand: tuple = ()          # default stand point (x, y) when no item is targeted
    via: tuple = ()            # waypoints between the access node and the stand point


DRESSER_TOP = 0.545
NIGHTSTAND_TOP = 0.70
COUNTER = 0.87
SHELF = 0.965
TABLE = 0.39
KITCHEN = 0.90
MATTRESS = BED["mattress_top"]

STATIONS = {s.name: s for s in [
    Station("dresser", "the dresser", "bedroom", (1, 0), 0.57, 0.80, DRESSER_TOP, "bed_corner", (2.95, 1.20)),
    Station("bed", "the bed", "bedroom", (0, 1), 0.53, 0.78, MATTRESS, "suitcase", (0.72, 0.64)),
    Station("bed_far", "the far side of the bed", "bedroom", (0, -1), 0.55, 0.78, MATTRESS, "bed_north", (0.95, 2.70)),
    Station("nightstand", "the nightstand", "bedroom", (-1, 0), 0.55, 0.86, NIGHTSTAND_TOP, "bed_north", (1.00, 2.92)),
    Station("vanity", "the bathroom vanity", "bathroom", (0, -1), 0.56, 0.90, COUNTER, "bath_in", (4.55, 0.86),
            via=((4.55, 1.20),)),
    Station("towel_shelf", "the towel shelf", "bathroom", (1, 0), 0.47, 0.90, SHELF, "bath_in", (4.92, 1.15),
            via=((4.70, 1.15),)),
    Station("kitchen_counter", "the kitchen counter", "kitchen", (1, 0), 0.56, 0.90, KITCHEN, "kitchen_in", (6.06, 1.10)),
    Station("coffee_table", "the coffee table", "living", (0, 1), 0.56, 0.62, TABLE, "living_mid", (2.10, 4.22)),
    # The open suitcase is packed from beside the bed; the lid and the final lift are
    # handled from the foot of the bed.
    Station("suitcase", "the suitcase", "bedroom", (0, 1), 0.0, 0.80, MATTRESS, "suitcase", (1.55, 0.64)),
    Station("bed_foot", "the foot of the bed", "bedroom", (-1, 0), 0.0, 0.86, MATTRESS, "bed_foot", (2.48, 1.80)),
]}

# Walking graph: straight segments that clear walls, doorways and furniture.
NAV_NODES = {
    "suitcase": (1.55, 0.64), "bed_corner": (2.45, 0.62), "bed_foot": (2.48, 1.80),
    "bed_east": (2.62, 2.20), "bed_north": (2.35, 2.95), "bed_door": (3.20, 2.75),
    "hall_door": (3.20, 3.95), "living_mid": (3.00, 4.25), "hall_east": (3.90, 3.95),
    "bath_door": (4.55, 3.30), "bath_in": (4.55, 1.85), "kitchen_hall": (5.95, 2.70),
    "kitchen_in": (5.95, 1.00),
}
NAV_EDGES = [
    ("suitcase", "bed_corner"), ("bed_corner", "bed_foot"), ("bed_corner", "bed_east"),
    ("bed_foot", "bed_east"), ("bed_east", "bed_north"), ("bed_east", "bed_door"),
    ("bed_door", "hall_door"), ("hall_door", "living_mid"), ("hall_door", "hall_east"),
    ("hall_east", "bath_door"), ("bath_door", "bath_in"), ("hall_east", "kitchen_hall"),
    ("kitchen_hall", "kitchen_in"), ("bath_door", "kitchen_hall"),   # along the hall, clear of furniture
]

# ---------------------------------------------------------------- items
# The family decides which skill grounds `pick up` and `put`.
FAMILIES = {
    "folded_pile": dict(label="folded garment pile", pick="two-hand side squeeze", place="lay flat",
                        hands=2, stackable=True, fragile=False),
    "rigid_box": dict(label="rigid box", pick="two-hand side squeeze", place="lay flat",
                      hands=2, stackable=True, fragile=False),
    "fragile_box": dict(label="fragile box", pick="two-hand side squeeze", place="lay flat, nothing on top",
                        hands=2, stackable=False, fragile=True),
    "bottle": dict(label="bottle", pick="one-hand power grip", place="stand upright",
                   hands=1, stackable=False, fragile=False),
}

# Scanned BlenderKit piles (royalty-free licence; fetched by fetch_assets.py and prepared
# by prepare_assets.py). raw = scan extents in metres; yaw_deg puts the short side on x.
# Piles are scaled so the short side (the squeeze axis) is at most 20 cm, which leaves
# a palm's width at the case walls and between the two garment columns, and the long
# side at most 28.5 cm, which leaves room for the toiletry bag beside a column.
GARMENT_SCANS = {
    "trousers": dict(asset="13029600-f765-4d9c-81c6-40a132c0b90b", name="Stack of blue pants",
                     author="antoine f", raw=(0.418, 0.476, 0.2106), scale=0.478),
    "tshirts": dict(asset="0b960a0d-01f7-4038-95bf-6915f58360f1", name="Stack of folded t-shirts",
                    author="Paweł Wałasiewicz", raw=(0.2392, 0.3951, 0.1571), scale=0.72),
    "sweater_pile": dict(asset="09e88778-bfa7-49a8-b53f-71d6273f3774", name="Folded garment stack",
                         author="antoine f", raw=(0.3432, 0.4187, 0.2412), scale=0.58),
    "striped_shirts": dict(asset="4fdb482d-2642-4388-ae57-8097afa99519", name="Folded Clothes Stack",
                           author="Paweł Wałasiewicz", raw=(0.4107, 0.298, 0.1939), scale=0.67, yaw_deg=90),
    "sweatshirts": dict(asset="f7449db4-8eaa-4298-849a-7c2dbff4e75d", name="Stack of folded sweatshirts",
                        author="Paweł Wałasiewicz", raw=(0.3021, 0.4628, 0.2045), scale=0.615),
    "towel": dict(asset="755a9a16-d738-44ab-9ab6-07d35db05e47", name="Folded Towel", author="Rex Hans",
                  raw=(0.2672, 0.2262, 0.0454), scale=0.88, yaw_deg=90, objects=("Cube",)),
    "towel_stack": dict(asset="2d9f13aa-15aa-48b1-8116-d891377fc162", name="Stack of Towels",
                        author="GL3D Gabriela Linhares", raw=(0.2265, 0.3475, 0.1436), scale=0.82),
    "gym_clothes": dict(asset="eab5b1cb-c75d-4e80-a679-c9a4ea26c3b7", name="Folded Clothes with Socks",
                        author="GL3D Gabriela Linhares", raw=(0.2629, 0.2887, 0.1678), scale=0.76),
}


def scan_size(key):
    g = GARMENT_SCANS[key]
    x, y, z = g["raw"]
    if g.get("yaw_deg", 0) % 180 == 90:
        x, y = y, x
    return tuple(round(v * g["scale"] + 0.001, 4) for v in (x, y, z))


@dataclass
class Item:
    name: str
    label: str
    kind: str              # pile | bag | bottle | asset (geometry of the physics body)
    family: str
    size: tuple            # collision box full extents (local x = squeeze axis, y, z)
    mass: float
    station: str           # home station
    xy: tuple              # centre on the home station
    yaw: float = None      # default: short side across the station normal
    heavy: bool = False
    asset: str = ""
    tint: tuple = (1, 1, 1)
    soft: bool = False
    mesh_offset: tuple = (0, 0, 0)   # asset origin in the item body frame
    mesh_objects: tuple = ()         # asset objects to render (empty = all)
    cloth: bool = False              # loose cloth (simulated as a flexible sheet): can be folded

    @property
    def room(self):
        return STATIONS[self.station].room

    @property
    def start(self):
        return (self.xy[0], self.xy[1], STATIONS[self.station].top + self.size[2] / 2)

    @property
    def start_yaw(self):
        if self.yaw is not None:
            return self.yaw
        n = STATIONS[self.station].normal
        return float(np.arctan2(n[1], n[0]) + np.pi / 2)


def _pile(name, label, station, xy, mass, heavy=False):
    return Item(name, label, "pile", "folded_pile", scan_size(name), mass, station, xy, heavy=heavy, soft=True)


ITEMS = [
    _pile("trousers", "folded trousers", "dresser", (3.52, 0.98), 1.2, heavy=True),
    _pile("tshirts", "T-shirt pile", "dresser", (3.52, 1.42), 1.0),
    _pile("sweater_pile", "sweater pile", "bed", (0.72, 1.20), 1.2),
    _pile("striped_shirts", "striped shirts", "bed_far", (0.95, 2.15), 1.0),
    _pile("sweatshirts", "sweatshirts", "nightstand", (0.45, 2.92), 1.2),
    _pile("towel", "folded towel", "towel_shelf", (5.39, 1.15), 0.4),
    _pile("towel_stack", "stack of towels", "vanity", (5.02, 0.30), 1.2),
    _pile("gym_clothes", "sweatpants and socks", "kitchen_counter", (6.62, 1.55), 0.8),
    Item("toiletry_bag", "toiletry bag", "bag", "rigid_box", (0.23, 0.13, 0.09), 0.65, "vanity", (4.55, 0.30),
         heavy=True, tint=(0.12, 0.25, 0.20)),
    Item("bottle", "water bottle", "bottle", "bottle", (0.072, 0.072, 0.23), 0.55, "kitchen_counter", (6.62, 0.75),
         yaw=0.0, tint=(0.36, 0.45, 0.38)),
    Item("notebook", "leather notebook", "asset", "rigid_box", (0.17, 0.20, 0.025), 0.35, "coffee_table", (1.88, 4.78),
         yaw=0.0, asset="binder_notebook", mesh_offset=(-0.342, 0, -0.0105), mesh_objects=("binder_notebook_closed",)),
    # lens towards the sofa: picked up that way, it is packed with the lens facing into the case
    Item("camera", "camera", "asset", "fragile_box", (0.148, 0.035, 0.078), 0.62, "coffee_table", (2.32, 4.80),
         yaw=np.pi, asset="Camera_01", mesh_offset=(0, -0.0015, -0.039), mesh_objects=("Camera_01",)),
]


# ---------------------------------------------------------------- pack regions
@dataclass
class Region:
    """Where things go in the open case (suitcase-base frame: x across, y along the
    length, +y = handle end). Items in a region stack on the case floor."""
    name: str
    label: str
    xy: tuple
    families: tuple
    max_len: float         # longest accepted extent along the case length (squeeze axis)
    max_across: float


# Inner base: x +-0.241, y +-0.337; inner height 0.333 with the lid closed. Two garment
# columns lie near the hinge side (1.5 cm clear of the arc the lid's hinge-side wall sweeps)
# with a palm-width gap between them and at the
# end walls; the strip on the person's side holds the bag (camera on top) and the bottle.
REGIONS = {r.name: r for r in [
    Region("wheel_end", "the wheel-end garment column", (-0.068, -0.155), ("folded_pile", "rigid_box"), 0.205, 0.30),
    Region("handle_end", "the handle-end garment column", (-0.068, 0.155), ("folded_pile", "rigid_box"), 0.205, 0.30),
    Region("front_strip", "the front strip", (0.170, 0.120), ("rigid_box", "fragile_box"), 0.24, 0.15),
    Region("bottle_corner", "the front wheel-end corner", (0.140, -0.205), ("bottle",), 0.08, 0.08),
]}


def all_static_boxes():
    boxes = list(WALLS) + [b for b in PROCEDURAL]
    for a in ASSETS:
        boxes += a.collision
    return boxes


def item(name):
    return next(i for i in ITEMS if i.name == name)


# ---------------------------------------------------------------- setting things down
# Anything carried can be set down on a furniture top (put_down), on one of a few fixed
# spots, tried in order. A spot takes an item only if it stays on the top, clear of the
# decor and the open case, with room for the hands beside it and beside everything lying
# there now. Where an item was is free once it is gone, so something set down there
# keeps it from being put back until it is moved.
SURFACES = ("dresser", "bed", "bed_far", "nightstand", "vanity", "towel_shelf", "kitchen_counter", "coffee_table")
SET_DOWN = {
    # chosen by scanning each top for where the most items fit; the last ones are home
    # spots, free once their item is gone. The person stands a standoff in front of a
    # spot, so none puts them against the west wall, the nightstand or onto the bed.
    "dresser": [(3.34, 1.16), (3.72, 1.20), (3.52, 0.98), (3.52, 1.42)],
    "bed": [(1.00, 1.15), (0.92, 1.25), (0.72, 1.20)],
    "bed_far": [(1.30, 2.31), (1.64, 2.31), (1.98, 2.31)],
    "nightstand": [(0.13, 2.93), (0.45, 2.92)],
    "vanity": [(4.20, 0.42), (4.22, 0.16), (4.55, 0.30), (5.02, 0.30)],
    "towel_shelf": [(5.26, 0.88), (5.26, 1.42), (5.39, 1.15)],
    "kitchen_counter": [(6.48, 1.11), (6.78, 1.11), (6.48, 0.47)],
    "coffee_table": [(1.91, 5.02), (1.59, 5.02), (2.23, 4.66), (1.88, 4.78), (2.32, 4.80)],
}
_TOP_BOX = dict(dresser="dresser_coll", bed="mattress", bed_far="mattress", nightstand="nightstand_coll",
                vanity="vanity_top", towel_shelf="towel_shelf", kitchen_counter="kitchen_top", coffee_table="coffee_table_coll")
# (x0, x1, y0, y1) kept clear: the alarm clock, the open binder, the open case and its lid
_CASE_OPEN = (1.17, 1.93, 0.98, 2.05)
KEEP_OUT = dict(nightstand=[(0.09, 0.25, 2.98, 3.14)], coffee_table=[(2.17, 2.59, 4.86, 5.14)],
                bed=[_CASE_OPEN], bed_far=[_CASE_OPEN])


@lru_cache(maxsize=None)
def surface_rect(station):
    b = next(b for b in all_static_boxes() if b.name == _TOP_BOX[station])
    return (b.center[0] - b.size[0] / 2, b.center[0] + b.size[0] / 2, b.center[1] - b.size[1] / 2, b.center[1] + b.size[1] / 2)


def set_down_yaw(name, station):
    """Set down the way it is picked up there: short side across the station normal."""
    if name == "bottle":
        return 0.0
    n = STATIONS[station].normal
    return float(np.arctan2(n[1], n[0]) + np.pi / 2)


GRIP = 0.12      # room each hand needs beside an item along its squeeze axis (a 9 cm gap fails in physics)


def footprint(name, xy, yaw, grip=0.0):
    """(x0, x1, y0, y1) of an item lying at xy turned by yaw (a multiple of 90 deg); with
    `grip`, widened by that much each side along the squeeze axis, where the hands go."""
    hx, hy = item(name).size[0] / 2 + grip, item(name).size[1] / 2
    if abs(np.sin(yaw)) > 0.7:
        hx, hy = hy, hx
    return (xy[0] - hx, xy[0] + hx, xy[1] - hy, xy[1] + hy)


def _overlap(a, b, gap=0.01):
    return a[0] < b[1] + gap and b[0] < a[1] + gap and a[2] < b[3] + gap and b[2] < a[3] + gap


def room_at(name, xy, yaw, station, lying=(), keep_out=True):
    """Whether `name` can lie at xy on `station` next to `lying` ((name, xy, yaw) of what
    lies there now): on the top, clear of the decor, and neither item's hands meeting the
    other (the hands squeeze from both sides, to set it down and to pick either up).
    keep_out=False checks only against `lying` (a home spot is on its top by design)."""
    x0, x1, y0, y1 = surface_rect(station)
    f, fg = footprint(name, xy, yaw), footprint(name, xy, yaw, GRIP)
    if keep_out and not (f[0] >= x0 - 0.02 and f[1] <= x1 + 0.02 and f[2] >= y0 - 0.02 and f[3] <= y1 + 0.02):
        return False
    if keep_out and any(_overlap(fg, t) for t in KEEP_OUT.get(station, [])):
        return False
    return not any(_overlap(fg, footprint(o, oxy, oyaw)) or _overlap(f, footprint(o, oxy, oyaw, GRIP))
                   for o, oxy, oyaw in lying)


def set_down_spot(name, station, lying=()):
    """Index of the first spot on `station` with room for `name` (None: no room)."""
    yaw = set_down_yaw(name, station)
    return next((k for k, xy in enumerate(SET_DOWN[station]) if room_at(name, xy, yaw, station, lying)), None)


# ---------------------------------------------------------------- suitcase hardware
# Positions in the suitcase-base frame of the closed case, measured on the scan (lock
# plate recessed in the lid's front wall above the seam, zip sliders on the seam, pull
# dock above them, lid strap along the middle of the lid front).
HARDWARE = dict(
    wall_px_visual=0.2385,            # scan's front-wall surface around the lock (physics wall is at W/2)
    lock_window=(-0.004, 0.136),      # y span where the lid's front wall is recessed to the scan surface
    dial_y=(0.0185, 0.031, 0.0435), dial_z=0.237, dial_r=0.0080, dial_w=0.0050, dial_x=0.2385,
    dial_detent=0.0025,               # N m detent torque amplitude (10 detents per turn)
    button=dict(y=0.074, z=0.237, half=(0.004, 0.015, 0.008), travel=0.003, stiffness=400.0),
    dock_y=0.115, dock_z=0.248,       # where the two pull tabs latch (slot under the lock plate)
    slider=dict(x=0.2594, z=0.139, half=(0.006, 0.012, 0.008), open_a=-0.300, open_b=0.296, gap=0.012,
                friction=1.5),        # N: zip resistance
    tab=dict(len=0.090, r=0.0045),        # pull tab: hangs from its slider, swings up flat against the side
    trolley=dict(levels=(0.0, 0.20, 0.38), travel=0.40, friction=3.0,
                 button=dict(half=(0.008, 0.003, 0.006), travel=0.002, stiffness=1500.0, proud=0.0035)),
    strap=dict(x=0.0, y_female=0.124, y_male=0.150, z=0.3906, width=0.050, travel=0.05),
)


# ---------------------------------------------------------------- loose cloth
# A flat beach towel on the kitchen counter, simulated as a flexible sheet (MuJoCo flex,
# passive in-plane and bending elasticity, no equality constraints). It is added to the
# model only for cases that use it (it multiplies the cost of the physics).
CLOTH = dict(name="beach_towel", label="beach towel", station="kitchen_counter",
             centre=(6.45, 1.15),           # near edge hangs 4 cm over the counter's front edge
             size=(0.32, 0.48), count=(5, 7), mass=0.25, thickness=0.003,
             young=3e4, poisson=0.2, radius=0.004)
