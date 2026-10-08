# 3D asset sources

## CC0 environment and props

Poly Haven models are CC0. The project includes their 1K glTF versions and referenced JPEG textures. Suitcase maps are reused on the open suitcase built for the interaction target.

- [Vintage Suitcase](https://polyhaven.com/a/vintage_suitcase) — suitcase exterior maps.
- [Drawer Cabinet](https://polyhaven.com/a/drawer_cabinet) — dresser model.
- [Modern Wooden Cabinet](https://polyhaven.com/a/modern_wooden_cabinet) — cupboard model.
- [Camera 01](https://polyhaven.com/a/Camera_01) — camera item.
- [Book Encyclopedia Set 01](https://polyhaven.com/a/book_encyclopedia_set_01) — room dressing.
- [Terlenka](https://polyhaven.com/a/terlenka) — 1K CC0 woven-fabric diffuse, normal, and packed AO/roughness/metal maps used by the clothing fallback.

Poly Haven asset library and license information: <https://polyhaven.com/license>.

## CC0 bedroom furniture

The bed, side table, lamp, wool rug, and curtains are from 3DAssets.dev's [Bedroom and Living Room Furniture](https://3dassets.dev/packs/bedroom-and-living-room-furniture) pack. The pack is published under CC0 1.0 Universal and the GLBs are bundled locally under `assets/3dassets-bedroom/`. The wardrobe and dresser use the locally bundled Poly Haven models listed above.

## CC0 clothing and footwear

The following small GLB models are from 3DAssets.dev's [Clothing Rail and Wardrobe](https://3dassets.dev/packs/clothing-rail-and-wardrobe), [Skate Park and Street Sports](https://3dassets.dev/packs/skate-park-and-street-sports), and [Hotel and Resort Operations](https://3dassets.dev/packs/hotel-and-resort-operations) packs. All three packs are CC0 1.0 Universal. The GLBs are bundled under `assets/3dassets/` so the room does not need those asset sites at runtime.

- [Folded t shirt](https://3dassets.dev/assets/clothing-rail-and-wardrobe-tshirt-folded-a3a7c296) — linen-shirt item.
- [Folded jeans](https://3dassets.dev/assets/clothing-rail-and-wardrobe-jeans-folded-b1b8698f) — jeans item.
- [Folded jumper](https://3dassets.dev/assets/clothing-rail-and-wardrobe-jumper-folded-8bbd99f8) — sweater item.
- [Rolled pair of socks](https://3dassets.dev/assets/clothing-rail-and-wardrobe-socks-pair-rolled-19ba2865) — socks item.
- [Pair of skate shoes](https://3dassets.dev/assets/skate-park-and-street-sports-skate-shoes-84eabcd7) — weekend-shoes item.
- [Bathroom amenity tray](https://3dassets.dev/assets/hotel-and-resort-operations-amenity-tray-ab925636) — small toiletry tray item.

## Hand model reference

- **Rigged Hand** by **J-Toastie**, sourced from [Poly Pizza](https://poly.pizza/m/BEy8jbxm6A), Creative Commons Attribution (CC BY). The bundled GLB and preview are retained at `assets/rigged_hand/` as reference material.
- The bundled rigged GLB and preview are retained at `assets/rigged_hand/`. The runtime loads two skinned copies, applies the interaction pose to the hand root, and drives finger curl with the bundled animation clip. If the model fails to load, locally generated articulated hands remain as a fallback. Hands are visual geometry, not a contact or grasp physics model.
