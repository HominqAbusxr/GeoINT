# Skylines: where a photo was taken from the line of its hills

SKILL.md › Step 6 — Pinpoint › Mountains / skyline. Where land meets the sky, the line is drawn by the terrain
alone, and an elevation model can draw it from any point on Earth. `terrain.py find` draws it from every point of
an area and ranks the places by how much of the photo's skyline **shape** each one explains; `terrain.py check`
asks the same of one place. The answer is a short list with a measure of how far to trust it, not a proof:
confirm it with the overlay, the named summits and a second clue.

| You have | Do | Section |
|---|---|---|
| Hills or mountains on the skyline, the region roughly known | trace the skyline, `find` over the region | 1–3 |
| A candidate place: a guess, a named viewpoint, a reverse-search hit | `check --at … --radius 500` | 4 |
| A place and a heading, to name what you see | `view --peaks --overlay` | 4 |
| A railway, road or power line and one mountain beside a flat horizon | `scan` → `fit` | `corridors.md` 4.3 |

Everything below is `uv run ${CLAUDE_SKILL_DIR}/scripts/terrain.py`, run in the photo's archive folder.

## 1. Trace the skyline

```bash
terrain.py ridge original/photo.jpg --step 8 --png ridge.png --out ridge.json   # one point every 8 columns
```

- **Open `ridge.png` before anything else.** The yellow dots must sit on the edge against the sky. Leave out
  what is not land: `--skip x0:x1,…` over trees, poles, buildings, a cloud resting on the ridge, a stretch
  traced on an inner edge; `--x0/--x1` where the ridge leaves the frame; `--ymin/--ymax` to keep clouds above and
  the foreground below out. **A gap is better than a wrong line**: a wrong stretch can sink the true place.
- `--by auto` (the default) reads a clear blue sky by blueness and an overcast or hazy one by brightness. Under a
  blue sky, snow and sunlit rock are brighter than the sky and brightness alone stops below them; force `--by
  blue` or `--by luma` when the check image shows the dots below the edge. Lower `--drop` (15–20) in haze.
- The lens matters as much as the line. `ridge` reads the EXIF 35 mm-equivalent focal length; without it, it
  assumes a 26 mm phone lens and `find` tries 1.5× either way. When the archived original has EXIF but the
  working copy lost it, pass `--f0` (pixels = f35 / 43.27 × the image diagonal in pixels); when the photo is
  plainly zoomed (compressed ranges, a 2× or 5× phone lens) or wide, give `find --hfov a:b`.
- **Panoramas and collages**: the search assumes one ordinary camera. Trace a single frame of a pan or a phone
  panorama, never the stitched strip, whose edges bend.
- **By hand** when the tracer cannot follow the line (night, haze, a collage, a busy edge): write
  `{"image_size": [W, H], "ridge": [[x, y], …]}` with points along the skyline every 5–10 px, carefully on the
  peaks and notches, which carry most of the information. Trace the skyline only, not ridges in front of it.

## 2. Search an area

```bash
terrain.py find --ridge ridge.json --bbox S,W,N,E --out find.json --sheet top.jpg --heat heat.png \
  [--dist 2:60] [--facing NE,E | --heading 200:260] [--hfov 55:70] [--stands summit,slope] [--quick]
```

- **The area**: as small as the other clues allow (the country funnel, a valley, a massif). About 250 km² takes
  under a minute on 8 cores for a normal lens, at a point every ~200–400 m refined down to 30 m round the best
  places; a long lens gets a denser grid by itself (it reads the parallax between near and far ridges in more
  pixels, so its best place is a narrower peak), and so does a `--dist` band that starts near. Terrain tiles are
  downloaded once into `.geo-cache/dem`. For a large region, `--quick` first, then a finer search round the region
  it finds. `--at lat,lon --radius m` searches a circle.
- **What you know narrows the search and the chance fits. A wrong hint can rule the true place out, so leave out
  what you are unsure of, or give it wide.**

| Hint | Give it when | Effect |
|---|---|---|
| `--dist a:b` km, where the skyline stands | you can judge it: haze layers, the size of trees and buildings on the ridge, snow on far summits | the strongest hint in tests: it rules out hillsides near a wrong camera that mimic a far skyline, and it grids the area finely when the band starts near |
| `--facing N,NE` or `--heading a:b` | shadows or the sun give the direction (`sun.py`); tick the neighbours when unsure | each direction removes 7/8 of the headings a wrong place can use |
| `--stands summit,slope,valley,plain` | the camera clearly stands on a summit or ridge, or down in a valley | never searched on water unless `water` is ticked (a boat) |
| `--eye m`, `--floors n`, `--mode drone` | taken from a building (3 m a storey; on a slope give fewer) or a drone | |
| `--hfov a:b` | the zoom is clear from the picture, or EXIF is missing | a guessed lens is the common reason the best place sits at the edge of the lens range |
| `--clear` | crisp air: nothing hidden behind the traced line | without it the skyline is also read cut at 5, 10, 20 and 50 km, since haze hides far ranges |
| `--key x0:x1` | a peak or notch you would recognise anywhere | without it, the trace's sharpest stretches are held anyway |
| `--selfie` | a front-camera photo that may be mirrored | searches twice |

## 3. Read the result

- **`explained`** is the share of the trace's shape the terrain there accounts for: 100% when every peak and
  notch lies on the terrain, 0% when it does no better than a straight line (which any hillside does, given a
  little roll). It is a score, not a probability. The whole trace counts and so does each quarter of it across
  the frame: a place that fits the easy stretches and misses the peaks is held to its weakest part.
- **`trust`** (solid / check / weak) and its `reasons` say how far the best place stands out: another place
  3 km or more away explaining nearly as much; barely above what the area explains by chance (`chance`, its
  top tenth); a lone spike (ground one could stand on nearby explains far less: within 150 m, or 5% of the
  distance to the matched ridges, at the same height and with the same view open, since a few steps back from a
  viewpoint's rail the rock in front hides the valley); a plain trace (fewer than 5 turns fits many places); the
  lens at the edge of its range; the roll or the tilt at its limit; a decoy fitting nearly as well. `verdict`: match, loose (under half), ambiguous (a separate place as good), none,
  flat (the trace has no shape to place).
- **Decoys**: each of the best places is searched again with the trace mirrored and turned upside down. A real
  place explains its own trace far better than its decoys (in tests: 74% against 23%); where a decoy does nearly
  as well, the terrain there fits any shape.
- **Look at the first five overlays** (`--sheet`), not only the first: red is the place's skyline through the
  fitted camera, green the trace points on it, orange the points off it. A look-alike misses somewhere, often
  at the foot of a mountain or at the frame's edge.
- **When a place far from what you expected wins**, ask what would rule it out (a direction, a distance, a
  landmark, the sun) and search again with that hint, rather than trusting either answer.
- **The map** (`--heat`) shows sight lines as bands: far ridges change little as the camera moves along the line
  of sight, so a far skyline fixes a line better than a point, and the best place can be a few hundred metres
  off along it (`geometry.md` 7.4). Ridges 1–5 km away pin the point more tightly.

## 4. Confirm, and check one place

```bash
terrain.py view --at LAT,LON --heading H --hfov F --pitch T --roll R --range 20000 --zoom 13 --peaks \
  --photo original/photo.jpg --overlay --out view.png       # find prints this command for its best place
terrain.py check --ridge ridge.json --at LAT,LON --radius 500 --out check.json --overlay check.jpg
```

- `view --peaks` labels the named summits (OpenStreetMap) the camera would see, grey when the terrain in front
  hides them: name the peaks in the photo and check them against the labels.
- `check` scores one place over every heading and lens, lists the headings that fit, runs the decoys, and with
  `--radius` tries the ground round it: a candidate's pin is often a few hundred metres off.
- Then a second, independent clue at the best place: imagery (`tiles.py`), Street View or photo spheres
  (`pano.py`), a road, a lake shore, a building, the sun (`sky.md`). Report the best place as a sight line or an
  area unless that second clue pins it.
- **A geotag can be checked the same way.** EXIF GPS is evidence, not truth: a phone deep in a valley can keep a
  fix from kilometres back. `check --at <the geotag>` tells: in tests, a photo whose geotag sat in one village
  fitted no heading there (18%), while 1.8 km up the valley the whole skyline lay on the terrain, its named
  summits in place. When they disagree, say so and give both.

## 5. Where it fails

- **Near ridges only, a few hundred metres away.** The terrain is known to ~19–30 m a cell (finer where a
  national lidar survey is published): precise for a ridge 10 km away, coarse for a hillside 500 m away. Unless
  `--dist` says so, ground nearer than 250 m (`--floor`) cannot be the skyline: from a slope the terrain's grain
  draws the eye's own hillside as a ridge any trace can be laid on.
- **Plain shapes**: low far hills, a flat plain, the sea's horizon. Many places look the same (`flat`, few turns).
- **Haze and night**: the visible line may be a nearer ridge than the top of the land; trace only what is sharp,
  leave `--clear` off.
- **Trees and buildings on the ridge**: the model holds part of them; leave those stretches out of the trace.
- **Wide-angle edges**: a kit zoom at its widest bends straight lines near the frame's edges by several pixels;
  the model is a perfect pinhole, so the edges of such a trace fit worse everywhere (the true place included).
- **Snowfields and glaciers** change from year to year; a snowy skyline fits the terrain model less closely.
- **A busy foreground** hiding most of the skyline, and **large areas**, which meet more look-alikes.
- **The trace decides.** The same photo traced differently can go from found to missed; the search answers the
  trace and the hints, not the photo.

## 6. Measured

Geotagged Wikimedia Commons photos, each searched over a box of about 200–250 km² with the camera off-centre,
with no hints and with the distance band one would judge from the photo. Distances are to the photo's geotag,
which is itself only a few tens of metres good at best.

| Photo | Lens | Best place, no hints | Best place, with `--dist` | Explains (decoys), with `--dist` | Trust |
|---|---|---|---|---|---|
| Half Dome from Glacier Point, overcast | 67° (EXIF) | 565 m | 127 m | 74% (23%) | solid |
| Glacier Point, a wider view, clear sky | guessed | 171 m | 605 m | 76% (46%) | solid |
| Matterhorn from Gornergrat | 20° (EXIF) | 86 m | 86 m | 72% (21%) | solid |
| Ober Gabelhorn group from Gornergrat | 20° (EXIF) | 636 m | 636 m | 69% (35%) | solid |
| Monte Rosa snowfields from Gornergrat | 64° (EXIF) | 198 m | 151 m | 49% (24%) | weak (loose) |
| High Atlas valley floor, ridges 0.5–3 km away | 63° (phone EXIF) | 280 m | 282 m | 88% (45%) | check |
| High Atlas village | 67° (phone EXIF) | 1.8 km | 1.8 km | 59% (34%) | solid |

Wherever the geotag held, the first place listed was the camera's, 90–640 m from it, without hints as with them:
far ridges fix a sight line and the best place sat a few hundred metres along it. The snowfields' skyline fitted the
terrain model only by half, at the camera too, and was rated weak accordingly. The village photo is the geotag case
of §4: no heading fits at its geotag, while 1.8 km up the valley its whole skyline lies on the terrain.

## How it works

- **Terrain**: Mapterhorn tiles (Copernicus GLO-30 worldwide, national lidar where published; the Matterhorn's
  summit comes out at 4,468 m against 4,478 m surveyed), read at 19 m a cell out to 11 km and at coarser levels
  farther out, with Earth curvature and refraction (k = 0.13). AWS Terrain Tiles are the fallback (`--dem aws`).
- **The skyline round each point**: rays every 0.1–0.5° cast out to 50 km, the running maximum of the elevation
  angle; the cut at each reach and the distance of the ground that sets each column are kept.
- **The fit**: for each heading and lens, the camera's tilt is the median of what the trace's points ask (a tilt
  is not a shift: off-centre points rise less and turn sideways), then roll and tilt are laid together on the
  points already near the terrain, so a stretch traced wrong cannot steer them. Misses are measured in pixels of
  a 1,200-px-wide frame with a loss that stops growing past 12 px, and set against the trace's own best straight
  line. The best fits are polished exactly over heading, tilt, roll and lens.
- **The search**: a hex grid, cells filled in finer where the matched ridges stand near, refined three by three
  round the best points down to 30 m, the best places scored in full on a patch round them, then decoys and the
  ground round the best checked.
