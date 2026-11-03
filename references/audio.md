# Sound: where and when a recording was made

SKILL.md › Special situations › Video or audio. A video's frames can be useless (a dark room, a wall, a security
camera facing water) while its soundtrack still holds the place. Aircraft passing overhead, thunder and the mains
hum leave traces that are often inaudible but plain on a spectrogram, and each can be matched against open
records. The method treats the soundtrack as a second photo: look at it, list what it contains, turn each clue
into a constraint, and combine them with whatever the pixels give.

| What you hear (or see on the spectrogram) | Matched against | What it gives | Typical precision |
|---|---|---|---|
| A propeller aircraft or helicopter passing: a falling tone, often 60–300 Hz, with harmonics | ADS-B tracks of that day | the microphone's position, and the recording's start time | 100–300 m from two passes a known time apart; one pass with a known clock: the right flight and a short segment |
| Thunder: low (20–250 Hz) bursts | lightning-strike records | city or neighbourhood, start time | ~2–5 km with free data; sub-km with precise networks |
| Mains hum: 50 or 60 Hz and harmonics | the world's grid frequencies; grid-frequency records | the grid region; with a record, the exact time | country group; seconds |
| Voices, announcements, the call to prayer, bells, sirens, crossing signals, trains, birds, traffic | your knowledge, web search | language, country, setting | country to street |

Everything below uses `uv run ${CLAUDE_SKILL_DIR}/scripts/audio.py`. Archive the recording like a photo
(`photo_session.py start --image clip.mp4`) and keep every output in that folder.

## 1. Look at the sound first

```bash
audio.py extract original/clip.mp4 --out audio.wav           # mono WAV (any ffmpeg-readable file works directly too)
audio.py look audio.wav --band 40,600 --out spec.png          # rows of 5 minutes, whitened: steady noise removed
audio.py events audio.wav --png events.png                    # candidate passes, steady tones, hum, low-frequency bursts
```

Open the sheets and read them yourself; `events` is a first pass and misses faint or broken curves.

- **A passing aircraft** draws a curve that falls from high to low: flat while it approaches, steepest at the
  closest approach, flat again as it leaves. Harmonics stack above it at 2×, 3×… A long, gentle fall is a far or
  slow pass; a short, steep one is close. Jets are broadband roar with high fan tones; propellers and rotors give
  clean low tones, which is why they carry the method.
- **Flat lines** are machines, fans, transformers or the mains hum (`hum`).
- **Vertical bursts** low in the band are thunder, doors, wind gusts or trucks; thunder rolls for seconds.
- **Inaudible is not absent.** A pass 20 dB under the noise can be clear on a 1-second, 1-Hz spectrogram. Zoom
  (`look --start --end --band`) and lower `--contrast` before calling a stretch empty.
- **Edits break timing.** A cut changes the background abruptly; only use time offsets inside one uncut stretch.
  Re-encoding at another speed scales every frequency and time together (a known tone, such as the mains hum,
  reveals it). High-pass filters, MP3 compression and phone calls usually leave these tones recoverable.

## 2. Aircraft: from a curve to a spot

### 2.1 The physics in brief

For a source at constant speed on a straight line, two numbers shape the curve. The **speed** `v` sets the size of
the fall: `f_high / f_low = (c + v) / (c − v)` with `c` ≈ 340 m/s. The **closest-approach slant distance** `d` sets
how fast it falls: at the closest approach the slope is `f0 · v² / (c · d)`. The emitted frequency `f0` is heard at
the moment the closest approach is heard (`heard CPA = emitted CPA + d / c`).

- **Model the emission time.** What the microphone hears at time `t` left the aircraft at `t − r/c`, where `r` is
  the distance at that earlier moment. Computing the radial speed at `t` and then delaying the curve gives a
  different, wrong curve, and the search fails. `audio.py` solves the emission time exactly.
- **Slant distance needs the ground.** ADS-B altitudes are above sea level (barometric or GNSS); the microphone
  stands on terrain. `locate` takes ground elevation from a DEM (`--flat` only for tests or open sea).
- **Doppler speed is air speed.** Wind of 5–20 m/s separates it from the ADS-B ground speed, so the search ranges
  are padded (`fit --speed-pad`, default 10 m/s or 10 %).
- **Imperfect curves are information.** Turns, climbs and speed changes bend the curve away from the ideal shape.
  The fit only needs ranges; the final scoring uses each candidate's real track, so the true flight's bends match
  and a lookalike's do not.

### 2.2 Fit each pass

```bash
audio.py fit audio.wav --start 350 --end 500 --band 90,240 --name e1      # → fit_e1.json + fit_e1.png
audio.py fit audio.wav --trace e2_points.csv --name e2                    # very faint: hand-traced t,f points
```

- Give the whole pass (approach to departure) and a band around **one** harmonic; any harmonic gives the same speed
  and distance, so take the cleanest. Check `fit_e1.png`: the cyan trace must follow the curve and the white fit
  must sit on it.
- `best` is the fit; `region_99` the speeds and distances the curve allows; `search` the padded ranges the search
  uses. A warning that the trace touches the edge of the window or the band means part of the pass was cut off:
  widen them and refit, because one side of a pass alone gives confidently wrong numbers.
- **Manoeuvres bias the fit.** Near airports aircraft climb, turn, level off and accelerate through the pass; a
  constant-speed straight line then fits with a large residual (above ~5‰, against 2–5‰ for a clean pass) and
  wrong speed and distance. `fit` then widens its search ranges automatically (`loose` in the JSON; `--loose` forces
  it: speed ×0.5–1.6, distance ×0.33–3, closest approach ±8 s), so the funnel keeps the true flight and the exact-curve scoring, which uses the real track, decides.
- **Jets.** Where only airliners fly low (approach and departure paths; much of the world outside North America and
  Europe has little light-aircraft traffic in ADS-B), a jet's fan whine is a tone too, about 1–3 kHz on approach,
  and it falls through a Doppler curve like a propeller's. It fades faster with distance (high frequencies are
  absorbed), so it works for passes within a few km: decode at a higher rate (`--rate 16000`), fit a band around
  the tone, and search with `locate --types all --silence-types all`.
- For a curve too faint to trace automatically, zoom a `look` sheet, read off 15–40 points along the curve (time in
  seconds, frequency in Hz) into a CSV, and fit with `--trace`; then score with `locate --score trace`.

### 2.3 Flight tracks

```bash
audio.py adsb --date 2025-09-08 --bbox S,W,N,E --out tracks.jsonl                 # whole UTC day
audio.py adsb --date 2025-09-08 --from 18:00 --to 23:59 --bbox S,W,N,E --out tracks.jsonl
```

- The default source is the open **adsb.lol** archive (ODbL): one release per UTC day of every aircraft its
  volunteer receivers heard, positions every few seconds, with the ICAO type, so the engine class is known
  (P piston, T turboprop or turboshaft, J jet; helicopters are type H…). A day is 3–4 GB; it is streamed and
  filtered to the box, not stored. A recording near midnight UTC needs both days (concatenate the two files).
- **Check coverage before searching.** The output counts segments by engine type. Volunteer coverage is dense in
  North America and Europe and thinner elsewhere; low-flying aircraft drop out first (line of sight to the
  receivers). Many light aircraft outside the US carry no ADS-B transmitter at all, while airline turboprops
  nearly always do. An aircraft missing from the data cannot be found: if the true one is absent, the top
  candidate is a coincidence, and a weak margin is the warning. On one sample day (November 2026) the archive held,
  below 3,000 m: around New York ~360 piston aircraft, ~60 turboprops and ~65 helicopters; around Paris and over
  the UAE a few dozen of each; over northern and central Morocco 8 airline turboprops, 1 helicopter and ~40 jets
  (no piston aircraft); over Cairo, Riyadh, Algiers and Tunis almost nothing. Where turboprops do fly (Morocco's
  regional routes), recordings simulated on that day's real tracks were searched blind across the whole country
  and day: 5 of 8 were found at 18–340 m with the right aircraft and the start time to a few seconds; the 3 misses
  involved landing, decelerating or circling aircraft by an airport, and their winners led by less than 0.15
  (clear wins led by 0.5–2). Where the archive is empty, look for a commercial source or rely on the other sounds.
- Tracks are split wherever positions stop for more than 90 s (an aircraft on the ground, out of coverage);
  never interpolate across such a gap, or a parked aircraft becomes a slow phantom pass.
- Other sources become a CSV (`id,time,lat,lon,alt_m` or `alt_ft`, optional `gs_ms,type`) for `adsb --csv`:
  OpenSky Network's historical database (research access), ADS-B Exchange (paid API; free full-day samples for the
  first day of each month), Flightradar24 or FlightAware track exports (subscription).

### 2.4 The search: `locate`

```bash
audio.py locate audio.wav --tracks tracks.jsonl --fit fit_e1.json --fit fit_e2.json \
                --date 2025-09-08 --bbox S,W,N,E --out-dir locate
audio.py locate audio.wav --tracks tracks.jsonl --fit fit_e1.json --audio-start 2025-09-08T20:37:00Z --clock-tol 30
```

It narrows the whole box to a few cells, then judges those cells on the full curves:

1. **Distance and speed.** For each pass, every non-jet flight (`--types`) whose 3-D speed is in the fitted range
   marks the cells it passed abeam of at the fitted slant distance (grid `--grid`, 200 m).
2. **Time offsets.** Each later pass must be heard at the same offset from the first as in the recording (± 5 s
   plus the fit's uncertainty). With a known start time (`--audio-start`, `--clock-tol`), the first pass must
   also fall at the right absolute time.
3. **Silence.** A cell is dropped if another non-jet aircraft would have passed it within the farthest event's
   distance between the passes (`--silence between`; `whole` uses the whole recording; `--audible` sets the
   distance). Silence removes the airports, where heavy traffic makes coincidences likely. Near an airport, the
   many passes are themselves the signal: fit three or four.
4. **Exact curves.** For each surviving cell, the curve each candidate flight would have produced there (emission
   time, the real interpolated track, the DEM) is laid on the spectrogram and scored by the brightness under it.
   The first pass may slide ± 3 s and ± 1 % in frequency to absorb measurement error; every later pass is
   **locked** to it by the flight geometry, and that lock is what separates the true pair from lookalikes.
5. **Refine.** The best cell of each flight combination is refined on 50 m and 12.5 m grids.

Read `locate.json`: `best` (the microphone's estimated position, the flights with registration and type, their
heard closest approaches, the implied recording start time), `margin_to_second`, the `funnel` counts, and
`curves.png` (the modelled curves on the spectrogram for the top three combinations) and `map.png`.

- **Trust** a winner that leads clearly (a margin of 0.5 or more in tests; below ~0.2 treat the top few as a
  short list), matches every pass in `curves.png`, and keeps first place when the search is rerun with wider
  ranges (`--offset-tol 8`, `--types all`, `fit --loose`) or a finer grid (`--grid 100`).
- **On a real recording** (one hour by a harbour, with boats, birds, wind and distant traffic), the two cleanest
  passes, picked by eye from the sheets, located the microphone 234 m from the true camera among ~2,300 non-jet
  flight segments over a 130 × 200 km area and seven hours, with a clear lead (0.58), the two aircraft named and the
  start time to seconds. A single faint pass of an airliner with that clock gave a short list holding the true
  spot second (`--types all`): one pass narrows, two passes decide.
- **Hardest near airports.** Landing, decelerating and circling aircraft bend their curves the most; a slow,
  slowing approach can even fit as a faster pass far away with a small residual. Prefer passes of aircraft in
  steady flight, add a third pass when you have one, and check the winner's curves by eye.
- **One pass** with a clock known to ± Δt: the position along the track trades exactly against the clock, so the
  uncertainty is about speed × Δt (60 m/s × 5 s = 300 m). One pass with no clock gives only a list.
- **Precision comes from constraints, not from the grid.** The area searched × the time window × the traffic
  decide how many coincidences survive. A town and a day is easy; a whole country needs two passes and silence.
  Narrow the box first with everything else (language, sounds, the scene).
- **Start time.** `recording_start` dates the recording to a few seconds: chronolocation for free, and a check
  against any clock in the video.

### 2.5 Verify

- Fit another harmonic, or another pair of passes, and rerun: the same spot should win.
- Go to the spot (satellite, street level): the point is the **microphone**, ± the final grid step. Find the
  camera itself with the scene's geometry (a pole, a window, a balcony) as for any photo (`geometry.md`).
- In `report.md`, state that the position is audio-derived, list the flights (registration, type), heard
  closest-approach times, slant distances, scores and the margin, and say which constraints were used.

## 3. Thunder

```bash
audio.py lightning --date 2025-07-02 --from 17:00 --to 20:00 --bbox S,W,N,E --out strikes.csv   # GOES GLM, the Americas
audio.py thunder audio.wav --strikes strikes.csv --bbox S,W,N,E --date 2025-07-02 --from 17:00 --to 19:30
```

- **Method.** Each strike's time and place give, for every grid point, when its thunder would arrive (slant
  distance from a source about 2 km up, at 343 m/s) and how loud it would be (falling with distance). The
  recording's 20–250 Hz envelope is compared with each point's modelled envelope at every possible start time
  (cosine similarity, computed with FFTs over the whole window). The best point and start time win; rerun on a
  smaller box with a finer `--grid` to sharpen it.
- **Data.** GOES GLM (`lightning`): the Americas and nearby oceans, free, total lightning with a position error of
  several km. Meteosat Third Generation's Lightning Imager covers Europe and Africa (EUMETSAT Data Store, free
  account). National networks are far more precise (some Nordic and European services publish strikes with errors
  of a few hundred metres); the Blitzortung community network shares its archive with contributing stations;
  commercial networks reach sub-km. Any of them becomes a CSV with `time,lat,lon`.
- **Settings.** Position errors become timing errors (1.5 km ≈ 4 s): `--smooth` ≈ error / 343 m/s. The default
  `--power 0.25` compresses loudness so that the cadence of every burst counts, not only the loudest one. When most
  strikes come from one cluster, positions along a line score alike; `--balance` weights isolated strikes up.
- **Limits.** Wind noise sits in the same band (sheltered microphones work best); free data needs roughly ten or
  more bursts over half an hour; a single storm cell overhead constrains less than strikes on several sides.
- **Then listen.** Thunder gives the area; the rest of the sound often gives the spot: cars passing fast in both
  directions → a straight two-lane road; rain drumming on metal → a vehicle or a light roof; then the scene.

## 4. Mains hum

```bash
audio.py hum audio.wav --csv enf.csv                         # 50 or 60 Hz, harmonics, ENF trace
audio.py hum audio.wav --reference grid_frequency.csv         # when was it recorded? (time,frequency record)
```

- 50 Hz: Europe, Africa, most of Asia, Australia, most of South America. 60 Hz: North America, Brazil, Colombia,
  Venezuela, Peru, Saudi Arabia, South Korea, Taiwan, the Philippines, western Japan (eastern Japan is 50 Hz).
- The hum identifies a **synchronous grid**, not a country (continental Europe is one grid from Portugal to
  Turkey). Its frequency wanders by tens of millihertz in a pattern shared across that grid, so the ENF trace
  matched against a grid-frequency record (many transmission operators publish second-by-second frequency data)
  dates a recording to the second and confirms the grid.
- A recorder on battery away from wiring may show none; a hum on a portable recording means mains power nearby.

## 5. Listening clues (no tools needed)

Listen with headphones, then check every convention on the web before relying on it: they differ by city and
change over time.

- **Speech:** language, dialect, code-switching, place names, prices, phone numbers, announcements (transit,
  airports, shops). Transcribe and treat it as text (`textgeo.py --text "…"`).
- **Religious and civic sounds:** the call to prayer (its time ties to sunrise, noon or sunset at that longitude:
  `sun.py`), church bells and their hour patterns, school bells, sirens (two-tone hi-lo, wail, yelp patterns
  differ by country), pedestrian-crossing signals (ticks, cuckoo calls, melodies), garbage-truck or ice-cream
  tunes, train horns (multi-chime North American horns vs two-tone European ones), tram bells.
- **Nature:** bird species (with a bird-sound identifier such as BirdNET or Merlin) bound the region and the
  season; insects (cicadas) and frogs too.
- **Traffic:** driving side from passes in stereo, speed and road surface, horn use, motorbike density.

## 6. Pitfalls

- **Clocks.** File and container times can be the start or the end of a recording, in local time or UTC; prefer
  `--clock-tol` generous enough for that, and let `recording_start` correct it.
- **Speed and pitch changes.** A re-timed video scales frequencies and durations; the Doppler ratio survives, the
  distance does not. Check against the mains hum (exactly 50 or 60 Hz on average) before fitting.
- **Sample-rate drift** in cheap recorders (~0.01 %) is negligible for passes, noticeable only over hours.
- **Gaps in the flight data** are not silence: a cell is never dropped for an aircraft the data does not hold, so
  incomplete coverage weakens the funnel but cannot by itself exclude the truth. A missing event aircraft can.
- **Coverage first.** Before a long search, confirm the box and day hold low non-jet traffic in the tracks; if the
  area has none, the aircraft method cannot work there and the other sounds have to carry the case.
