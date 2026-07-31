# GPS receiver and logger research

Status: **research recommendation**
Date: **2026-07-25**
Scope: GoPro HERO10 Black telemetry, an external GNSS logger for Mapper, and a
small personal “GPS tamagotchi” derivative.

Prices and availability below were checked on 2026-07-25. They exclude tax and
shipping and will change. Accuracy figures are vendor specifications under
their stated test conditions, not expected accuracy under tree cover or in a
canyon.

## Recommendation

1. **Do a no-purchase control first.** The Kings Canyon files show that the
   camera did not merely have noisy 3D positions: it never reported a 3D fix.
   Repeat a short known route after acquiring a good fix in open sky.
   [GoPro Labs](https://gopro.github.io/labs/control/actions/) has a documented
   HERO10/11 `!Dx` action that waits for GPS lock below a selected DOP and then
   synchronizes time; `!D5` is the relevant starting threshold. Keep the
   top/record-button side of the camera unobstructed and avoid a metal frame.
2. **For the first external purchase, use a Columbus P-10 Pro.** It is the
   cleanest test of the hypothesis we care about: a self-contained,
   weather-resistant, dual-band L1/L5 logger with its own antenna, storage,
   battery, UTC, fix and DOP data. It is $239, 82 g, logs at 1 or 5 Hz, and is
   rated for 50 hours at 1 Hz. It needs only a microSD card. Five hertz is ample
   for a person walking; better reception and honest quality metadata matter
   more than the HERO10's approximately 18 Hz sample cadence.
3. **Do not build a ZED-F9P/RTK unit first.** RTK can supply centimetre-level
   positions, but only with a suitable multiband antenna and a correction
   source (NTRIP, local base, or a paid correction service). The complete
   SparkFun RTK Facet is $739.95, 583 g, and about 25 hours. It is useful as a
   borrowed/rented ground-truth instrument, not the first everyday logger.
4. **For a module-based prototype, use OpenLog Artemis plus either
   SAM-M10Q or DAN-F10N.** The SAM-M10Q path is solderless, single-band, and can
   log GNSS plus its onboard 9-axis IMU with essentially no logger firmware.
   The DAN-F10N adds dual-band reception and an integrated antenna, but uses
   UART and benefits from custom firmware for PPS-aligned GNSS/IMU logging.
5. **Build the tamagotchi on the same OpenLog Artemis stack.** Add a 1.3-inch
   128×64 Qwiic OLED, two buttons, a protected 1,500 mAh LiPo, and an enclosure.
   A breakout-board prototype is roughly $180–205 and 60–85 g. Expect roughly
   14–18 hours with the OLED continuously lit, or 24–36 hours with a
   button-wake display. These runtime and finished-weight figures are
   engineering estimates and must be measured.

The P-10 Pro and the tamagotchi answer different questions. The P-10 Pro is the
fastest way to determine whether better GNSS changes Mapper alignment. The
tamagotchi is a pleasant product project whose first revision can share the
same ingestion contract, but it should not delay that experiment.

## What is inside the HERO10?

GoPro does not publish a receiver part number. A detailed third-party
[HERO10 teardown](https://gethypoxic.com/blogs/technical/gopro-hero10-teardown)
identifies two production variants:

| HERO10 production hardware | Receiver | Receiver-level specification |
| --- | --- | --- |
| Hardware A | u-blox `UBX-M8030-CT` chip | L1-only GPS/QZSS, Galileo, GLONASS and BeiDou; up to three concurrent constellations; up to 18 Hz for one GNSS or 10 Hz for two; 2.0 m CEP; 21 mA at 3.0 V continuous for two GNSS (about 63 mW), or 5.3 mA in 1 Hz power-save mode |
| Hardware B | u-blox `MAX-M10S` module, reportedly substituted during parts shortages | L1-only GPS/QZSS, Galileo, GLONASS and BeiDou; up to four concurrent constellations; up to 10 Hz; 1.5 m CEP; approximately 24–32 mW at 3.0 V depending on enabled constellations |

The Hardware A values come from the official
[UBX-M8030 product summary](https://content.u-blox.com/sites/default/files/products/documents/UBX-M8030_ProductSummary_%28UBX-15029937%29.pdf).
The Hardware B values come from the official
[MAX-M10S data sheet](https://content.u-blox.com/sites/default/files/MAX-M10S_DataSheet_UBX-20035208.pdf)
and [MAX-M10 product summary](https://cdn.sparkfun.com/assets/9/6/d/6/5/MAX-M10_ProductSummary_UBX-20017987.pdf).

Those are receiver-component figures, not camera figures. They do not include
the antenna, RF losses, processor, storage or the rest of the camera. They also
do not guarantee that GoPro enables every constellation or the maximum
navigation rate.

Both parts require an antenna system:

- `UBX-M8030-CT` is a bare chip. The product design must supply its oscillator,
  RF matching/filtering, antenna and other support parts.
- `MAX-M10S` integrates a TCXO, LNA and SAW filter but still does **not**
  integrate an antenna. It supports active or passive antennas.
- The HERO9/10 GPS daughterboard has a ceramic patch antenna. Teardown/reuse
  documentation shows that the ceramic face must point toward the sky. The
  small enclosure, the wearer’s body, terrain, foliage and nearby electronics
  can dominate receiver-chip specifications.

The same teardown identifies the HERO10 IMU as a Bosch `BMI260`. GoPro's
official [GPMF documentation](https://github.com/gopro/gpmf-parser) says the
HERO10 inherits HERO9/HERO8/HERO6 telemetry behavior: `ACCL` and `GYRO` are
approximately 200 Hz, while `CORI`, `IORI`, and `GRAV` are emitted at video
frame rate. It documents `GPS5` as latitude, longitude, WGS84 altitude, 2D
speed and 3D speed; `GPSF` is fix state; `GPSP` is DOP multiplied by 100; and
`GPSU` is UTC.

There is no known reliable, nondestructive public mapping from a HERO10 serial
number or GPMF field to Hardware A versus Hardware B. An 18 Hz track is
consistent with the M8's 18 Hz ceiling and makes Hardware A plausible, but it
is not proof: the GPMF sample cadence need not equal independent native GNSS
solutions. Reading the chip marking after disassembly would be conclusive but
is not worth damaging a waterproof camera.

## What the actual Kings Canyon files contain

This was checked directly with the repository's `gopropy` checkout against the
three source MP4s; it is not inferred from marketing specifications.

| File | GPS5 samples / measured cadence | GPSF values | GPSP raw values | Result |
| --- | ---: | --- | --- | --- |
| `kings_canyon_1.MP4` | 2,382 / 18.174 Hz | all `0` | all `9999` | no lock; one repeated coordinate |
| `kings_canyon_2.MP4` | 1,307 / 18.172 Hz | 999 at `0`, 308 at `2` | `9999`, then `336`/`337` | no 3D lock; best state was a 2D fix |
| `kings_canyon_3.MP4` | 417 / 18.149 Hz | all `0` | all `9999` | no lock; one repeated coordinate |

For the first file, the measured sensor cadences are 202.031 Hz for both
accelerometer and gyroscope, 59.94 Hz for orientation/image-orientation/gravity,
and 18.174 Hz for GPS5.

This explains Phase 1's `gps_unavailable` result. The `min_fix = 3` gate rejected
every sample, correctly. A stale coordinate in a no-fix packet is not a
position.

### GPSP semantic issue found during this review

The official GPMF definition calls `GPSP` a dimensionless DOP value multiplied
by 100. The current local `gopro-py` code divides it by 100 but labels the
result `gps_error_m`; Mapper consequently routes GPS5 `GPSP` through
`horizontal_accuracy_m` instead of `position_dops`. For example, raw `336`
means DOP 3.36, not a demonstrated 3.36 metre horizontal error.

This did not cause the Kings Canyon rejection—the lack of any 3D fix did—but it
will make future receiver comparisons and weighting ambiguous. Before the A/B
test:

- expose GPS5 `GPSP` as dimensionless `dop`, not metres;
- put it through the DOP gate;
- do not claim `horizontal_accuracy_m` unless a receiver supplies a real
  accuracy estimate such as u-blox `hAcc`;
- retain the raw value and its source semantics.

## Complete logger options

### Shortlist

| Product | GNSS and antenna | Logging | Rate | Battery / weight | Current price | Assessment |
| --- | --- | --- | ---: | --- | ---: | --- |
| [Columbus P-10 Pro](https://columbusgps.com/products/columbus-p-10-pro-submeter-0-5m-gps-gnss-data-logger-and-usb-receiver) | Dual-band L1/L5; GPS, Galileo, GLONASS, BeiDou, QZSS, IRNSS; built-in dual active/patch antennas | CSV, GPX or NMEA to microSD up to 256 GB; USB mass storage | 1 or 5 Hz | 50 h at 1 Hz; 82 g; 55×85×17.7 mm; IP66 | $239 | **Best first purchase.** It changes both frequency diversity and antenna implementation. Vendor claims 0.5 m CEP50 / 1.5 m CEP95 horizontally, but ±15 m vertically. Runtime at 5 Hz is not published. |
| [Columbus P-1 Mark II](https://www.columbus-gps.de/produkte/columbus-p1-mark2-gnss-datenlogger) | Single-band L1 multi-GNSS; built-in antenna | CSV, GPX or NMEA to microSD up to 256 GB; USB mass storage | 1, 5 or 10 Hz | 48 h at 1 Hz; 80 g; IP66 | €159 | Solid and faster, but less diagnostic value: like the HERO10, it is still single-band. Improvement would mostly come from antenna placement and a dedicated power/receiver enclosure. |
| [Qstarz BL-1000GT](https://racing.qstarz.com/Products/BL-1000GT.html) | Single-band L1 GPS/GLONASS/QZSS; built-in antenna | 16 GB microSD; Bluetooth; vendor-specific racing workflow | 10 Hz | 20 h at 10 Hz with Bluetooth; 69 g | availability/price varies | Useful for fast vehicles, not the best walking/mapping choice. It is older, single-band, and its BLE stream uses a Qstarz-specific protocol. |
| [SparkFun RTK Facet](https://www.sparkfun.com/sparkfun-rtk-facet.html) | u-blox ZED-F9P L1/L2 with built-in survey antenna | microSD, USB, Bluetooth/Wi-Fi/radio; rover or base | configurable | >25 h; 583 g; IP53 | $739.95 | Appropriate as a validation reference. Centimetre accuracy requires corrections and an RTK fix; standalone accuracy is not centimetre-grade. Too heavy and expensive for routine body capture. |

The P-10 Pro's 5 Hz limit is not a disadvantage for this use. At walking speed,
5 Hz produces a point roughly every 0.2–0.3 m. Camera poses are paired by
timestamp and interpolated; the GNSS need not run at video or IMU rate.

### Nearly off-the-shelf module stacks

#### Solderless, low-risk logger

| Part | Function | Published electrical/physical facts | Price |
| --- | --- | --- | ---: |
| [SparkFun OpenLog Artemis](https://www.sparkfun.com/sparkfun-openlog-artemis.html) | MCU, microSD logger, RTC, LiPo charger and ICM-20948 9-axis IMU | about 20 mA running; 80 µA sleep; 18 µA deep sleep; logs IMU up to 250 Hz and auto-detects u-blox Qwiic receivers | $59.95 |
| [SparkFun SAM-M10Q breakout](https://www.sparkfun.com/sparkfun-gps-breakout-chip-antenna-sam-m10q-qwiic.html) | GNSS with integrated 15×15 mm patch antenna and PPS | 1.5 m CEP; about 10 mA continuous with four constellations; up to 5 Hz with four constellations, 10 Hz with GPS+Galileo, or 18 Hz single-GNSS | $51.50 |
| protected single-cell 1,500 mAh LiPo with verified JST polarity | field power | OpenLog charges at about 450 mA; use at least a 450 mAh cell | about $15.95 |
| 32 GB microSD, Qwiic cable, switch and simple enclosure | storage and packaging | storage capacity is vastly more than GNSS alone needs | about $30–45 |

Expected total: **$157–172**, before shipping. The two boards connect by Qwiic
without soldering. OpenLog can produce CSV immediately. This is an excellent
instrumentation prototype and includes an independent IMU, but its L1-only
receiver is not a fundamental accuracy step beyond a healthy late-revision
HERO10.

#### Dual-band DIY logger

Replace the SAM-M10Q with the
[SparkFun DAN-F10N breakout](https://www.sparkfun.com/sparkfun-dualband-l1-l5-gnss-breakout-dan-f10n.html):

- $59.95, 23.75 g, integrated 20×20 mm dual-band patch antenna;
- GPS L1/L5, Galileo E1/E5a, BeiDou B1C/B2a, QZSS and NavIC;
- up to 10 Hz, 1.0 m CEP with SBAS / 1.5 m without, and 63 mW continuous;
- optional external active antenna, PPS, NMEA and UBX over UART.

Expected complete total with OpenLog, battery, SD and enclosure:
**$165–185**. This is the most attractive custom hardware direction. It is not
as plug-and-play as the Qwiic stack: basic UART logging is straightforward,
but PPS-aligned GNSS/IMU data and a display require firmware and a few soldered
connections. SparkFun also warns that GPS L5 is not used by default while that
signal remains pre-operational; the exact enabled signal set must be recorded
with each capture.

The bare
[MAX-M10S breakout](https://www.sparkfun.com/sparkfun-gnss-receiver-breakout-max-m10s-qwiic.html)
is $45.95 and consumes roughly 6–25 mA, but it requires a separate SMA antenna.
After adding a decent antenna and enclosure, the integrated-antenna SAM-M10Q is
smaller and simpler. For dual band, an antenna can easily be the largest part:
u-blox's rugged ANN-MB1 L1/L5 active antenna is $99.95 and 164 g including its
5 m cable. The integrated DAN-F10N avoids that burden.

### What an RTK module actually entails

A u-blox ZED-F9P breakout alone is about $260. A usable mobile logger also
needs:

- a good L1/L2 antenna, with an unobstructed sky view and a stable orientation;
- an MCU/logger and microSD;
- roughly 100 mA for the receiver plus logger/communications overhead;
- a phone/cellular/radio path to NTRIP, a nearby base station, or another
  correction service;
- status handling for `no fix`, `float`, and `fixed`; “RTK-capable” must never
  be recorded as if every sample were centimetre accurate;
- an enclosure, battery, cabling and antenna mount.

The integrated RTK Facet consumes about 240 mA worst-case with Bluetooth and
tracking active. Its 6,000 mAh battery yields the published 25-hour runtime.
This is why RTK is a different operational tier, not simply a more accurate
$20 GPS chip.

## Antenna, placement and power

- A patch antenna needs its ceramic face toward the sky. Put a dedicated logger
  on top of a shoulder strap or backpack, not deep in a pocket next to the
  body.
- Keep it away from the GoPro's Wi-Fi, high-current DC converters, large metal
  surfaces and long noisy digital cables. If two receivers are compared, mount
  them close enough to see the same sky but do not stack one directly above the
  other's antenna.
- `MAX-M10S` is not an antenna module. `SAM-M10Q` and `DAN-F10N` are.
- Active antennas contain an LNA and need bias power from a compatible receiver
  board. Passive antennas do not, but layout, cable loss and ground plane
  become more critical.
- A 1,200–1,500 mAh single-cell LiPo is sufficient for an all-day optimized
  logger. A screen left on continuously can consume as much as or more than the
  GNSS receiver.
- Vendor battery life is usually specified at 1 Hz, room temperature and open
  sky. Higher rates, cold weather, repeated acquisition under canopy, Bluetooth
  and screen use all reduce it.

## Mapper ingestion requirements

An external logger should supplement, not replace, GoPro GPMF initially:

```text
GoPro MP4
  ├─ video + camera-relative timestamps
  └─ GPMF IMU/orientation/gravity

external logger sidecar
  ├─ raw NMEA or UBX (preserved unchanged)
  ├─ GNSS UTC / PPS and quality
  └─ optional independent IMU

ingest
  └─ synchronized, normalized GPSTrack -> telemetry/gps.parquet
```

The current `GPSTrack` is already receiver-neutral, but the ingest path accepts
only GoPro video telemetry. Add a sidecar importer before changing alignment.
It must handle these details explicitly:

1. **Time.** Preserve absolute GNSS UTC, convert it to `timestamp_s` relative
   to the video, and store the chosen clock offset and method. Start with UTC
   matching, then retain the existing speed cross-correlation refinement. For
   a custom device, use GNSS PPS to drive a bright LED or short buzzer event
   visible/audible in the GoPro recording; log that event's UTC for an
   unambiguous synchronization marker.
2. **Height reference.** Mapper's canonical field is
   `ellipsoidal_height_m`. NMEA GGA reports mean-sea-level/orthometric altitude
   plus geoid separation. Parse both and add them to recover ellipsoidal
   height. Never copy the GGA altitude field directly into the canonical
   ellipsoidal field.
3. **Quality.** Preserve fix type, number of satellites, HDOP, VDOP, PDOP,
   receiver-reported horizontal/vertical accuracy, correction age and RTK
   state when available. Do not convert DOP to metres.
4. **Vertical weighting.** The P-10 Pro advertises ±15 m vertical accuracy.
   Horizontal and vertical uncertainty need separate weights; a 3D alignment
   must not treat them as equal observations.
5. **Raw provenance.** Keep the original NMEA/UBX/GPX file with a checksum and
   record receiver model, firmware, configured rate, enabled constellations,
   antenna, mount and correction source in capture metadata.

For a u-blox logger, `UBX-NAV-PVT` is preferable to position-only GPX: it
includes UTC/time validity, fix type, satellite count, ellipsoidal height, MSL
height, horizontal/vertical accuracy, velocity, heading and pDOP. Keep raw UBX
even if the first importer only normalizes a subset.

## A/B evaluation

Use one short open-sky loop and one representative difficult route. Record the
same session with:

1. HERO10 after an explicit good pre-lock;
2. a phone track as a free sanity check;
3. P-10 Pro at 5 Hz;
4. optionally a borrowed RTK receiver as reference.

Measure:

- time to first 3D fix;
- percentage of capture with a valid 3D fix;
- gap count and longest gap;
- HDOP/PDOP and reported horizontal/vertical accuracy distributions;
- static scatter before/after the walk and loop endpoint error;
- repeated-route lateral separation;
- position jumps and implied acceleration;
- alignment inlier count, horizontal residual and clock-offset peak quality;
- usable battery life at the selected logging rate.

Do not score only against a basemap. Trails and aerial imagery can themselves
be offset. A repeated route, surveyed point, or temporarily borrowed RTK
reference is a more useful truth source.

## GPS tamagotchi

### Fastest credible prototype

```text
SAM-M10Q GNSS + PPS
          │ Qwiic/I²C
          ▼
OpenLog Artemis ─── microSD
   │  ├─ RTC
   │  ├─ 9-axis IMU
   │  └─ LiPo charger
   ├── 1.3" 128×64 OLED
   ├── two buttons
   └── 1,500 mAh protected LiPo
```

| Part | Price |
| --- | ---: |
| OpenLog Artemis | $59.95 |
| SAM-M10Q Qwiic GNSS with integrated antenna | $51.50 |
| [1.3-inch 128×64 Qwiic OLED](https://www.sparkfun.com/sparkfun-qwiic-oled-1-3in-128x64.html) | $19.95 |
| protected 1,500 mAh single-cell LiPo | about $15.95 |
| 32 GB microSD | about $14.95 |
| cables, two buttons, switch and prototype enclosure | about $18–40 |
| **Prototype total** | **about $180–205** |

The logger and receiver work without custom firmware for a first data-capture
test. The character UI does require custom Artemis/Arduino firmware.

The OpenLog board is about 20 mA while running, the SAM-M10Q about 10 mA with
four constellations, and a mostly lit monochrome OLED is commonly around
25 mA. Including SD writes and regulator losses, use 55–70 mA as the
first always-on design estimate: roughly 14–18 useful hours from 1,500 mAh
after derating. Turning the OLED fully off after 10–20 seconds and buffering SD
writes should put a first prototype around 24–36 hours. A purpose-built PCB and
sleep-oriented firmware can do better.

Estimated assembled weight is 60–85 g:

- battery roughly 25–30 g;
- GNSS, logger, OLED, SD and cabling roughly 20–30 g;
- printed enclosure/buttons roughly 15–25 g.

These are estimates because the breakout vendors do not publish every board's
weight and an enclosure has not been designed.

### Product behavior

Use two modes:

- **Capture mode:** 5 Hz GNSS and high-rate IMU when paired with video.
- **Diary mode:** 1 Hz while moving, a sparse heartbeat while stationary, and
  screen-off sleep. Motion detection can wake the logger.

The pet should communicate device truth rather than hide it:

- mood/animation from fix quality and satellite count;
- walking animation from speed;
- “new place” events from coarse local grid cells;
- battery, logging and last-successful-write indicators always reachable;
- a physical pause/private-mode control;
- a POI/event button that creates a timestamped marker useful for both diary
  events and video synchronization.

Write daily append-only UBX/NMEA plus a tiny recovery index. Buffer samples and
flush every 10–30 seconds rather than rewriting GPX/XML for every point. Convert
to GPX/Parquet on the computer. Capacity is not the constraint: even 100–200
bytes at 1 Hz is only about 9–17 MB/day. Power, crash-safe writes, antenna
placement and privacy are the real constraints.

Location history is unusually sensitive data. Keep v1 local-only, make
recording state obvious, provide a hardware pause, and treat a removable
unencrypted SD card as readable by anyone who obtains the device. Encryption
can be added later, but it changes key recovery and safe-write design.

### Faster UI proof, worse field logger

An [M5Stack Core2 v1.3](https://shop.m5stack.com/products/m5stack-core2-esp32-iot-development-kit-v1-3)
is a $42.90 enclosed ESP32 device with a 2-inch 320×240 touch display, three
virtual buttons, microSD, IMU and a 500 mAh battery. Adding the current $13.95
[M5Stack GPS module](https://shop.m5stack.com/products/gps-module-v2-1-with-antenna-atgm336h)
makes a roughly $70 UI prototype after a card. It is the fastest route to a
cute interface, but the ESP32, color LCD, small battery and external-antenna
GPS make it a poor all-day logger. Use it to validate the interaction, not as
the target electrical architecture.

## Proposed next decision

Purchase one P-10 Pro and run the A/B protocol before selecting custom GNSS
silicon. In parallel, correct GPSP semantics and add an external NMEA/UBX
sidecar importer. If the P-10 materially improves valid-fix coverage and
alignment, use the integrated dual-band DAN-F10N direction for a custom logger
and the lower-power SAM-M10Q direction for the first tamagotchi prototype.
