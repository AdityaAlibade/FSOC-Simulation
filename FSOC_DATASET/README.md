# FSOC Virtual Camera Tracking System — Synthetic Simulation Dataset

## 1. Dataset Purpose
This dataset is designed and generated for the **FSOC (Free Space Optical Communications) Virtual Camera Tracking System**. In free-space optical communications between satellites and ground optical ground stations (OGS) or inter-satellite links, coarse and fine optical tracking systems must reliably acquire, track, and retain optical beacons under harsh atmospheric disturbances, high-frequency platform vibrations, and dynamic relative orbital motions.

This dataset provides an optical benchmark suite containing **8 core operational scenarios** with exact frame-by-frame ground truth, comprehensive environmental disturbance logs, and standardized benchmarking configurations.

> **Note**: This dataset is completely synthetic, deterministic, and fully reproducible via fixed pseudo-random generator seeds. It uses no webcam, hardware PTZ, or real satellite photographs, in strict accordance with SIH problem-statement requirements.

---

## 2. Directory Structure

```
FSOC_DATASET/
├── videos/
│   ├── SCN_001.mp4
│   ├── SCN_002.mp4
│   ├── SCN_003.mp4
│   ├── SCN_004.mp4
│   ├── SCN_005.mp4
│   ├── SCN_006.mp4
│   ├── SCN_007.mp4
│   └── SCN_008.mp4
├── frames/
│   ├── SCN_001/ (frame_000001.png ... frame_000900.png)
│   ├── SCN_002/ (frame_000001.png ... frame_000900.png)
│   ├── SCN_003/ (frame_000001.png ... frame_000900.png)
│   ├── SCN_004/ (frame_000001.png ... frame_000900.png)
│   ├── SCN_005/ (frame_000001.png ... frame_000900.png)
│   ├── SCN_006/ (frame_000001.png ... frame_000900.png)
│   ├── SCN_007/ (frame_000001.png ... frame_000900.png)
│   └── SCN_008/ (frame_000001.png ... frame_000900.png)
├── ground_truth/
│   ├── SCN_001.csv
│   ├── SCN_002.csv
│   ├── SCN_003.csv
│   ├── SCN_004.csv
│   ├── SCN_005.csv
│   ├── SCN_006.csv
│   ├── SCN_007.csv
│   └── SCN_008.csv
├── scenarios/
│   └── scenarios.csv
├── disturbances/
│   └── disturbances.csv
├── benchmarks/
│   └── benchmark_config.csv
├── dataset_config/
│   └── config.json
└── README.md
```

---

## 3. Camera & Sensor Configuration

The camera simulation parameters adhere strictly to the SIH problem statement:

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Sensor Resolution** | 640 × 480 pixels | Standard tracking camera resolution |
| **Frame Rate** | 30 FPS | Simulation update rate (30 Hz) |
| **Field of View (FOV)** | 4.0° (H) × 3.0° (V) | Pixel pitch: 160 pixels/degree |
| **Virtual Environment** | 2000 × 2000 pixels | Full celestial / orbital observation canvas |
| **Imagery Mode** | Monochrome (Grayscale) | 8-bit single-channel optical detection |
| **Max Pan Speed** | 5.0 °/s (800 px/s) | Gimbal coarse alignment rate |
| **Max Tilt Speed** | 5.0 °/s (800 px/s) | Gimbal coarse alignment rate |
| **Scenario Duration** | 30 seconds | 900 frames per scenario |

---

## 4. Target / Beacon Specifications

The satellite/optical communication beacon is represented by a synthetic square optical target:
- **Default Beacon Size**: 10 × 10 pixels
- **Allowed Size Range**: 5 × 5 to 20 × 20 pixels
- **Tested Sizes**:
  - `SCN_001`, `SCN_002`, `SCN_004`, `SCN_008`: 10 × 10 pixels
  - `SCN_003`, `SCN_007`: 15 × 15 pixels
  - `SCN_005`: 12 × 12 pixels
  - `SCN_006`: 8 × 8 pixels
- **Optical Profile**: Solid high-contrast intensity profile (pixel values 240–255 under clear conditions; attenuated under low-light/fog).

---

## 5. 8 Core Scenarios

| Scenario ID | Scenario Name | Motion Model | Target Size | Atmosphere | Lighting | Disturbances |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SCN_001** | Straight Line + Clear | Straight Line | 10×10 | Clear | Normal | None (Baseline) |
| **SCN_002** | Circular + Gaussian Noise | Circular Orbit | 10×10 | Clear | Normal | Gaussian noise ($\sigma=15.0$) |
| **SCN_003** | Figure-8 + Camera Jitter | Lissajous Figure-8 | 15×15 | Clear | Normal | High-frequency jitter ($\pm15\text{ px}$) |
| **SCN_004** | Random Motion + Platform Motion | Damped Langevin Random Walk | 10×10 | Clear | Normal | Circular platform sway ($\pm12\text{ px}$) |
| **SCN_005** | Straight Line + Fog | Straight Line | 12×12 | Fog (Mie scattering) | Diffuse | Contrast compression + halo |
| **SCN_006** | Circular + Low Light + Noise | Circular Orbit | 8×8 | Low Light | Low Light | Poisson shot noise + low SNR |
| **SCN_007** | Figure-8 + Camera Jitter + Platform Motion | Lissajous Figure-8 | 15×15 | Haze | Normal | Jitter + linear sway + salt & pepper |
| **SCN_008** | Random Motion + Combined Disturbances + Target Loss | Damped Langevin Random Walk | 10×10 | Rain | Variable | Jitter + spiral sway + Gaussian noise + 2 loss periods |

---

## 6. Mathematical Motion Models

### A. Straight Line
The beacon travels at a constant velocity vector between initial and boundary coordinates:
$$x(t) = x_0 + v_x \cdot t, \quad y(t) = y_0 + v_y \cdot t$$

### B. Circular Orbit
The target traces a continuous circle around the optical boresight:
$$x(t) = c_x + R \cdot \cos(\phi_0 + \omega t), \quad y(t) = c_y + R \cdot \sin(\phi_0 + \omega t)$$

### C. Figure-8 (Lissajous Lemniscate)
Represents harmonic orbital perturbations:
$$x(t) = c_x + A_x \cdot \sin(\omega t), \quad y(t) = c_y + A_y \cdot \sin(2\omega t + \phi)$$

### D. Random Walk (Damped Langevin Process)
Continuous, smooth, physically bounded random motion avoiding teleportation:
$$\mathbf{a}(t) = -k (\mathbf{p}(t) - \mathbf{p}_{\text{center}}) - \gamma \mathbf{v}(t) + \mathbf{\eta}(t)$$
$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \mathbf{a}(t) \Delta t$$
$$\mathbf{p}(t + \Delta t) = \mathbf{p}(t) + \mathbf{v}(t) \Delta t$$

---

## 7. Disturbance Models

### A. Camera Jitter
High-frequency mechanical jitter (reaction wheel imbalances, gimbal tremors):
- Harmonic decomposition with stochastic component
- Bounded strictly within $\pm 20$ pixels/frame (actual peak $\le 15$ px).

### B. Platform Motion
Low-frequency structural sway (satellite bus attitude oscillation, naval/ground mast sway):
- Supported trajectories: linear, circular, spiral.
- Bounded strictly within $\pm 20$ pixels/frame.

### C. Atmospheric Conditions
- **Clear**: Baseline vacuum/clean optical path.
- **Haze**: Aerosol forward scattering; dynamic range reduced by 18%, background offset $+22\text{ DN}$.
- **Fog**: Heavy Mie scattering; diffuse Gaussian blur with $11\times11$ kernel, background veil $+45\text{ DN}$, target halo.
- **Rain**: Refractive transmission degradation and directional precipitation streaks.
- **Low Light**: Minimal incident photons, sensor gain adjusted, target intensity attenuated to $\sim80\text{ DN}$.

### D. Sensor Noise
- **Gaussian Noise**: Thermal and amplifier noise $\mathcal{N}(\mu, \sigma^2)$ with $\sigma \in [15.0, 18.0]$.
- **Poisson Noise**: Photon shot noise following Poisson distribution.
- **Salt & Pepper**: Random sensor dead/saturated pixels ($p \approx 0.006$).

### E. Target Loss & Reacquisition (SCN_008)
- Controlled target eclipse / deep fade periods:
  - **Loss Period 1**: Frames 240 to 300 (61 frames $\approx 2.03\text{ s}$)
  - **Loss Period 2**: Frames 600 to 660 (61 frames $\approx 2.03\text{ s}$)
- Ground truth coordinates maintain continuous tracking trajectory with `visible = false`.

---

## 8. Metadata & File Formats

### Ground Truth CSV (`ground_truth/SCN_xxx.csv`)
Contains exact ground truth for all 900 frames:
```csv
scenario_id,video_id,frame_id,timestamp,gt_x,gt_y,gt_width,gt_height,gt_center_x,gt_center_y,visible,motion_type
SCN_001,SCN_001,1,0.0,95,85,10,10,100.0,90.0,true,straight_line
```

### Disturbances CSV (`disturbances/disturbances.csv`)
Frame-by-frame log of all active disturbances (7,200 total records):
```csv
scenario_id,frame_id,noise_type,noise_strength,camera_jitter_x,camera_jitter_y,platform_motion_type,platform_motion_x,platform_motion_y,atmospheric_condition,lighting_condition,target_visible
SCN_003,1,none,0.0,0.0,5.0,none,0.0,0.0,Clear,Normal,true
```

### Scenario Metadata (`scenarios/scenarios.csv`)
High-level configuration parameters for all 8 scenarios.

### Benchmark Configuration (`benchmarks/benchmark_config.csv`)
Standardized benchmark parameters (`BM_001` to `BM_008`) for downstream evaluation of detection, centroid estimation, and tracking algorithms.

### Dataset Configuration (`dataset_config/config.json`)
Global dataset configuration with fixed base random seed `26169`.

---

## 9. Generation and Reproducibility

The dataset is generated via `generate_dataset.py` using NumPy deterministic random state seeds:
- Base seed: `26169`
- Scenario seeds: `26169 + scenario_index`
- Independent deterministic streams for background stars, camera jitter, and sensor noise.

To regenerate or validate:
```bash
python generate_dataset.py
python validate_dataset.py
```

---

## 10. Automated Validation Results (14 / 14 Passed)

Validation executed via [validate_dataset.py](file:///c:/Users/aliba/Desktop/FSOC/validate_dataset.py):

| # | Validation Criterion | Status | Detail |
|---|---|---|---|
| 1 | Every video has exactly 900 frames | **PASSED [OK]** | 8 videos × 900 frames verified via OpenCV |
| 2 | Every video is 640×480 | **PASSED [OK]** | All 8 videos exact 640×480 resolution |
| 3 | Every video is 30 FPS | **PASSED [OK]** | All 8 videos exact 30 FPS |
| 4 | Every frame has corresponding ground truth | **PASSED [OK]** | Exactly 900 records in each of 8 ground truth CSVs |
| 5 | Every frame has corresponding disturbance metadata | **PASSED [OK]** | Exactly 7,200 records in disturbances.csv |
| 6 | Ground-truth coordinates within 640×480 when target visible | **PASSED [OK]** | Bounding box `[gt_x, gt_y, gt_w, gt_h]` strictly bounded |
| 7 | Target size remains within 5–20 pixels | **PASSED [OK]** | Verified across all scenarios (8×8, 10×10, 12×12, 15×15) |
| 8 | Motion is continuous | **PASSED [OK]** | No teleportation; frame-to-frame displacements strictly bounded |
| 9 | No duplicate frame IDs | **PASSED [OK]** | Unique frame IDs 1..900 in each scenario |
| 10 | No missing CSV records | **PASSED [OK]** | All records present across ground_truth, disturbances, scenarios, benchmarks |
| 11 | PNG frame count equals MP4 frame count | **PASSED [OK]** | Exactly 900 PNG files per folder = 900 MP4 video frames |
| 12 | Video and PNG frames correspond exactly | **PASSED [OK]** | Structural & pixel fidelity verified across sample frames |
| 13 | All random generation uses fixed seeds | **PASSED [OK]** | Base seed 26169 in dataset_config/config.json |
| 14 | SCN_008 contains intentional target-loss/reacquisition | **PASSED [OK]** | 2 distinct loss periods (frames 240–300 & 600–660) with true continuous coordinates |

---

## 11. Summary Specification

```
Total scenarios: 8
Total videos: 8
Total frames: 7200
Resolution: 640×480
FPS: 30
Duration/scenario: 30 sec
```

