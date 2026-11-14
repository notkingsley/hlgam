# High-Level Gesture Abstraction Middleware

This repository implements a Python-based High-Level Gesture Abstraction Middleware (HLGAM) for gesture-controlled interfaces. The framework converts 21-point hand landmarks into semantic gesture events such as pinch, drag, swipe, zoom, static hand poses, path gestures, and two-hand gestures.

For a full technical and research-oriented description of the current project, see [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).

## Current Implementation Status

The first implementation milestone includes:

- Core data models for landmarks, frames, features, and gesture events.
- Landmark normalization and smoothing utilities.
- Geometric feature extraction for pinch distance, palm center, velocity, direction, and finger curl states.
- FSM-style recognizers for pinch/drag, swipe, and single-hand zoom.
- Static pose, finger-count, finger-pointing, and flat-hand recognizers for open palm, fist, pointing, numeric poses, index-finger direction, OK sign, thumbs, L-shape, rock, hang loose, and directional open-hand poses.
- Configurable learned static-pose inference that can run alongside FSM recognizers when a trained checkpoint is enabled.
- Swipe-motion handles directional swipes; flick handles short sharp motions; pinch-drag handles pinch taps; rotation handles circle paths and orientation rotation; scroll handles two-finger cardinal scrolling; air-draw handles freeform drawing; dynamic motion handles wave, push/pull, and grab-and-throw.
- Two-hand recognizer support for pair zoom, pair rotation, clap, frame rectangle, and two-hand pan through `GestureFramework.process_hands()`.
- A developer-facing `GestureFramework` API for callbacks and event processing.
- Runtime adapters for OpenCV camera capture, MediaPipe hand tracking, and WebSocket broadcasting.
- Optional OSC broadcasting through the `python-osc` extra.
- Unit tests that validate the core logic without requiring a webcam.

## Installation

Install the framework with required runtime dependencies:

```bash
uv sync
```

Install with development tools:

```bash
uv sync --extra dev
```

Install with optional OSC support:

```bash
uv sync --extra osc
```

## Run Tests

```bash
uv run --extra dev pytest
```

## Synthetic Example

```bash
uv run python examples/synthetic_events.py
```

## End-to-End Camera Test

Run the live camera test to start OpenCV capture, MediaPipe hand tracking, HLGAM gesture detection, and event logging:

```bash
uv run python scripts/e2e_camera_test.py --truncate-log
```

Before opening the framework, the script scans available camera indices and lets you choose one interactively. Detected gesture events are printed to the terminal as JSON lines and written to `logs/e2e_camera_events.jsonl` by default.

Useful options:

```bash
uv run python scripts/e2e_camera_test.py --preview --truncate-log
uv run python scripts/e2e_camera_test.py --camera-index 1 --log-file logs/camera_1_events.jsonl
uv run python scripts/e2e_camera_test.py --camera-scan-limit 5
uv run python scripts/e2e_camera_test.py --no-camera-prompt
uv run python scripts/e2e_camera_test.py --no-mirror
uv run python scripts/e2e_camera_test.py --max-frames 300
```

Use `--camera-index` to skip camera detection and force a specific device. Use `--no-camera-prompt` for non-interactive runs; it selects the configured camera if detected, otherwise the first detected readable camera.

Camera input is mirrored horizontally by default so the preview and gesture directions match the user's left/right movement. Use `--no-mirror` if a camera is already oriented correctly.

Press `Ctrl+C` to stop the test. If `--preview` is enabled, press `q` or `Esc` in the preview window to stop.

If you see `module 'mediapipe' has no attribute 'solutions'`, rerun `uv sync --extra dev`. The project pins `mediapipe==0.10.21` because newer MediaPipe releases removed the legacy `solutions.Hands` API used by the tracker.

## WebSocket Server for the UI

Start the gesture engine WebSocket server from the project root:

```bash
uv run python server.py --no-jwt
```

Then open the React Router UI from the `ui` folder:

```bash
cd ui
npm run dev
```

The UI connects to `ws://127.0.0.1:8765` by default. Use `--require-jwt` to require signed JWTs, or paste a valid JWT into the UI if your config requires authentication.

Useful server options:

```bash
uv run python server.py --no-jwt --preview
uv run python server.py --no-jwt --camera-index 1 --max-hands 2
uv run python server.py --no-jwt --source synthetic
uv run python server.py --host 0.0.0.0 --port 8765 --require-jwt
```

## WebSocket JWT Security

WebSocket event streaming requires EdDSA/Ed25519 JWT authentication by default. Configure the verification public key in `.env`:

```env
HLGAM_WS_JWT_PUBLIC_KEY_FILE="./keys/ws_public.pem"
HLGAM_WS_JWT_AUDIENCE="gesture-clients"
HLGAM_WS_JWT_ISSUER="your-token-issuer"
```

Clients may authenticate with `Authorization: Bearer <jwt>` during the WebSocket upgrade request. If no bearer token is present, the first client message must be JSON in this form:

```json
{"token":"<jwt>"}
```

Clients can also provide client-local gesture parameters, per-event enable flags, and gesture mappings in the same first message:

```json
{
	"token": "<jwt>",
	"gestures": {
		"pinch_drag": { "distance_threshold": 0.1, "hold_ms": 80 },
		"swipe_motion": { "min_velocity": 0.75 },
		"pose": { "min_hold_ms": 120 },
		"finger_pointing": { "direction_ratio": 1.25 },
		"rotation": { "rotate_min_degrees": 70 },
		"scroll": { "min_distance": 0.08 },
		"air_draw": { "min_distance": 0.012 },
		"motion": { "wave_min_amplitude": 0.1 },
		"two_hand": { "zoom_min_delta": 0.12 }
	},
	"gesture_mapping": {
		"PINCH_START": "CLIENT_CLICK"
	},
	"recognition": {
		"fsm": { "enabled": true },
		"models": {
			"static_pose": {
				"enabled": false,
				"pose": { "enabled": true },
				"finger_count": { "enabled": true },
				"finger_pointing": { "enabled": true },
				"flat_hand": { "enabled": true }
			},
			"motion": { "enabled": false }
		}
	},
	"gesture_events": {
		"PINCH_HOLD": false,
		"WAVE": true
	}
}
```

During an active stream, clients can update their own recognition settings without affecting other clients:

```json
{
	"type": "update_config",
	"gestures": {
		"pinch_drag": { "pinch_tap_max_duration_ms": 160 },
		"zoom": { "enabled": true, "min_duration_ms": 150 },
		"pose": { "enabled": true },
		"finger_pointing": { "enabled": true },
		"rotation": { "enabled": true },
		"scroll": { "enabled": true },
		"flick": { "min_velocity": 1.4 }
	},
	"recognition": {
		"fsm": { "enabled": false },
		"models": {
			"static_pose": {
				"enabled": true,
				"pose": { "enabled": false },
				"finger_count": { "enabled": true },
				"finger_pointing": { "enabled": false },
				"flat_hand": { "enabled": false }
			},
			"motion": { "enabled": true }
		}
	},
	"gesture_events": { "OPEN_PALM": false, "PINCH_TAP": true },
	"gesture_mapping": { "ZOOM_IN": "CLIENT_SCALE_UP" }
}
```

FSM and learned-model recognition pipelines run independently. A client can disable all deterministic recognizers with `recognition.fsm.enabled: false`, enable the learned static pose model with `recognition.models.static_pose.enabled: true`, enable the learned motion model with `recognition.models.motion.enabled: true`, or keep pipelines enabled together so overlapping events from both sources are delivered and then filtered through `gesture_events` and `gesture_mapping`. The learned static pose model has the same group keys as the FSM recognizers: `pose`, `finger_count`, `finger_pointing`, and `flat_hand`.

Clients may also update learned model settings with shorthand `models.static_pose` and `models.motion` payloads; the server merges them into `recognition.models`.

## WebSocket Video Feed Protocol

The WebSocket broker always exposes the built-in camera as `internal_camera`. Connected clients can also become remote input feeds, event clients can switch which source drives their gesture events, and output clients can receive video frames plus detected hand data from their selected source.

Register a client connection as a remote input feed:

```json
{
	"type": "start_video_input",
	"source_id": "phone_camera",
	"label": "Phone camera",
	"metadata": { "device": "Pixel" }
}
```

Then stream frames using JPEG/PNG base64 data, optionally with precomputed MediaPipe-style landmarks. If `hands` is omitted, the server runs MediaPipe on the submitted frame.

```json
{
	"type": "video_frame",
	"source_id": "phone_camera",
	"frame": {
		"encoding": "jpeg_base64",
		"data": "<base64 jpeg bytes>",
		"timestamp": 1721390000.123,
		"index": 42
	}
}
```

Any event client can switch the source used by its own gesture recognizer:

```json
{ "type": "select_video_source", "source_id": "phone_camera" }
```

Any client can list sources or become an output video feed for its selected source:

```json
{ "type": "list_video_sources" }
{ "type": "start_video_output", "source_id": "internal_camera", "include_features": true }
```

Output clients receive `video_frame` messages containing the selected source id, optional JPEG frame data, detected `hands`, and extracted `features`. Stop input or output mode with `stop_video_input` or `stop_video_output`.

Only authenticated WebSocket clients receive gesture events. See [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md#websocket-jwt-access-control) for full details.

## Optional Runtime Integrations

- Learned static-pose inference requires the `ml` extra and required `HLGAM_STATIC_POSE_*` environment variables when `recognition.models.static_pose.enabled` is true. Set `HLGAM_STATIC_POSE_MODEL_CHECKPOINT`, `HLGAM_STATIC_POSE_LABEL_SCHEMA`, `HLGAM_STATIC_POSE_FEATURE_SCHEMA`, `HLGAM_STATIC_POSE_MODEL_DEVICE`, `HLGAM_STATIC_POSE_MODEL_THRESHOLD`, `HLGAM_STATIC_POSE_MODEL_MIN_HOLD_MS`, and `HLGAM_STATIC_POSE_MODEL_COOLDOWN_MS` in the environment or `.env`.
- Learned motion inference requires the `ml` extra and required `HLGAM_MOTION_*` environment variables when `recognition.models.motion.enabled` is true. Set `HLGAM_MOTION_MODEL_CHECKPOINT`, `HLGAM_MOTION_LABEL_SCHEMA`, `HLGAM_MOTION_FEATURE_SCHEMA`, and optionally `HLGAM_MOTION_MODEL_DEVICE`, `HLGAM_MOTION_MODEL_THRESHOLD`, `HLGAM_MOTION_MODEL_MIN_SEQUENCE_FRAMES`, `HLGAM_MOTION_MODEL_MIN_HOLD_MS`, `HLGAM_MOTION_MODEL_COOLDOWN_MS`, and `HLGAM_MOTION_MODEL_ID` in the environment or `.env`.
- OSC broadcasting requires `python-osc` and can be installed with `uv sync --extra osc`.

Camera capture, hand tracking, WebSocket broadcasting, and YAML configuration loading are installed by default.