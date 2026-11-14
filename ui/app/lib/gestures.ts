export type GestureCategory =
  | "zoom"
  | "pinchDrag"
  | "static"
  | "fingerCount"
  | "fingerPointing"
  | "flatHand"
  | "swipeMotion"
  | "flick"
  | "dynamic"
  | "rotation"
  | "scroll"
  | "airDraw"
  | "twoHand";

export type GestureDefinition = {
  name: string;
  label: string;
  category: GestureCategory;
  description: string;
  defaultCommand?: string;
};

export type GesturePrimitive = boolean | number;
export type RecognizerConfig = Record<string, GesturePrimitive>;
export type RecognizerConfigMap = Record<string, RecognizerConfig>;
export type GestureToggleMap = Record<string, boolean>;
export type GestureCommandMap = Record<string, string>;
export type StaticPoseModelGestureGroup =
  | "pose"
  | "finger_count"
  | "finger_pointing"
  | "flat_hand";
export type RecognitionPipelineConfig = {
  fsm: {
    enabled: boolean;
  };
  models: {
    static_pose: {
      enabled: boolean;
      pose: { enabled: boolean };
      finger_count: { enabled: boolean };
      finger_pointing: { enabled: boolean };
      flat_hand: { enabled: boolean };
    };
    motion: {
      enabled: boolean;
    };
  };
};

export const GESTURE_CATEGORIES: Record<
  GestureCategory,
  { label: string; description: string }
> = {
  zoom: {
    label: "Zoom",
    description: "Single-hand pinch-distance expansion and contraction events.",
  },
  pinchDrag: {
    label: "Pinch and drag",
    description:
      "Pinch lifecycle, quick pinch taps, and pinch-held drag events.",
  },
  static: {
    label: "Static poses",
    description:
      "Single-frame hand shapes gated by pose stability and cooldowns.",
  },
  fingerCount: {
    label: "Finger counts",
    description: "Numeric hand poses based on how many fingers are extended.",
  },
  fingerPointing: {
    label: "Finger pointing",
    description:
      "Index-finger pointing direction across cardinal and diagonal directions.",
  },
  flatHand: {
    label: "Flat hand directions",
    description:
      "Open-hand directional poses based on the wrist-to-middle-finger axis.",
  },
  swipeMotion: {
    label: "Swipe motions",
    description: "Cardinal and diagonal fast palm-motion gestures.",
  },
  flick: {
    label: "Flick motions",
    description: "Short, sharp cardinal and diagonal hand movements.",
  },
  dynamic: {
    label: "Dynamic motion",
    description: "Temporal path, wave, push/pull, and throw gestures.",
  },
  rotation: {
    label: "Rotation",
    description:
      "Circle path gestures and single-hand orientation rotation events.",
  },
  scroll: {
    label: "Scroll",
    description: "Two-finger cardinal scrolling gestures.",
  },
  airDraw: {
    label: "Air draw",
    description: "Index-finger drawing lifecycle events for freeform paths.",
  },
  twoHand: {
    label: "Two-hand gestures",
    description:
      "Pair gestures produced when the engine receives two tracked hands.",
  },
};

export const STATIC_POSE_MODEL_GESTURE_GROUPS: Array<{
  name: StaticPoseModelGestureGroup;
  label: string;
  description: string;
}> = [
  {
    name: "pose",
    label: "Static poses",
    description: "Model predictions for the same labels handled by the FSM pose recognizer.",
  },
  {
    name: "finger_count",
    label: "Finger counts",
    description: "Model predictions for FINGER_COUNT_1 through FINGER_COUNT_5.",
  },
  {
    name: "finger_pointing",
    label: "Finger pointing",
    description: "Model predictions for index-finger cardinal and diagonal pointing labels.",
  },
  {
    name: "flat_hand",
    label: "Flat hand directions",
    description: "Model predictions for open-hand cardinal direction labels.",
  },
];

export const GESTURES: GestureDefinition[] = [
  {
    name: "OPEN_PALM",
    label: "Open palm",
    category: "static",
    description: "All fingers extended.",
    defaultCommand: "STOP",
  },
  {
    name: "FIST",
    label: "Fist",
    category: "static",
    description: "All fingers curled.",
    defaultCommand: "GRAB",
  },
  {
    name: "POINT_INDEX",
    label: "Index point",
    category: "static",
    description: "Index finger extended for pointer mode.",
    defaultCommand: "POINTER_MODE",
  },
  {
    name: "TWO_FINGER_POINT",
    label: "Two-finger point",
    category: "static",
    description: "Index and middle fingers extended.",
    defaultCommand: "SCROLL_MODE",
  },
  {
    name: "PEACE_SIGN",
    label: "Peace sign",
    category: "static",
    description: "Index and middle fingers spread apart.",
    defaultCommand: "MODE_TOGGLE",
  },
  {
    name: "OK_SIGN",
    label: "OK sign",
    category: "static",
    description: "Thumb/index pinch with other fingers extended.",
    defaultCommand: "CONFIRM",
  },
  {
    name: "THUMBS_UP",
    label: "Thumbs up",
    category: "static",
    description: "Thumb points upward.",
    defaultCommand: "APPROVE",
  },
  {
    name: "THUMBS_DOWN",
    label: "Thumbs down",
    category: "static",
    description: "Thumb points downward.",
    defaultCommand: "REJECT",
  },
  {
    name: "L_SHAPE",
    label: "L shape",
    category: "static",
    description: "Thumb and index form an L shape.",
    defaultCommand: "REGION_SELECT",
  },
  {
    name: "ROCK_SIGN",
    label: "Rock sign",
    category: "static",
    description: "Index and pinky extended.",
    defaultCommand: "SHORTCUT_ONE",
  },
  {
    name: "HANG_LOOSE",
    label: "Hang loose",
    category: "static",
    description: "Thumb and pinky extended.",
    defaultCommand: "SHORTCUT_TWO",
  },

  {
    name: "FINGER_COUNT_1",
    label: "One finger",
    category: "fingerCount",
    description: "One finger extended.",
    defaultCommand: "COUNT_ONE",
  },
  {
    name: "FINGER_COUNT_2",
    label: "Two fingers",
    category: "fingerCount",
    description: "Two fingers extended.",
    defaultCommand: "COUNT_TWO",
  },
  {
    name: "FINGER_COUNT_3",
    label: "Three fingers",
    category: "fingerCount",
    description: "Three fingers extended.",
    defaultCommand: "COUNT_THREE",
  },
  {
    name: "FINGER_COUNT_4",
    label: "Four fingers",
    category: "fingerCount",
    description: "Four fingers extended.",
    defaultCommand: "COUNT_FOUR",
  },
  {
    name: "FINGER_COUNT_5",
    label: "Five fingers",
    category: "fingerCount",
    description: "Five fingers extended.",
    defaultCommand: "COUNT_FIVE",
  },

  {
    name: "FINGER_POINT_LEFT",
    label: "Point left",
    category: "fingerPointing",
    description: "Index finger points left.",
    defaultCommand: "POINT_LEFT",
  },
  {
    name: "FINGER_POINT_RIGHT",
    label: "Point right",
    category: "fingerPointing",
    description: "Index finger points right.",
    defaultCommand: "POINT_RIGHT",
  },
  {
    name: "FINGER_POINT_UP",
    label: "Point up",
    category: "fingerPointing",
    description: "Index finger points upward.",
    defaultCommand: "POINT_UP",
  },
  {
    name: "FINGER_POINT_DOWN",
    label: "Point down",
    category: "fingerPointing",
    description: "Index finger points downward.",
    defaultCommand: "POINT_DOWN",
  },
  {
    name: "FINGER_POINT_UP_LEFT",
    label: "Point up-left",
    category: "fingerPointing",
    description: "Index finger points diagonally up and left.",
    defaultCommand: "POINT_UP_LEFT",
  },
  {
    name: "FINGER_POINT_UP_RIGHT",
    label: "Point up-right",
    category: "fingerPointing",
    description: "Index finger points diagonally up and right.",
    defaultCommand: "POINT_UP_RIGHT",
  },
  {
    name: "FINGER_POINT_DOWN_LEFT",
    label: "Point down-left",
    category: "fingerPointing",
    description: "Index finger points diagonally down and left.",
    defaultCommand: "POINT_DOWN_LEFT",
  },
  {
    name: "FINGER_POINT_DOWN_RIGHT",
    label: "Point down-right",
    category: "fingerPointing",
    description: "Index finger points diagonally down and right.",
    defaultCommand: "POINT_DOWN_RIGHT",
  },

  {
    name: "FLAT_HAND_LEFT",
    label: "Flat hand left",
    category: "flatHand",
    description: "Open hand points left.",
    defaultCommand: "DIRECTION_LEFT",
  },
  {
    name: "FLAT_HAND_RIGHT",
    label: "Flat hand right",
    category: "flatHand",
    description: "Open hand points right.",
    defaultCommand: "DIRECTION_RIGHT",
  },
  {
    name: "FLAT_HAND_UP",
    label: "Flat hand up",
    category: "flatHand",
    description: "Open hand points upward.",
    defaultCommand: "DIRECTION_UP",
  },
  {
    name: "FLAT_HAND_DOWN",
    label: "Flat hand down",
    category: "flatHand",
    description: "Open hand points downward.",
    defaultCommand: "DIRECTION_DOWN",
  },

  {
    name: "ZOOM_IN",
    label: "Zoom in",
    category: "zoom",
    description: "Single-hand pinch distance expands.",
    defaultCommand: "SCALE_UP",
  },
  {
    name: "ZOOM_OUT",
    label: "Zoom out",
    category: "zoom",
    description: "Single-hand pinch distance contracts.",
    defaultCommand: "SCALE_DOWN",
  },

  {
    name: "PINCH_START",
    label: "Pinch start",
    category: "pinchDrag",
    description: "Thumb/index pinch begins.",
    defaultCommand: "CLICK_DOWN",
  },
  {
    name: "PINCH_HOLD",
    label: "Pinch hold",
    category: "pinchDrag",
    description: "Pinch remains active.",
  },
  {
    name: "PINCH_END",
    label: "Pinch end",
    category: "pinchDrag",
    description: "Pinch releases.",
    defaultCommand: "CLICK_UP",
  },
  {
    name: "DRAG_START",
    label: "Drag start",
    category: "pinchDrag",
    description: "Pinch has held long enough to begin dragging.",
  },
  {
    name: "DRAG_MOVE",
    label: "Drag move",
    category: "pinchDrag",
    description: "Pinched hand moves while dragging.",
    defaultCommand: "POINTER_DRAG",
  },
  {
    name: "DRAG_END",
    label: "Drag end",
    category: "pinchDrag",
    description: "Drag ends when pinch releases.",
  },
  {
    name: "PINCH_TAP",
    label: "Pinch tap",
    category: "pinchDrag",
    description: "Quick pinch and release.",
    defaultCommand: "TAP",
  },

  {
    name: "SWIPE_LEFT",
    label: "Swipe left",
    category: "swipeMotion",
    description: "Fast leftward palm motion.",
    defaultCommand: "NAVIGATE_BACK",
  },
  {
    name: "SWIPE_RIGHT",
    label: "Swipe right",
    category: "swipeMotion",
    description: "Fast rightward palm motion.",
    defaultCommand: "NAVIGATE_FORWARD",
  },
  {
    name: "SWIPE_UP",
    label: "Swipe up",
    category: "swipeMotion",
    description: "Fast upward palm motion.",
  },
  {
    name: "SWIPE_DOWN",
    label: "Swipe down",
    category: "swipeMotion",
    description: "Fast downward palm motion.",
  },
  {
    name: "DIAGONAL_SWIPE_UP_LEFT",
    label: "Diagonal up-left",
    category: "swipeMotion",
    description: "Fast diagonal motion up and left.",
    defaultCommand: "DIAGONAL_UP_LEFT",
  },
  {
    name: "DIAGONAL_SWIPE_UP_RIGHT",
    label: "Diagonal up-right",
    category: "swipeMotion",
    description: "Fast diagonal motion up and right.",
    defaultCommand: "DIAGONAL_UP_RIGHT",
  },
  {
    name: "DIAGONAL_SWIPE_DOWN_LEFT",
    label: "Diagonal down-left",
    category: "swipeMotion",
    description: "Fast diagonal motion down and left.",
    defaultCommand: "DIAGONAL_DOWN_LEFT",
  },
  {
    name: "DIAGONAL_SWIPE_DOWN_RIGHT",
    label: "Diagonal down-right",
    category: "swipeMotion",
    description: "Fast diagonal motion down and right.",
    defaultCommand: "DIAGONAL_DOWN_RIGHT",
  },

  {
    name: "FLICK_LEFT",
    label: "Flick left",
    category: "flick",
    description: "Short sharp leftward motion.",
    defaultCommand: "QUICK_LEFT",
  },
  {
    name: "FLICK_RIGHT",
    label: "Flick right",
    category: "flick",
    description: "Short sharp rightward motion.",
    defaultCommand: "QUICK_RIGHT",
  },
  {
    name: "FLICK_UP",
    label: "Flick up",
    category: "flick",
    description: "Short sharp upward motion.",
    defaultCommand: "QUICK_UP",
  },
  {
    name: "FLICK_DOWN",
    label: "Flick down",
    category: "flick",
    description: "Short sharp downward motion.",
    defaultCommand: "QUICK_DOWN",
  },
  {
    name: "FLICK_UP_LEFT",
    label: "Flick up-left",
    category: "flick",
    description: "Short sharp diagonal up-left motion.",
  },
  {
    name: "FLICK_UP_RIGHT",
    label: "Flick up-right",
    category: "flick",
    description: "Short sharp diagonal up-right motion.",
  },
  {
    name: "FLICK_DOWN_LEFT",
    label: "Flick down-left",
    category: "flick",
    description: "Short sharp diagonal down-left motion.",
  },
  {
    name: "FLICK_DOWN_RIGHT",
    label: "Flick down-right",
    category: "flick",
    description: "Short sharp diagonal down-right motion.",
  },

  {
    name: "WAVE",
    label: "Wave",
    category: "dynamic",
    description: "Repeated left-right oscillation.",
    defaultCommand: "WAVE_TOGGLE",
  },
  {
    name: "PUSH",
    label: "Push",
    category: "dynamic",
    description: "Hand scale grows as if moving toward the camera.",
    defaultCommand: "PUSH_FORWARD",
  },
  {
    name: "PULL",
    label: "Pull",
    category: "dynamic",
    description: "Hand scale shrinks as if moving away from the camera.",
    defaultCommand: "PULL_BACK",
  },
  {
    name: "GRAB_AND_THROW",
    label: "Grab and throw",
    category: "dynamic",
    description: "Fist closes, moves quickly, and releases.",
    defaultCommand: "THROW",
  },

  {
    name: "CIRCLE_CLOCKWISE",
    label: "Circle clockwise",
    category: "rotation",
    description: "Index or palm traces a clockwise loop.",
    defaultCommand: "CYCLE_NEXT",
  },
  {
    name: "CIRCLE_COUNTERCLOCKWISE",
    label: "Circle counterclockwise",
    category: "rotation",
    description: "Index or palm traces a counterclockwise loop.",
    defaultCommand: "CYCLE_PREVIOUS",
  },
  {
    name: "ROTATE_CLOCKWISE",
    label: "Rotate clockwise",
    category: "rotation",
    description: "Single-hand orientation rotates clockwise.",
    defaultCommand: "ROTATE_RIGHT",
  },
  {
    name: "ROTATE_COUNTERCLOCKWISE",
    label: "Rotate counterclockwise",
    category: "rotation",
    description: "Single-hand orientation rotates counterclockwise.",
    defaultCommand: "ROTATE_LEFT",
  },

  {
    name: "SCROLL_UP",
    label: "Scroll up",
    category: "scroll",
    description: "Two-finger vertical scroll upward.",
    defaultCommand: "SCROLL_UP",
  },
  {
    name: "SCROLL_DOWN",
    label: "Scroll down",
    category: "scroll",
    description: "Two-finger vertical scroll downward.",
    defaultCommand: "SCROLL_DOWN",
  },
  {
    name: "SCROLL_LEFT",
    label: "Scroll left",
    category: "scroll",
    description: "Two-finger horizontal scroll left.",
    defaultCommand: "SCROLL_LEFT",
  },
  {
    name: "SCROLL_RIGHT",
    label: "Scroll right",
    category: "scroll",
    description: "Two-finger horizontal scroll right.",
    defaultCommand: "SCROLL_RIGHT",
  },

  {
    name: "AIR_DRAW_START",
    label: "Air draw start",
    category: "airDraw",
    description: "Index drawing begins.",
    defaultCommand: "DRAW_START",
  },
  {
    name: "AIR_DRAW_MOVE",
    label: "Air draw move",
    category: "airDraw",
    description: "Index drawing point moves.",
    defaultCommand: "DRAW_MOVE",
  },
  {
    name: "AIR_DRAW_END",
    label: "Air draw end",
    category: "airDraw",
    description: "Index drawing ends.",
    defaultCommand: "DRAW_END",
  },

  {
    name: "TWO_HAND_ZOOM_IN",
    label: "Two-hand zoom in",
    category: "twoHand",
    description: "Hands move apart.",
    defaultCommand: "SCALE_UP_LARGE",
  },
  {
    name: "TWO_HAND_ZOOM_OUT",
    label: "Two-hand zoom out",
    category: "twoHand",
    description: "Hands move closer.",
    defaultCommand: "SCALE_DOWN_LARGE",
  },
  {
    name: "TWO_HAND_ROTATE_CLOCKWISE",
    label: "Two-hand rotate clockwise",
    category: "twoHand",
    description: "Two-hand angle rotates clockwise.",
    defaultCommand: "ROTATE_RIGHT_LARGE",
  },
  {
    name: "TWO_HAND_ROTATE_COUNTERCLOCKWISE",
    label: "Two-hand rotate counterclockwise",
    category: "twoHand",
    description: "Two-hand angle rotates counterclockwise.",
    defaultCommand: "ROTATE_LEFT_LARGE",
  },
  {
    name: "CLAP",
    label: "Clap",
    category: "twoHand",
    description: "Hands rapidly close together.",
    defaultCommand: "CLAP_ACTION",
  },
  {
    name: "FRAME_RECTANGLE",
    label: "Frame rectangle",
    category: "twoHand",
    description: "Two hands form a framing pose.",
    defaultCommand: "CAMERA_FRAME",
  },
  {
    name: "TWO_HAND_PAN",
    label: "Two-hand pan",
    category: "twoHand",
    description: "Both hands move as a pair.",
    defaultCommand: "PAN_VIEW",
  },
];

export const DEFAULT_RECOGNIZER_CONFIG: RecognizerConfigMap = {
  swipe_motion: {
    enabled: true,
    min_velocity: 0.65,
    min_distance: 0.12,
    diagonal_swipe_min_distance: 0.16,
    diagonal_swipe_axis_balance_ratio: 2.25,
    max_duration_ms: 500,
    cooldown_ms: 300,
    direction_ratio: 1.35,
  },
  zoom: {
    enabled: true,
    min_reference_distance: 0.04,
    min_delta: 0.08,
    min_duration_ms: 120,
    max_duration_ms: 700,
    cooldown_ms: 250,
  },
  pose: {
    enabled: true,
    min_hold_ms: 80,
    cooldown_ms: 250,
    pinch_distance_threshold: 0.12,
    ok_distance_threshold: 0.14,
    spread_distance_threshold: 0.45,
    direction_ratio: 1.25,
    l_shape_min_angle_degrees: 55,
    l_shape_max_angle_degrees: 125,
  },
  finger_count: { enabled: true, min_hold_ms: 80, cooldown_ms: 250 },
  finger_pointing: {
    enabled: true,
    min_hold_ms: 80,
    cooldown_ms: 250,
    direction_ratio: 1.25,
    min_vector_length: 0.05,
  },
  flat_hand: {
    enabled: true,
    min_hold_ms: 80,
    cooldown_ms: 250,
    direction_ratio: 1.25,
  },
  pinch_drag: {
    enabled: true,
    distance_threshold: 0.12,
    release_threshold: 0.16,
    hold_ms: 120,
    drag_pinch_threshold: 0.12,
    drag_release_threshold: 0.16,
    drag_activation_hold_ms: 150,
    drag_min_move: 0.002,
    pinch_tap_distance_threshold: 0.12,
    pinch_tap_release_threshold: 0.16,
    pinch_tap_min_duration_ms: 20,
    pinch_tap_max_duration_ms: 180,
  },
  flick: {
    enabled: true,
    history_size: 16,
    max_history_ms: 1500,
    cooldown_ms: 250,
    direction_ratio: 1.35,
    min_velocity: 1.2,
    min_distance: 0.04,
    max_duration_ms: 180,
  },
  motion: {
    enabled: true,
    history_size: 32,
    max_history_ms: 1500,
    cooldown_ms: 250,
    direction_ratio: 1.35,
    wave_min_amplitude: 0.08,
    wave_min_direction_changes: 3,
    wave_max_duration_ms: 1200,
    push_pull_min_scale_delta: 0.18,
    push_pull_max_duration_ms: 900,
    throw_min_distance: 0.12,
    throw_min_velocity: 0.8,
    throw_max_duration_ms: 900,
  },
  rotation: {
    enabled: true,
    history_size: 32,
    max_history_ms: 1500,
    cooldown_ms: 250,
    circle_min_radius: 0.05,
    circle_min_arc_degrees: 300,
    circle_max_duration_ms: 1400,
    circle_closure_ratio: 1.5,
    rotate_min_degrees: 70,
    rotate_max_duration_ms: 1000,
  },
  scroll: {
    enabled: true,
    history_size: 32,
    max_history_ms: 1500,
    cooldown_ms: 250,
    direction_ratio: 1.35,
    min_distance: 0.08,
    max_duration_ms: 900,
  },
  air_draw: { enabled: true, min_distance: 0.01 },
  two_hand: {
    enabled: true,
    cooldown_ms: 300,
    zoom_min_delta: 0.1,
    rotate_min_degrees: 35,
    pan_min_distance: 0.1,
    clap_distance_threshold: 0.14,
    clap_min_closing_delta: 0.08,
    frame_min_width: 0.3,
  },
};

export const DEFAULT_GESTURE_EVENTS: GestureToggleMap = Object.fromEntries(
  GESTURES.map((gesture) => [gesture.name, true]),
);

export const DEFAULT_VISIBLE_GESTURES: GestureToggleMap = Object.fromEntries(
  GESTURES.map((gesture) => [gesture.name, true]),
);

export const DEFAULT_GESTURE_MAPPING: GestureCommandMap = Object.fromEntries(
  GESTURES.filter((gesture) => gesture.defaultCommand).map((gesture) => [
    gesture.name,
    gesture.defaultCommand ?? "",
  ]),
);

export const DEFAULT_RECOGNITION_CONFIG: RecognitionPipelineConfig = {
  fsm: { enabled: true },
  models: {
    static_pose: {
      enabled: false,
      pose: { enabled: true },
      finger_count: { enabled: true },
      finger_pointing: { enabled: true },
      flat_hand: { enabled: true },
    },
    motion: { enabled: false },
  },
};

export function buildClientConfig(
  recognizers: RecognizerConfigMap,
  gestureEvents: GestureToggleMap,
  gestureMapping: GestureCommandMap,
  recognition: RecognitionPipelineConfig = DEFAULT_RECOGNITION_CONFIG,
) {
  return {
    gestures: cloneRecognizers(recognizers),
    recognition: cloneRecognitionConfig(recognition),
    gesture_events: { ...gestureEvents },
    gesture_mapping: Object.fromEntries(
      Object.entries(gestureMapping)
        .map(([gesture, command]) => [gesture, command.trim()])
        .filter(([, command]) => command.length > 0),
    ),
  };
}

export function cloneRecognitionConfig(
  recognition: RecognitionPipelineConfig,
): RecognitionPipelineConfig {
  return {
    fsm: {
      enabled:
        recognition.fsm?.enabled ?? DEFAULT_RECOGNITION_CONFIG.fsm.enabled,
    },
    models: {
      static_pose: {
        enabled:
          recognition.models?.static_pose?.enabled ??
          DEFAULT_RECOGNITION_CONFIG.models.static_pose.enabled,
        pose: {
          enabled:
            recognition.models?.static_pose?.pose?.enabled ??
            DEFAULT_RECOGNITION_CONFIG.models.static_pose.pose.enabled,
        },
        finger_count: {
          enabled:
            recognition.models?.static_pose?.finger_count?.enabled ??
            DEFAULT_RECOGNITION_CONFIG.models.static_pose.finger_count.enabled,
        },
        finger_pointing: {
          enabled:
            recognition.models?.static_pose?.finger_pointing?.enabled ??
            DEFAULT_RECOGNITION_CONFIG.models.static_pose.finger_pointing.enabled,
        },
        flat_hand: {
          enabled:
            recognition.models?.static_pose?.flat_hand?.enabled ??
            DEFAULT_RECOGNITION_CONFIG.models.static_pose.flat_hand.enabled,
        },
      },
      motion: {
        enabled:
          recognition.models?.motion?.enabled ??
          DEFAULT_RECOGNITION_CONFIG.models.motion.enabled,
      },
    },
  };
}

export function cloneRecognizers(
  recognizers: RecognizerConfigMap,
): RecognizerConfigMap {
  const migrated = migrateLegacyRecognizerSettings(recognizers);
  return Object.fromEntries(
    Object.entries(DEFAULT_RECOGNIZER_CONFIG).map(([name, defaults]) => [
      name,
      Object.fromEntries(
        Object.entries(defaults).map(([key, defaultValue]) => [
          key,
          migrated[name]?.[key] ?? defaultValue,
        ]),
      ),
    ]),
  );
}

function migrateLegacyRecognizerSettings(
  recognizers: RecognizerConfigMap,
): RecognizerConfigMap {
  const migrated = Object.fromEntries(
    Object.entries(recognizers).map(([name, config]) => [name, { ...config }]),
  ) as RecognizerConfigMap;
  const motion = migrated.motion;
  if (!motion) {
    return migrated;
  }

  const pinchDrag = { ...(migrated.pinch_drag ?? {}) };
  const swipeMotion = { ...(migrated.swipe_motion ?? {}) };
  const flick = { ...(migrated.flick ?? {}) };
  const rotation = { ...(migrated.rotation ?? {}) };
  const scroll = { ...(migrated.scroll ?? {}) };
  const airDraw = { ...(migrated.air_draw ?? {}) };

  copyRecognizerValue(motion, swipeMotion, "diagonal_swipe_min_distance");
  copyRecognizerValue(motion, swipeMotion, "diagonal_swipe_axis_balance_ratio");
  copyRecognizerValue(motion, swipeMotion, "direction_ratio");
  copyRecognizerValue(motion, flick, "direction_ratio");
  copyRecognizerValue(motion, scroll, "direction_ratio");
  copyRecognizerValue(motion, flick, "flick_min_velocity", "min_velocity");
  copyRecognizerValue(motion, flick, "flick_min_distance", "min_distance");
  copyRecognizerValue(
    motion,
    flick,
    "flick_max_duration_ms",
    "max_duration_ms",
  );
  copyRecognizerValue(motion, pinchDrag, "pinch_tap_distance_threshold");
  copyRecognizerValue(motion, pinchDrag, "pinch_tap_release_threshold");
  copyRecognizerValue(motion, pinchDrag, "pinch_tap_min_duration_ms");
  copyRecognizerValue(motion, pinchDrag, "pinch_tap_max_duration_ms");
  copyRecognizerValue(motion, airDraw, "air_draw_min_distance", "min_distance");
  for (const key of [
    "history_size",
    "max_history_ms",
    "cooldown_ms",
  ] as const) {
    copyRecognizerValue(motion, rotation, key);
    copyRecognizerValue(motion, scroll, key);
  }
  copyRecognizerValue(motion, rotation, "circle_min_radius");
  copyRecognizerValue(motion, rotation, "circle_min_arc_degrees");
  copyRecognizerValue(motion, rotation, "circle_max_duration_ms");
  copyRecognizerValue(motion, rotation, "circle_closure_ratio");
  copyRecognizerValue(motion, rotation, "rotate_min_degrees");
  copyRecognizerValue(motion, rotation, "rotate_max_duration_ms");
  copyRecognizerValue(motion, scroll, "scroll_min_distance", "min_distance");
  copyRecognizerValue(
    motion,
    scroll,
    "scroll_max_duration_ms",
    "max_duration_ms",
  );

  migrated.pinch_drag = pinchDrag;
  migrated.swipe_motion = swipeMotion;
  migrated.flick = flick;
  migrated.rotation = rotation;
  migrated.scroll = scroll;
  migrated.air_draw = airDraw;
  return migrated;
}

function copyRecognizerValue(
  source: RecognizerConfig,
  target: RecognizerConfig,
  sourceKey: string,
  targetKey = sourceKey,
) {
  if (sourceKey in source && !(targetKey in target)) {
    target[targetKey] = source[sourceKey];
  }
}

export function gestureDefinitionsByCategory(
  category: GestureCategory,
): GestureDefinition[] {
  return GESTURES.filter((gesture) => gesture.category === category);
}

export function coerceRecognizerValue(
  current: GesturePrimitive,
  value: string,
): GesturePrimitive {
  if (typeof current === "boolean") {
    return value === "true";
  }
  const next = Number(value);
  return Number.isFinite(next) ? next : current;
}
