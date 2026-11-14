import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";

import type { Route } from "./+types/home";
import {
  DEFAULT_GESTURE_EVENTS,
  DEFAULT_GESTURE_MAPPING,
  DEFAULT_RECOGNITION_CONFIG,
  DEFAULT_RECOGNIZER_CONFIG,
  DEFAULT_VISIBLE_GESTURES,
  GESTURE_CATEGORIES,
  GESTURES,
  STATIC_POSE_MODEL_GESTURE_GROUPS,
  buildClientConfig,
  cloneRecognitionConfig,
  cloneRecognizers,
  coerceRecognizerValue,
  gestureDefinitionsByCategory,
  type GestureCategory,
  type GestureCommandMap,
  type GesturePrimitive,
  type GestureToggleMap,
  type RecognitionPipelineConfig,
  type RecognizerConfigMap,
  type StaticPoseModelGestureGroup,
} from "~/lib/gestures";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Gesture Engine" },
    { name: "description", content: "Real-time WebSocket dashboard for HLGAM gesture events." },
  ];
}

type ConnectionState = "idle" | "connecting" | "connected" | "closing" | "closed" | "error";

type GestureEventPayload = {
  gesture: string;
  phase: string;
  confidence: number;
  timestamp: number;
  position?: Record<string, number> | null;
  velocity?: Record<string, number> | null;
  command?: string | null;
  metadata?: Record<string, unknown>;
};

type StreamEvent = GestureEventPayload & {
  id: string;
  receivedAt: number;
};

type ControlMessage = {
  id: string;
  receivedAt: number;
  type: string;
  payload: Record<string, unknown>;
};

type VideoPoint = {
  x: number;
  y: number;
  z: number;
};

type VideoHand = {
  landmarks: VideoPoint[];
  handedness: string | null;
  score: number | null;
  imageSize: [number, number] | null;
};

type VideoImageFrame = {
  data: string;
  mimeType: string;
  encoding: string | null;
  width: number | null;
  height: number | null;
  index: number | null;
  timestamp: number | null;
};

type VideoPreviewFrame = {
  sourceId: string;
  sourceLabel: string;
  receivedAt: number;
  timestamp: number | null;
  frame: VideoImageFrame | null;
  hands: VideoHand[];
  featureCount: number;
};

type BrowserCameraDevice = {
  deviceId: string;
  label: string;
};

const STORAGE_PREFIX = "hlgam.ui";
const DEFAULT_ENDPOINT = "ws://127.0.0.1:8765";
const DEFAULT_CHART_WINDOW_SECONDS = 15;
const MIN_CHART_WINDOW_SECONDS = 5;
const MAX_CHART_WINDOW_SECONDS = 600;
const CHART_TICK_MS = 100;
const DEFAULT_EVENT_LIMIT = 100;
const MIN_EVENT_LIMIT = 1;
const MAX_EVENT_LIMIT = 5000;
const MAX_CONTROLS = 40;
const DEFAULT_VIDEO_SOURCE_ID = "internal_camera";
const VIDEO_JPEG_QUALITY = 82;
const BROWSER_INPUT_SOURCE_PREFIX = "browser_camera_";
const BROWSER_INPUT_JPEG_QUALITY = 0.78;
const DEFAULT_BROWSER_INPUT_FPS = 8;
const MIN_BROWSER_INPUT_FPS = 1;
const MAX_BROWSER_INPUT_FPS = 30;
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
] as const;

export default function Home() {
  const [endpoint, setEndpoint] = usePersistentState(`${STORAGE_PREFIX}.endpoint`, DEFAULT_ENDPOINT);
  const [token, setToken] = useState("");
  const [autoReconnect, setAutoReconnect] = usePersistentState(`${STORAGE_PREFIX}.autoReconnect`, true);
  const [recognizers, setRecognizers] = usePersistentState<RecognizerConfigMap>(
    `${STORAGE_PREFIX}.recognizers`,
    DEFAULT_RECOGNIZER_CONFIG,
  );
  const [recognitionConfig, setRecognitionConfig] = usePersistentState<RecognitionPipelineConfig>(
    `${STORAGE_PREFIX}.recognitionConfig`,
    DEFAULT_RECOGNITION_CONFIG,
  );
  const [gestureEvents, setGestureEvents] = usePersistentState<GestureToggleMap>(
    `${STORAGE_PREFIX}.gestureEvents`,
    DEFAULT_GESTURE_EVENTS,
  );
  const [visibleGestures, setVisibleGestures] = usePersistentState<GestureToggleMap>(
    `${STORAGE_PREFIX}.visibleGestures`,
    DEFAULT_VISIBLE_GESTURES,
  );
  const [gestureMapping, setGestureMapping] = usePersistentState<GestureCommandMap>(
    `${STORAGE_PREFIX}.gestureMapping`,
    DEFAULT_GESTURE_MAPPING,
  );
  const [chartWindowSeconds, setChartWindowSeconds] = usePersistentState(
    `${STORAGE_PREFIX}.chartWindowSeconds`,
    DEFAULT_CHART_WINDOW_SECONDS,
  );
  const [showConfidenceTimeline, setShowConfidenceTimeline] = usePersistentState(
    `${STORAGE_PREFIX}.showConfidenceTimeline`,
    true,
  );
  const [eventLimit, setEventLimit] = usePersistentState(`${STORAGE_PREFIX}.eventLimit`, DEFAULT_EVENT_LIMIT);
  const [videoSourceId, setVideoSourceId] = usePersistentState(
    `${STORAGE_PREFIX}.videoSourceId`,
    DEFAULT_VIDEO_SOURCE_ID,
  );
  const [browserCameraDeviceId, setBrowserCameraDeviceId] = usePersistentState(
    `${STORAGE_PREFIX}.browserCameraDeviceId`,
    "",
  );
  const [browserInputFps, setBrowserInputFps] = usePersistentState(
    `${STORAGE_PREFIX}.browserInputFps`,
    DEFAULT_BROWSER_INPUT_FPS,
  );
  const [chartNow, setChartNow] = useState(() => Date.now());
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [videoFrame, setVideoFrame] = useState<VideoPreviewFrame | null>(null);
  const [videoFrameCount, setVideoFrameCount] = useState(0);
  const [browserCameraDevices, setBrowserCameraDevices] = useState<BrowserCameraDevice[]>([]);
  const [browserInputSourceId, setBrowserInputSourceId] = useState<string | null>(null);
  const [browserInputFrameCount, setBrowserInputFrameCount] = useState(0);
  const [controlMessages, setControlMessages] = useState<ControlMessage[]>([]);
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [videoConnectionState, setVideoConnectionState] = useState<ConnectionState>("idle");
  const [browserInputConnectionState, setBrowserInputConnectionState] = useState<ConnectionState>("idle");
  const [statusMessage, setStatusMessage] = useState("Disconnected");
  const [videoStatusMessage, setVideoStatusMessage] = useState("Video preview stopped");
  const [browserInputStatusMessage, setBrowserInputStatusMessage] = useState("Browser camera input stopped");
  const [lastSentConfigAt, setLastSentConfigAt] = useState<number | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const videoSocketRef = useRef<WebSocket | null>(null);
  const browserInputSocketRef = useRef<WebSocket | null>(null);
  const browserInputStreamRef = useRef<MediaStream | null>(null);
  const browserInputVideoRef = useRef<HTMLVideoElement | null>(null);
  const browserInputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const browserInputCaptureTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const browserInputSourceIdRef = useRef<string | null>(null);
  const browserInputRegisteredSourceIdRef = useRef<string | null>(null);
  const browserInputFrameIndexRef = useRef(0);
  const browserInputFpsRef = useRef(DEFAULT_BROWSER_INPUT_FPS);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const manualCloseRef = useRef(false);
  const eventLimitRef = useRef(DEFAULT_EVENT_LIMIT);
  const clampedEventLimit = clampEventLimit(eventLimit);
  const clampedBrowserInputFps = clampBrowserInputFps(browserInputFps);

  const currentClientConfig = useMemo(
    () => buildClientConfig(recognizers, gestureEvents, gestureMapping, recognitionConfig),
    [gestureEvents, gestureMapping, recognitionConfig, recognizers],
  );

  const visibleEvents = useMemo(
    () => events.filter((event) => visibleGestures[event.gesture] ?? true),
    [events, visibleGestures],
  );

  const stats = useMemo(() => buildStats(events, visibleEvents, gestureEvents), [events, gestureEvents, visibleEvents]);

  const addControlMessage = useCallback((type: string, payload: Record<string, unknown>) => {
    setControlMessages((current) => [
      { id: createId(), receivedAt: Date.now(), type, payload },
      ...current,
    ].slice(0, MAX_CONTROLS));
  }, []);

  const sendUpdate = useCallback((config = currentClientConfig) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setStatusMessage("Config is staged locally; connect to apply it to the engine.");
      return false;
    }
    socket.send(JSON.stringify({ type: "update_config", ...config }));
    setLastSentConfigAt(Date.now());
    return true;
  }, [currentClientConfig]);

  const selectEventDetectionSource = useCallback((sourceId: string, label = sourceId) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setStatusMessage(`Connect the event stream to use ${label} for detection.`);
      return false;
    }
    socket.send(JSON.stringify({ type: "select_video_source", source_id: sourceId }));
    setStatusMessage(`Selecting ${label} for event detection`);
    return true;
  }, []);

  const disconnect = useCallback(() => {
    manualCloseRef.current = true;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    setConnectionState("closing");
    socketRef.current?.close(1000, "client disconnected");
  }, []);

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN || socketRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }
    manualCloseRef.current = false;
    setConnectionState("connecting");
    setStatusMessage(`Connecting to ${endpoint}`);

    try {
      const socket = new WebSocket(endpoint);
      socketRef.current = socket;

      socket.onopen = () => {
        reconnectAttemptRef.current = 0;
        setConnectionState("connected");
        setStatusMessage("Connected and streaming");
        const payload = token.trim()
          ? { token: token.trim(), ...currentClientConfig }
          : { type: "update_config", ...currentClientConfig };
        socket.send(JSON.stringify(payload));
        setLastSentConfigAt(Date.now());
        const browserSourceId = browserInputRegisteredSourceIdRef.current;
        if (browserSourceId) {
          selectEventDetectionSource(browserSourceId, "browser camera input");
        }
      };

      socket.onmessage = (message) => {
        const parsed = parseSocketMessage(message.data);
        if (!parsed) {
          addControlMessage("invalid_json", { raw: String(message.data) });
          return;
        }
        if (isGestureEvent(parsed)) {
          setEvents((current) => [
            { ...parsed, id: createId(), receivedAt: Date.now() },
            ...current,
          ].slice(0, eventLimitRef.current));
          return;
        }
        const type = typeof parsed.type === "string" ? parsed.type : "control";
        addControlMessage(type, parsed);
      };

      socket.onerror = () => {
        setConnectionState("error");
        setStatusMessage("WebSocket error. Check the engine, URL, JWT, and CORS/network reachability.");
      };

      socket.onclose = (event) => {
        socketRef.current = null;
        setConnectionState(event.wasClean ? "closed" : "error");
        setStatusMessage(`Disconnected${event.reason ? `: ${event.reason}` : ""}`);
        if (!manualCloseRef.current && autoReconnect && event.code !== 1008) {
          const delay = Math.min(1000 * 2 ** reconnectAttemptRef.current, 8000);
          reconnectAttemptRef.current += 1;
          reconnectTimerRef.current = setTimeout(connect, delay);
        }
      };
    } catch (error) {
      setConnectionState("error");
      setStatusMessage(error instanceof Error ? error.message : "Unable to open WebSocket.");
    }
  }, [addControlMessage, autoReconnect, currentClientConfig, endpoint, selectEventDetectionSource, token]);

  const disconnectVideo = useCallback(() => {
    const socket = videoSocketRef.current;
    if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
      return;
    }
    setVideoFrame(null);
    setVideoConnectionState("closing");
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop_video_output" }));
    }
    socket.close(1000, "video preview stopped");
  }, []);

  const connectVideo = useCallback(() => {
    const existingSocket = videoSocketRef.current;
    if (existingSocket?.readyState === WebSocket.OPEN) {
      setVideoFrame(null);
      setVideoFrameCount(0);
      existingSocket.send(JSON.stringify(buildVideoOutputRequest(videoSourceId, token)));
      setVideoStatusMessage(`Switching preview to ${videoSourceId.trim() || DEFAULT_VIDEO_SOURCE_ID}`);
      return;
    }
    if (existingSocket?.readyState === WebSocket.CONNECTING) {
      return;
    }

    setVideoConnectionState("connecting");
  setVideoFrame(null);
  setVideoFrameCount(0);
    setVideoStatusMessage(`Opening video preview from ${endpoint}`);
    try {
      const socket = new WebSocket(endpoint);
      videoSocketRef.current = socket;

      socket.onopen = () => {
        setVideoConnectionState("connected");
        setVideoStatusMessage(`Requesting video source ${videoSourceId.trim() || DEFAULT_VIDEO_SOURCE_ID}`);
        socket.send(JSON.stringify(buildVideoOutputRequest(videoSourceId, token)));
      };

      socket.onmessage = (message) => {
        const parsed = parseSocketMessage(message.data);
        if (!parsed) {
          setVideoStatusMessage("Video preview received an invalid JSON message.");
          return;
        }
        const previewFrame = normalizeVideoPreviewFrame(parsed);
        if (previewFrame) {
          setVideoFrame(previewFrame);
          setVideoFrameCount((count) => count + 1);
          setVideoStatusMessage(`Streaming ${previewFrame.sourceLabel || previewFrame.sourceId}`);
          return;
        }
        const type = typeof parsed.type === "string" ? parsed.type : "control";
        if (type === "error") {
          const messageText = typeof parsed.message === "string" ? parsed.message : "video preview error";
          setVideoStatusMessage(messageText);
          return;
        }
        setVideoStatusMessage(labelForVideoControl(type, parsed));
      };

      socket.onerror = () => {
        setVideoConnectionState("error");
        setVideoStatusMessage("Video preview WebSocket error. Check the server, URL, JWT, and selected source.");
      };

      socket.onclose = (event) => {
        if (videoSocketRef.current === socket) {
          videoSocketRef.current = null;
        }
        setVideoConnectionState(event.wasClean ? "closed" : "error");
        setVideoStatusMessage(`Video preview stopped${event.reason ? `: ${event.reason}` : ""}`);
      };
    } catch (error) {
      setVideoConnectionState("error");
      setVideoStatusMessage(error instanceof Error ? error.message : "Unable to open video preview WebSocket.");
    }
  }, [endpoint, token, videoSourceId]);

  const refreshBrowserCameraDevices = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
      setBrowserInputStatusMessage("Browser camera selection is not available in this environment.");
      return;
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === "videoinput").map((device, index) => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${index + 1}`,
      }));
      setBrowserCameraDevices(cameras);
      if (browserCameraDeviceId && !cameras.some((device) => device.deviceId === browserCameraDeviceId)) {
        setBrowserCameraDeviceId("");
      }
    } catch (error) {
      setBrowserInputStatusMessage(error instanceof Error ? error.message : "Unable to list browser cameras.");
    }
  }, [browserCameraDeviceId, setBrowserCameraDeviceId]);

  const stopBrowserFrameCapture = useCallback(() => {
    if (browserInputCaptureTimerRef.current) {
      clearInterval(browserInputCaptureTimerRef.current);
      browserInputCaptureTimerRef.current = null;
    }
  }, []);

  const startBrowserFrameCapture = useCallback((socket: WebSocket, sourceId: string) => {
    stopBrowserFrameCapture();
    const sendFrame = () => {
      if (browserInputSocketRef.current !== socket || socket.readyState !== WebSocket.OPEN) {
        return;
      }
      if (browserInputRegisteredSourceIdRef.current !== sourceId) {
        return;
      }
      const video = browserInputVideoRef.current;
      if (!video) {
        return;
      }
      if (!browserInputCanvasRef.current) {
        browserInputCanvasRef.current = document.createElement("canvas");
      }
      const frameData = captureBrowserVideoFrame(video, browserInputCanvasRef.current);
      if (!frameData) {
        return;
      }
      const frameIndex = browserInputFrameIndexRef.current;
      browserInputFrameIndexRef.current += 1;
      socket.send(JSON.stringify(buildBrowserVideoFrameMessage(sourceId, frameData, frameIndex)));
      setBrowserInputFrameCount((count) => count + 1);
    };

    sendFrame();
    browserInputCaptureTimerRef.current = setInterval(
      sendFrame,
      Math.round(1000 / Math.max(MIN_BROWSER_INPUT_FPS, browserInputFpsRef.current)),
    );
  }, [stopBrowserFrameCapture]);

  const stopBrowserCameraInput = useCallback(() => {
    const registeredSourceId = browserInputRegisteredSourceIdRef.current;
    const socket = browserInputSocketRef.current;
    stopBrowserFrameCapture();
    stopMediaStream(browserInputStreamRef.current);
    browserInputStreamRef.current = null;
    if (browserInputVideoRef.current) {
      browserInputVideoRef.current.srcObject = null;
    }
    browserInputSourceIdRef.current = null;
    browserInputRegisteredSourceIdRef.current = null;
    setBrowserInputSourceId(null);
    setBrowserInputFrameCount(0);
    setBrowserInputConnectionState(socket && socket.readyState !== WebSocket.CLOSED ? "closing" : "closed");
    setBrowserInputStatusMessage("Browser camera input stopped");
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop_video_input" }));
    }
    if (socket && socket.readyState !== WebSocket.CLOSED && socket.readyState !== WebSocket.CLOSING) {
      socket.close(1000, "browser camera input stopped");
    }
    if (registeredSourceId) {
      selectEventDetectionSource(DEFAULT_VIDEO_SOURCE_ID, "internal camera");
    }
  }, [selectEventDetectionSource, stopBrowserFrameCapture]);

  const startBrowserCameraInput = useCallback(async () => {
    const existingSocket = browserInputSocketRef.current;
    if (existingSocket?.readyState === WebSocket.OPEN || existingSocket?.readyState === WebSocket.CONNECTING) {
      return;
    }
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setBrowserInputConnectionState("error");
      setBrowserInputStatusMessage("Browser camera input requires navigator.mediaDevices.getUserMedia.");
      return;
    }

    const sourceId = createBrowserInputSourceId();
    const cameraLabel = labelForBrowserCamera(browserCameraDevices, browserCameraDeviceId);
    browserInputSourceIdRef.current = sourceId;
    browserInputRegisteredSourceIdRef.current = null;
    browserInputFrameIndexRef.current = 0;
    setBrowserInputSourceId(sourceId);
    setBrowserInputFrameCount(0);
    setBrowserInputConnectionState("connecting");
    setBrowserInputStatusMessage(`Requesting ${cameraLabel}`);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: browserVideoConstraints(browserCameraDeviceId, clampedBrowserInputFps),
      });
      browserInputStreamRef.current = stream;

      const video = browserInputVideoRef.current ?? document.createElement("video");
      browserInputVideoRef.current = video;
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play();
      void refreshBrowserCameraDevices();

      setBrowserInputStatusMessage(`Opening input feed ${sourceId}`);
      const socket = new WebSocket(endpoint);
      browserInputSocketRef.current = socket;

      socket.onopen = () => {
        setBrowserInputStatusMessage(`Registering ${sourceId}`);
        socket.send(JSON.stringify(buildBrowserVideoInputRequest(sourceId, `Browser camera (${cameraLabel})`, token, browserCameraDeviceId)));
      };

      socket.onmessage = (message) => {
        const parsed = parseSocketMessage(message.data);
        if (!parsed) {
          setBrowserInputStatusMessage("Browser input received an invalid JSON message.");
          return;
        }
        const type = typeof parsed.type === "string" ? parsed.type : "control";
        addControlMessage(type, parsed);
        if (type === "video_input_started") {
          const registeredSourceId = stringOrNull(parsed.source_id) ?? sourceId;
          browserInputSourceIdRef.current = registeredSourceId;
          browserInputRegisteredSourceIdRef.current = registeredSourceId;
          setBrowserInputSourceId(registeredSourceId);
          setBrowserInputConnectionState("connected");
          startBrowserFrameCapture(socket, registeredSourceId);
          const selected = selectEventDetectionSource(registeredSourceId, "browser camera input");
          setBrowserInputStatusMessage(
            selected
              ? `Streaming ${cameraLabel}; event detection is switching to this feed.`
              : `Streaming ${cameraLabel}; connect the event stream to detect from this feed.`,
          );
          return;
        }
        if (type === "error") {
          const messageText = typeof parsed.message === "string" ? parsed.message : "browser camera input error";
          setBrowserInputConnectionState("error");
          setBrowserInputStatusMessage(messageText);
          return;
        }
        setBrowserInputStatusMessage(labelForBrowserInputControl(type, parsed));
      };

      socket.onerror = () => {
        stopBrowserFrameCapture();
        setBrowserInputConnectionState("error");
        setBrowserInputStatusMessage("Browser camera input WebSocket error. Check the server, URL, JWT, and camera permission.");
      };

      socket.onclose = (event) => {
        if (browserInputSocketRef.current === socket) {
          browserInputSocketRef.current = null;
        }
        stopBrowserFrameCapture();
        stopMediaStream(browserInputStreamRef.current);
        browserInputStreamRef.current = null;
        if (browserInputVideoRef.current) {
          browserInputVideoRef.current.srcObject = null;
        }
        browserInputSourceIdRef.current = null;
        browserInputRegisteredSourceIdRef.current = null;
        setBrowserInputSourceId(null);
        setBrowserInputConnectionState(event.wasClean ? "closed" : "error");
        setBrowserInputStatusMessage(`Browser camera input stopped${event.reason ? `: ${event.reason}` : ""}`);
      };
    } catch (error) {
      stopBrowserFrameCapture();
      stopMediaStream(browserInputStreamRef.current);
      browserInputStreamRef.current = null;
      if (browserInputVideoRef.current) {
        browserInputVideoRef.current.srcObject = null;
      }
      browserInputSourceIdRef.current = null;
      browserInputRegisteredSourceIdRef.current = null;
      setBrowserInputSourceId(null);
      setBrowserInputConnectionState("error");
      setBrowserInputStatusMessage(error instanceof Error ? error.message : "Unable to start browser camera input.");
    }
  }, [addControlMessage, browserCameraDeviceId, browserCameraDevices, clampedBrowserInputFps, endpoint, refreshBrowserCameraDevices, selectEventDetectionSource, startBrowserFrameCapture, stopBrowserFrameCapture, token]);

  useEffect(() => {
    if (!showConfidenceTimeline) {
      return;
    }
    setChartNow(Date.now());
    const timer = window.setInterval(() => setChartNow(Date.now()), CHART_TICK_MS);
    return () => window.clearInterval(timer);
  }, [showConfidenceTimeline]);

  useEffect(() => {
    eventLimitRef.current = clampedEventLimit;
    setEvents((current) => current.slice(0, clampedEventLimit));
  }, [clampedEventLimit]);

  useEffect(() => {
    browserInputFpsRef.current = clampedBrowserInputFps;
    const socket = browserInputSocketRef.current;
    const sourceId = browserInputRegisteredSourceIdRef.current;
    if (socket?.readyState === WebSocket.OPEN && sourceId) {
      startBrowserFrameCapture(socket, sourceId);
    }
  }, [clampedBrowserInputFps, startBrowserFrameCapture]);

  useEffect(() => {
    void refreshBrowserCameraDevices();
  }, [refreshBrowserCameraDevices]);

  useEffect(() => {
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (browserInputCaptureTimerRef.current) {
        clearInterval(browserInputCaptureTimerRef.current);
      }
      stopMediaStream(browserInputStreamRef.current);
      socketRef.current?.close(1000, "dashboard unmounted");
      videoSocketRef.current?.close(1000, "dashboard unmounted");
      browserInputSocketRef.current?.close(1000, "dashboard unmounted");
    };
  }, []);

  function applyConfig() {
    sendUpdate();
  }

  function resetConfig() {
    const nextRecognizers = cloneRecognizers(DEFAULT_RECOGNIZER_CONFIG);
    const nextRecognitionConfig = cloneRecognitionConfig(DEFAULT_RECOGNITION_CONFIG);
    setRecognizers(nextRecognizers);
    setRecognitionConfig(nextRecognitionConfig);
    setGestureEvents(DEFAULT_GESTURE_EVENTS);
    setVisibleGestures(DEFAULT_VISIBLE_GESTURES);
    setGestureMapping(DEFAULT_GESTURE_MAPPING);
    sendUpdate(buildClientConfig(nextRecognizers, DEFAULT_GESTURE_EVENTS, DEFAULT_GESTURE_MAPPING, nextRecognitionConfig));
  }

  function setGestureDetection(gesture: string, enabled: boolean) {
    const next = { ...gestureEvents, [gesture]: enabled };
    setGestureEvents(next);
    sendUpdate(buildClientConfig(recognizers, next, gestureMapping, recognitionConfig));
  }

  function setAllGestureDetection(enabled: boolean, category?: GestureCategory) {
    const names = category ? gestureDefinitionsByCategory(category).map((gesture) => gesture.name) : GESTURES.map((gesture) => gesture.name);
    const next = { ...gestureEvents };
    for (const name of names) {
      next[name] = enabled;
    }
    setGestureEvents(next);
    sendUpdate(buildClientConfig(recognizers, next, gestureMapping, recognitionConfig));
  }

  function setAllVisibility(enabled: boolean, category?: GestureCategory) {
    const names = category ? gestureDefinitionsByCategory(category).map((gesture) => gesture.name) : GESTURES.map((gesture) => gesture.name);
    const next = { ...visibleGestures };
    for (const name of names) {
      next[name] = enabled;
    }
    setVisibleGestures(next);
  }

  function setRecognizerValue(recognizer: string, key: string, value: GesturePrimitive) {
    const next = {
      ...recognizers,
      [recognizer]: { ...recognizers[recognizer], [key]: value },
    };
    setRecognizers(next);
    sendUpdate(buildClientConfig(next, gestureEvents, gestureMapping, recognitionConfig));
  }

  function setAllRecognizers(enabled: boolean) {
    const next = cloneRecognizers(recognizers);
    for (const config of Object.values(next)) {
      config.enabled = enabled;
    }
    setRecognizers(next);
    sendUpdate(buildClientConfig(next, gestureEvents, gestureMapping, recognitionConfig));
  }

  function setFsmRecognition(enabled: boolean) {
    const next = cloneRecognitionConfig({
      ...recognitionConfig,
      fsm: { ...recognitionConfig.fsm, enabled },
    });
    setRecognitionConfig(next);
    sendUpdate(buildClientConfig(recognizers, gestureEvents, gestureMapping, next));
  }

  function setStaticPoseModelRecognition(enabled: boolean) {
    const next = cloneRecognitionConfig({
      ...recognitionConfig,
      models: {
        ...recognitionConfig.models,
        static_pose: { ...recognitionConfig.models.static_pose, enabled },
      },
    });
    setRecognitionConfig(next);
    sendUpdate(buildClientConfig(recognizers, gestureEvents, gestureMapping, next));
  }

  function setMotionModelRecognition(enabled: boolean) {
    const current = cloneRecognitionConfig(recognitionConfig);
    const next = cloneRecognitionConfig({
      ...current,
      models: {
        ...current.models,
        motion: { ...current.models.motion, enabled },
      },
    });
    setRecognitionConfig(next);
    sendUpdate(buildClientConfig(recognizers, gestureEvents, gestureMapping, next));
  }

  function setStaticPoseModelGroupRecognition(group: StaticPoseModelGestureGroup, enabled: boolean) {
    const nextStaticPose = {
      ...recognitionConfig.models.static_pose,
      [group]: { ...recognitionConfig.models.static_pose[group], enabled },
    };
    const next = cloneRecognitionConfig({
      ...recognitionConfig,
      models: {
        ...recognitionConfig.models,
        static_pose: nextStaticPose,
      },
    });
    setRecognitionConfig(next);
    sendUpdate(buildClientConfig(recognizers, gestureEvents, gestureMapping, next));
  }

  function setAllStaticPoseModelGroups(enabled: boolean) {
    const current = recognitionConfig.models.static_pose;
    const nextStaticPose = {
      ...current,
      pose: { ...current.pose, enabled },
      finger_count: { ...current.finger_count, enabled },
      finger_pointing: { ...current.finger_pointing, enabled },
      flat_hand: { ...current.flat_hand, enabled },
    };
    const next = cloneRecognitionConfig({
      ...recognitionConfig,
      models: {
        ...recognitionConfig.models,
        static_pose: nextStaticPose,
      },
    });
    setRecognitionConfig(next);
    sendUpdate(buildClientConfig(recognizers, gestureEvents, gestureMapping, next));
  }

  function setCommandMapping(gesture: string, command: string) {
    setGestureMapping((current) => ({ ...current, [gesture]: command }));
  }

  return (
    <main className="min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.22),transparent_34rem),radial-gradient(circle_at_bottom_right,rgba(168,85,247,0.18),transparent_28rem)]" />
      <div className="relative mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <Hero connectionState={connectionState} statusMessage={statusMessage} />

        <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_24rem]">
          <ConnectionPanel
            endpoint={endpoint}
            token={token}
            autoReconnect={autoReconnect}
            connectionState={connectionState}
            lastSentConfigAt={lastSentConfigAt}
            onEndpointChange={setEndpoint}
            onTokenChange={setToken}
            onAutoReconnectChange={setAutoReconnect}
            onConnect={connect}
            onDisconnect={disconnect}
            onApplyConfig={applyConfig}
          />
          <StatsPanel stats={stats} />
        </section>

        <BrowserCameraInputPanel
          endpoint={endpoint}
          tokenProvided={Boolean(token.trim())}
          devices={browserCameraDevices}
          selectedDeviceId={browserCameraDeviceId}
          connectionState={browserInputConnectionState}
          statusMessage={browserInputStatusMessage}
          sourceId={browserInputSourceId}
          frameCount={browserInputFrameCount}
          fps={clampedBrowserInputFps}
          videoRef={browserInputVideoRef}
          onDeviceChange={setBrowserCameraDeviceId}
          onFpsChange={(fps) => setBrowserInputFps(clampBrowserInputFps(fps))}
          onRefreshDevices={refreshBrowserCameraDevices}
          onStart={startBrowserCameraInput}
          onStop={stopBrowserCameraInput}
        />

        <VideoPreviewPanel
          endpoint={endpoint}
          tokenProvided={Boolean(token.trim())}
          sourceId={videoSourceId}
          connectionState={videoConnectionState}
          statusMessage={videoStatusMessage}
          frame={videoFrame}
          frameCount={videoFrameCount}
          onSourceIdChange={setVideoSourceId}
          onStart={connectVideo}
          onStop={disconnectVideo}
        />

        <ConfidenceTimelinePanel
          enabled={showConfidenceTimeline}
          events={events}
          visibleGestures={visibleGestures}
          now={chartNow}
          windowSeconds={chartWindowSeconds}
          eventLimit={clampedEventLimit}
          onEnabledChange={setShowConfidenceTimeline}
          onClearData={() => setEvents([])}
          onEventLimitChange={(limit) => setEventLimit(clampEventLimit(limit))}
          onWindowSecondsChange={(seconds) => setChartWindowSeconds(clampChartWindowSeconds(seconds))}
        />

        <section className="grid gap-6 xl:grid-cols-[minmax(0,27rem)_minmax(0,1fr)]">
          <div className="xl:col-span-2">
            <RecognitionSettingsPanel
              recognitionConfig={recognitionConfig}
              recognizers={recognizers}
              onFsmChange={setFsmRecognition}
              onRecognizerChange={setRecognizerValue}
              onToggleAllRecognizers={setAllRecognizers}
              onStaticPoseModelChange={setStaticPoseModelRecognition}
              onMotionModelChange={setMotionModelRecognition}
              onStaticPoseModelGroupChange={setStaticPoseModelGroupRecognition}
              onToggleAllStaticPoseModelGroups={setAllStaticPoseModelGroups}
            />
          </div>

          <div className="flex flex-col gap-6">
            <ControlLog messages={controlMessages} />
          </div>

          <div className="flex flex-col gap-6">
            <GestureControls
              gestureEvents={gestureEvents}
              visibleGestures={visibleGestures}
              gestureMapping={gestureMapping}
              eventCounts={stats.countsByGesture}
              onDetectionChange={setGestureDetection}
              onVisibilityChange={(gesture, visible) => setVisibleGestures((current) => ({ ...current, [gesture]: visible }))}
              onMappingChange={setCommandMapping}
              onEnableAll={setAllGestureDetection}
              onShowAll={setAllVisibility}
              onApplyConfig={applyConfig}
              onResetConfig={resetConfig}
            />
            <StreamPanel events={visibleEvents} totalEvents={events.length} />
          </div>
        </section>
      </div>
    </main>
  );
}

function ConfidenceTimelinePanel({
  enabled,
  events,
  visibleGestures,
  now,
  windowSeconds,
  eventLimit,
  onEnabledChange,
  onClearData,
  onEventLimitChange,
  onWindowSecondsChange,
}: {
  enabled: boolean;
  events: StreamEvent[];
  visibleGestures: GestureToggleMap;
  now: number;
  windowSeconds: number;
  eventLimit: number;
  onEnabledChange: (enabled: boolean) => void;
  onClearData: () => void;
  onEventLimitChange: (limit: number) => void;
  onWindowSecondsChange: (seconds: number) => void;
}) {
  const clampedWindowSeconds = clampChartWindowSeconds(windowSeconds);
  const clampedEventLimit = clampEventLimit(eventLimit);
  const series = useMemo(
    () => enabled ? buildTimelineSeries(events, visibleGestures, now, clampedWindowSeconds) : [],
    [clampedWindowSeconds, enabled, events, now, visibleGestures],
  );
  const eventsInWindow = series.reduce((total, item) => total + item.eventCount, 0);
  const gridLines = [0, 0.25, 0.5, 0.75, 1];
  const left = 58;
  const right = 984;
  const top = 32;
  const bottom = 352;

  return (
    <section className="rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur">
      <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">Confidence timeline</h2>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Toggle checked={enabled} onChange={onEnabledChange} label={enabled ? "Timeline on" : "Timeline off"} />
          <button className="button-danger" disabled={events.length === 0} onClick={onClearData}>Clear event data</button>
          {enabled && (
            <label className="flex min-w-52 items-center gap-3 rounded-2xl border border-white/10 bg-slate-950/60 px-3 py-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Seconds</span>
              <input
                className="w-full bg-transparent text-right text-sm font-bold text-white outline-none"
                type="number"
                min={MIN_CHART_WINDOW_SECONDS}
                max={MAX_CHART_WINDOW_SECONDS}
                step={5}
                value={clampedWindowSeconds}
                onChange={(event) => onWindowSecondsChange(Number(event.target.value))}
              />
            </label>
          )}
          <label className="flex min-w-52 items-center gap-3 rounded-2xl border border-white/10 bg-slate-950/60 px-3 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Stored events</span>
            <input
              className="w-full bg-transparent text-right text-sm font-bold text-white outline-none"
              type="number"
              min={MIN_EVENT_LIMIT}
              max={MAX_EVENT_LIMIT}
              step={10}
              value={clampedEventLimit}
              onChange={(event) => onEventLimitChange(Number(event.target.value))}
            />
          </label>
        </div>
      </div>

      {enabled ? (
        <>
          <div className="mb-3 flex flex-wrap gap-2 text-xs text-slate-400">
            <span className="rounded-full bg-white/10 px-3 py-1">{eventsInWindow} events in window</span>
            <span className="rounded-full bg-white/10 px-3 py-1">keeping latest {clampedEventLimit}</span>
            <span className="rounded-full bg-white/10 px-3 py-1">{series.length} visible event types</span>
            <span className="rounded-full bg-white/10 px-3 py-1">window ends {formatTime(now)}</span>
          </div>

          <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/70">
            <svg className="h-112 w-full" viewBox="0 0 1000 420" role="img" aria-label="Gesture confidence over time">
              <defs>
                <linearGradient id="timelineFill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="rgba(34,211,238,0.12)" />
                  <stop offset="100%" stopColor="rgba(34,211,238,0)" />
                </linearGradient>
              </defs>
              <rect x={left} y={top} width={right - left} height={bottom - top} fill="url(#timelineFill)" />
              {gridLines.map((confidence) => {
                const y = yForConfidence(confidence, top, bottom);
                return (
                  <g key={confidence}>
                    <line x1={left} x2={right} y1={y} y2={y} stroke="rgba(148,163,184,0.16)" strokeDasharray="4 8" />
                    <text x={left - 12} y={y + 4} textAnchor="end" className="fill-slate-500 text-[11px]">
                      {Math.round(confidence * 100)}%
                    </text>
                  </g>
                );
              })}
              {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
                const x = left + (right - left) * ratio;
                const secondsAgo = Math.round(clampedWindowSeconds * (1 - ratio));
                return (
                  <g key={ratio}>
                    <line x1={x} x2={x} y1={top} y2={bottom} stroke="rgba(148,163,184,0.1)" />
                    <text x={x} y={390} textAnchor="middle" className="fill-slate-500 text-[11px]">
                      {ratio === 1 ? "now" : `-${secondsAgo}s`}
                    </text>
                  </g>
                );
              })}
              <line x1={left} x2={right} y1={bottom} y2={bottom} stroke="rgba(148,163,184,0.28)" />
              <line x1={left} x2={left} y1={top} y2={bottom} stroke="rgba(148,163,184,0.28)" />

              {series.map((item) => (
                <g key={item.gesture}>
                  <path
                    d={item.path}
                    fill="none"
                    stroke={item.color}
                    strokeWidth={item.isTopSeries ? 3.2 : 2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    opacity={item.isTopSeries ? 1 : 0.78}
                  />
                  {item.peaks.map((peak) => (
                    <circle key={peak.id} cx={peak.x} cy={peak.y} r={item.isTopSeries ? 4 : 3} fill={item.color} opacity="0.9">
                      <title>{`${item.label}: ${formatPercent(peak.confidence)} at ${formatTime(peak.receivedAt)}`}</title>
                    </circle>
                  ))}
                </g>
              ))}
            </svg>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {series.length === 0 ? (
              <p className="rounded-2xl border border-dashed border-white/15 p-4 text-sm text-slate-500 sm:col-span-2 xl:col-span-3">
                No visible events in the selected time window yet.
              </p>
            ) : series.map((item) => (
              <div key={item.gesture} className="flex min-w-0 items-center gap-3 rounded-2xl border border-white/10 bg-slate-950/50 p-3">
                <span className="h-3 w-3 shrink-0 rounded-full" style={{ backgroundColor: item.color }} />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold text-white">{item.label}</p>
                  <p className="truncate text-xs text-slate-500">{item.gesture}</p>
                </div>
                <div className="text-right text-xs text-slate-400">
                  <p>{item.eventCount} hits</p>
                  <p>{formatPercent(item.latestConfidence)}</p>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="rounded-2xl border border-dashed border-white/15 bg-slate-950/40 p-8 text-center">
          <p className="text-sm font-semibold text-slate-300">Gesture history timeline chart is off.</p>
        </div>
      )}
    </section>
  );
}

function BrowserCameraInputPanel({
  endpoint,
  tokenProvided,
  devices,
  selectedDeviceId,
  connectionState,
  statusMessage,
  sourceId,
  frameCount,
  fps,
  videoRef,
  onDeviceChange,
  onFpsChange,
  onRefreshDevices,
  onStart,
  onStop,
}: {
  endpoint: string;
  tokenProvided: boolean;
  devices: BrowserCameraDevice[];
  selectedDeviceId: string;
  connectionState: ConnectionState;
  statusMessage: string;
  sourceId: string | null;
  frameCount: number;
  fps: number;
  videoRef: RefObject<HTMLVideoElement | null>;
  onDeviceChange: (deviceId: string) => void;
  onFpsChange: (fps: number) => void;
  onRefreshDevices: () => void;
  onStart: () => void;
  onStop: () => void;
}) {
  const connected = connectionState === "connected";
  const connecting = connectionState === "connecting";
  const active = connected || connecting || connectionState === "closing";

  return (
    <details className="group rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur">
      <summary className="flex cursor-pointer list-none items-start justify-between gap-4 [&::-webkit-details-marker]:hidden">
        <div>
          <h2 className="text-xl font-bold text-white">Browser camera input</h2>
          <p className="max-w-2xl text-sm text-slate-400">
            Select a local camera to stream to the server as a remote input feed, and use it for gesture event detection.
          </p>
        </div>
        <span className="inline-flex shrink-0 items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-bold uppercase tracking-wide text-cyan-100">
          <span className="group-open:hidden">Expand</span>
          <span className="hidden group-open:inline">Collapse</span>
          <span className="transition group-open:rotate-180">⌄</span>
        </span>
      </summary>

      <div className="mt-5 border-t border-white/10 pt-5">
        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="inline-flex w-fit items-center gap-2 rounded-full border border-white/10 bg-slate-950/60 px-3 py-1 text-sm text-slate-300">
            <StatusDot state={connectionState} />
            <span>{labelForState(connectionState)}</span>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button className="button-secondary" disabled={active} onClick={onRefreshDevices}>Refresh cameras</button>
            {active ? (
              <button className="button-danger" onClick={onStop}>Stop input</button>
            ) : (
              <button className="button-primary" onClick={onStart}>Start input</button>
            )}
          </div>
        </div>

        <div className="mb-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_12rem] lg:items-end">
          <label className="space-y-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Browser camera</span>
            <select
              className="input"
              value={selectedDeviceId}
              disabled={active}
              onChange={(event) => onDeviceChange(event.target.value)}
            >
              <option value="">Default camera</option>
              {devices.map((device, index) => (
                <option key={`${device.deviceId || "default"}-${index}`} value={device.deviceId}>
                  {device.label}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Input FPS</span>
            <input
              className="input"
              type="number"
              min={MIN_BROWSER_INPUT_FPS}
              max={MAX_BROWSER_INPUT_FPS}
              step={1}
              value={fps}
              onChange={(event) => onFpsChange(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="mb-3 flex flex-wrap gap-2 text-xs text-slate-400">
          <span className="rounded-full bg-white/10 px-3 py-1">endpoint {endpoint}</span>
          <span className="rounded-full bg-white/10 px-3 py-1">JWT {tokenProvided ? "provided" : "not provided"}</span>
          <span className="rounded-full bg-white/10 px-3 py-1">source {sourceId ?? "not generated"}</span>
          <span className="rounded-full bg-white/10 px-3 py-1">frames sent {frameCount}</span>
          <span className="rounded-full bg-white/10 px-3 py-1">{statusMessage}</span>
        </div>

        <div className="overflow-hidden rounded-2xl border border-white/10 bg-black">
          {active ? (
            <video
              ref={videoRef}
              className="aspect-video w-full bg-black object-contain"
              muted
              playsInline
              autoPlay
            />
          ) : (
            <div className="flex aspect-video flex-col items-center justify-center gap-2 p-8 text-center text-slate-400">
              <p className="text-sm font-semibold text-slate-300">Browser camera input is off.</p>
            </div>
          )}
        </div>
      </div>
    </details>
  );
}

function VideoPreviewPanel({
  endpoint,
  tokenProvided,
  sourceId,
  connectionState,
  statusMessage,
  frame,
  frameCount,
  onSourceIdChange,
  onStart,
  onStop,
}: {
  endpoint: string;
  tokenProvided: boolean;
  sourceId: string;
  connectionState: ConnectionState;
  statusMessage: string;
  frame: VideoPreviewFrame | null;
  frameCount: number;
  onSourceIdChange: (sourceId: string) => void;
  onStart: () => void;
  onStop: () => void;
}) {
  const connected = connectionState === "connected";
  const connecting = connectionState === "connecting";
  const dimensions = videoDimensions(frame);
  const imageSrc = frame?.frame ? `data:${frame.frame.mimeType};base64,${frame.frame.data}` : "";

  return (
    <details className="group rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur">
      <summary className="flex cursor-pointer list-none items-start justify-between gap-4 [&::-webkit-details-marker]:hidden">
        <div>
          <h2 className="text-xl font-bold text-white">Camera video preview</h2>
        </div>
        <span className="inline-flex shrink-0 items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-bold uppercase tracking-wide text-cyan-100">
          <span className="group-open:hidden">Expand</span>
          <span className="hidden group-open:inline">Collapse</span>
          <span className="transition group-open:rotate-180">⌄</span>
        </span>
      </summary>
      <div className="mt-5 border-t border-white/10 pt-5">
        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap items-center gap-3">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-950/60 px-3 py-1 text-sm text-slate-300">
              <StatusDot state={connectionState} />
              <span>{labelForState(connectionState)}</span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {connected || connecting ? (
              <button className="button-danger" onClick={onStop}>Stop video</button>
            ) : (
              <button className="button-primary" onClick={onStart}>Start video</button>
            )}
          </div>
        </div>

      <div className="mb-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Video source</span>
          <input
            className="input"
            value={sourceId}
            onChange={(event) => onSourceIdChange(event.target.value)}
            placeholder={DEFAULT_VIDEO_SOURCE_ID}
          />
        </label>
        <button className="button-secondary" disabled={!connected} onClick={onStart}>Apply source</button>
      </div>

      <div className="mb-3 flex flex-wrap gap-2 text-xs text-slate-400">
        <span className="rounded-full bg-white/10 px-3 py-1">endpoint {endpoint}</span>
        <span className="rounded-full bg-white/10 px-3 py-1">JWT {tokenProvided ? "provided" : "not provided"}</span>
        <span className="rounded-full bg-white/10 px-3 py-1">frames {frameCount}</span>
        <span className="rounded-full bg-white/10 px-3 py-1">hands {frame?.hands.length ?? 0}</span>
        <span className="rounded-full bg-white/10 px-3 py-1">{statusMessage}</span>
      </div>

        <div className="overflow-hidden rounded-2xl border border-white/10 bg-black">
          {imageSrc ? (
            <div className="relative aspect-video w-full bg-black">
              <img
                src={imageSrc}
                alt={`Video preview from ${frame?.sourceLabel || frame?.sourceId || sourceId}`}
                className="h-full w-full object-contain"
              />
              <HandOverlay hands={frame?.hands ?? []} width={dimensions.width} height={dimensions.height} />
            </div>
          ) : (
            <div className="flex aspect-video flex-col items-center justify-center gap-2 p-8 text-center text-slate-400">
              <p className="text-sm font-semibold text-slate-300">
                {connected ? "Waiting for video frames." : "Video preview is off."}
              </p>
            </div>
          )}
        </div>
      </div>
    </details>
  );
}

function HandOverlay({ hands, width, height }: { hands: VideoHand[]; width: number; height: number }) {
  if (hands.length === 0 || width <= 0 || height <= 0) {
    return null;
  }

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      {hands.map((hand, handIndex) => {
        const points = hand.landmarks.map((point) => ({ x: point.x * width, y: point.y * height }));
        return (
          <g key={`${hand.handedness ?? "hand"}-${handIndex}`}>
            {HAND_CONNECTIONS.map(([start, end]) => {
              const startPoint = points[start];
              const endPoint = points[end];
              if (!startPoint || !endPoint) {
                return null;
              }
              return (
                <line
                  key={`${start}-${end}`}
                  x1={startPoint.x}
                  y1={startPoint.y}
                  x2={endPoint.x}
                  y2={endPoint.y}
                  stroke="rgb(255,180,0)"
                  strokeLinecap="round"
                  strokeWidth={Math.max(width, height) * 0.003}
                />
              );
            })}
            {points.map((point, pointIndex) => (
              <circle
                key={pointIndex}
                cx={point.x}
                cy={point.y}
                r={Math.max(width, height) * 0.006}
                fill="rgb(0,255,0)"
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

function Hero({ connectionState, statusMessage }: { connectionState: ConnectionState; statusMessage: string }) {
  return (
    <header className="rounded-4xl border border-white/10 bg-white/[0.07] p-6 shadow-2xl shadow-cyan-950/40 backdrop-blur md:p-8">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl">
          <p className="mb-3 inline-flex rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-cyan-200">
            HLGAM realtime client
          </p>
          <h1 className="text-2xl font-black tracking-tight text-white md:text-4xl">
            Gesture stream engine dev control panel
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
            Connect to a running gesture engine, push per-client recognition settings over WebSocket, and inspect detected gesture events as they arrive.
          </p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4 text-sm text-slate-300">
          <div className="mb-2 flex items-center gap-2">
            <StatusDot state={connectionState} />
            <span className="font-semibold text-white">{labelForState(connectionState)}</span>
          </div>
          <p className="max-w-sm text-slate-400">{statusMessage}</p>
        </div>
      </div>
    </header>
  );
}

function ConnectionPanel(props: {
  endpoint: string;
  token: string;
  autoReconnect: boolean;
  connectionState: ConnectionState;
  lastSentConfigAt: number | null;
  onEndpointChange: (value: string) => void;
  onTokenChange: (value: string) => void;
  onAutoReconnectChange: (value: boolean) => void;
  onConnect: () => void;
  onDisconnect: () => void;
  onApplyConfig: () => void;
}) {
  const connected = props.connectionState === "connected";
  const connecting = props.connectionState === "connecting";

  return (
    <section className="rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-white">Connection</h2>
        </div>
        <StatusDot state={props.connectionState} />
      </div>

      <div className="grid gap-4 md:grid-cols-[1fr_1fr]">
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">WebSocket URL</span>
          <input
            value={props.endpoint}
            onChange={(event) => props.onEndpointChange(event.target.value)}
            className="input"
            placeholder="ws://127.0.0.1:8765"
          />
        </label>
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">JWT token</span>
          <input
            value={props.token}
            onChange={(event) => props.onTokenChange(event.target.value)}
            className="input"
            placeholder="Paste token if JWT is required"
            type="password"
          />
        </label>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-4 text-sm text-slate-400">
        <div className="flex items-center gap-4">
          <Toggle checked={props.autoReconnect} onChange={props.onAutoReconnectChange} label="Auto reconnect" />
          <span>Last config push: {props.lastSentConfigAt ? formatTime(props.lastSentConfigAt) : "not sent yet"}</span>
        </div>
        <div className="flex items-center gap-2">
          {connected ? (
            <button className="button-danger" onClick={props.onDisconnect}>Disconnect</button>
          ) : (
            <button className="button-primary" disabled={connecting} onClick={props.onConnect}>{connecting ? "Connecting" : "Connect"}</button>
          )}
          <button className="button-secondary" onClick={props.onApplyConfig}>Apply config</button>
        </div>
      </div>
    </section>
  );
}

function StatsPanel({ stats }: { stats: ReturnType<typeof buildStats> }) {
  const cards = [
    ["Total", stats.total.toString()],
    ["Visible", stats.visible.toString()],
    ["Enabled", stats.enabled.toString()],
    ["Top event", stats.topGesture || "—"],
  ];
  return (
    <section className="grid grid-cols-2 gap-3 rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur">
      {cards.map(([label, value]) => (
        <div key={label} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
          <p className="mt-2 truncate text-2xl font-black text-white">{value}</p>
        </div>
      ))}
    </section>
  );
}

function RecognitionSettingsPanel({
  recognitionConfig,
  recognizers,
  onFsmChange,
  onRecognizerChange,
  onToggleAllRecognizers,
  onStaticPoseModelChange,
  onMotionModelChange,
  onStaticPoseModelGroupChange,
  onToggleAllStaticPoseModelGroups,
}: {
  recognitionConfig: RecognitionPipelineConfig;
  recognizers: RecognizerConfigMap;
  onFsmChange: (enabled: boolean) => void;
  onRecognizerChange: (recognizer: string, key: string, value: GesturePrimitive) => void;
  onToggleAllRecognizers: (enabled: boolean) => void;
  onStaticPoseModelChange: (enabled: boolean) => void;
  onMotionModelChange: (enabled: boolean) => void;
  onStaticPoseModelGroupChange: (group: StaticPoseModelGestureGroup, enabled: boolean) => void;
  onToggleAllStaticPoseModelGroups: (enabled: boolean) => void;
}) {
  const config = cloneRecognitionConfig(recognitionConfig);
  const fsmEnabled = config.fsm.enabled;
  const staticPoseModel = config.models.static_pose;
  const staticPoseModelEnabled = staticPoseModel.enabled;
  const motionModel = config.models.motion;
  const motionModelEnabled = motionModel.enabled;
  const currentRecognizers = cloneRecognizers(recognizers);

  return (
    <details className="group rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur" open>
      <summary className="flex cursor-pointer list-none items-start justify-between gap-4 [&::-webkit-details-marker]:hidden">
        <div>
          <h2 className="text-xl font-bold text-white">Recognition settings</h2>
          <p className="text-sm text-slate-400">Configure FSM and learned model prediction pipelines for this client.</p>
        </div>
        <span className="inline-flex shrink-0 items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-bold uppercase tracking-wide text-cyan-100">
          <span className="group-open:hidden">Expand</span>
          <span className="hidden group-open:inline">Collapse</span>
          <span className="transition group-open:rotate-180">⌄</span>
        </span>
      </summary>

      <div className="mt-5 space-y-5 border-t border-white/10 pt-5">
        <section className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="font-semibold text-white">FSM prediction</p>
              <p className="mt-1 text-sm text-slate-400">Runs deterministic rule/FSM recognizer groups and keeps their thresholds editable.</p>
            </div>
            <Toggle checked={fsmEnabled} onChange={onFsmChange} label={fsmEnabled ? "FSM on" : "FSM off"} />
          </div>

          <div className="mt-4 flex flex-wrap gap-2 border-t border-white/10 pt-4">
            <button className="button-secondary" onClick={() => onToggleAllRecognizers(true)}>Enable all FSM groups</button>
            <button className="button-secondary" onClick={() => onToggleAllRecognizers(false)}>Disable all FSM groups</button>
          </div>

          <div className="mt-4 grid gap-3">
            {Object.entries(currentRecognizers).map(([name, recognizerConfig]) => (
              <details key={name} className="group rounded-2xl border border-white/10 bg-black/20 p-4" open={false}>
                <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="font-semibold capitalize text-white">{name.replace("_", " ")}</p>
                      <code className="rounded-full bg-cyan-400/10 px-2 py-0.5 text-xs text-cyan-200">{name}</code>
                    </div>
                    <p className="mt-1 text-xs text-slate-500">{Object.keys(recognizerConfig).length - 1} editable parameters</p>
                  </div>
                  <Toggle
                    checked={Boolean(recognizerConfig.enabled)}
                    onChange={(checked) => onRecognizerChange(name, "enabled", checked)}
                    label={Boolean(recognizerConfig.enabled) ? "Enabled" : "Disabled"}
                  />
                </summary>
                <div className="mt-4 grid gap-3 border-t border-white/10 pt-4 sm:grid-cols-2">
                  {Object.entries(recognizerConfig).filter(([key]) => key !== "enabled").map(([key, value]) => (
                    <label key={key} className="space-y-1">
                      <span className="text-xs font-medium text-slate-400">{key}</span>
                      <input
                        className="input h-10"
                        type="number"
                        step={Number.isInteger(value) ? 1 : 0.001}
                        min={0}
                        value={String(value)}
                        onChange={(event) => onRecognizerChange(name, key, coerceRecognizerValue(value, event.target.value))}
                      />
                    </label>
                  ))}
                </div>
              </details>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="font-semibold text-white">Static pose model prediction</p>
              <p className="mt-1 text-sm text-slate-400">Runs learned static-pose predictions from the configured server-side checkpoint.</p>
            </div>
            <Toggle
              checked={staticPoseModelEnabled}
              onChange={onStaticPoseModelChange}
              label={staticPoseModelEnabled ? "Model on" : "Model off"}
            />
          </div>

          <div className="mt-4 flex flex-wrap gap-2 border-t border-white/10 pt-4">
            <button className="button-secondary" onClick={() => onToggleAllStaticPoseModelGroups(true)}>Enable all model groups</button>
            <button className="button-secondary" onClick={() => onToggleAllStaticPoseModelGroups(false)}>Disable all model groups</button>
          </div>

          <div className="mt-4 grid gap-2">
            {STATIC_POSE_MODEL_GESTURE_GROUPS.map((group) => {
              const enabled = staticPoseModel[group.name].enabled;
              return (
                <div key={group.name} className="rounded-2xl border border-white/10 bg-black/20 p-3">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="font-semibold text-white">{group.label}</p>
                        <code className="rounded-full bg-cyan-400/10 px-2 py-0.5 text-xs text-cyan-200">{group.name}</code>
                      </div>
                      <p className="mt-1 text-sm text-slate-400">{group.description}</p>
                    </div>
                    <Toggle
                      checked={enabled}
                      onChange={(checked) => onStaticPoseModelGroupChange(group.name, checked)}
                      label={enabled ? "Detect" : "Muted"}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="font-semibold text-white">Motion model prediction</p>
              <p className="mt-1 text-sm text-slate-400">
                Runs learned temporal motion predictions from the configured server-side checkpoint.
              </p>
            </div>
            <Toggle
              checked={motionModelEnabled}
              onChange={onMotionModelChange}
              label={motionModelEnabled ? "Model on" : "Model off"}
            />
          </div>

          <div className="mt-4 rounded-2xl border border-cyan-300/10 bg-cyan-300/5 p-3 text-sm text-slate-400">
            Toggle this on to let the learned motion model emit gestures independently of the FSM motion recognizers.
            Checkpoint paths, thresholds, and runtime artifacts stay server-controlled.
          </div>
        </section>
      </div>
    </details>
  );
}

function GestureControls(props: {
  gestureEvents: GestureToggleMap;
  visibleGestures: GestureToggleMap;
  gestureMapping: GestureCommandMap;
  eventCounts: Record<string, number>;
  onDetectionChange: (gesture: string, enabled: boolean) => void;
  onVisibilityChange: (gesture: string, visible: boolean) => void;
  onMappingChange: (gesture: string, command: string) => void;
  onEnableAll: (enabled: boolean, category?: GestureCategory) => void;
  onShowAll: (enabled: boolean, category?: GestureCategory) => void;
  onApplyConfig: () => void;
  onResetConfig: () => void;
}) {
  return (
    <details className="group rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur">
      <summary className="flex cursor-pointer list-none items-start justify-between gap-4 [&::-webkit-details-marker]:hidden">
        <div>
          <h2 className="text-xl font-bold text-white">Gesture event controls</h2>
          <p className="max-w-2xl text-sm text-slate-400">Click to expand per-event detection, visibility, and command mapping controls.</p>
        </div>
        <span className="inline-flex shrink-0 items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-bold uppercase tracking-wide text-cyan-100">
          <span className="group-open:hidden">Expand</span>
          <span className="hidden group-open:inline">Collapse</span>
          <span className="transition group-open:rotate-180">⌄</span>
        </span>
      </summary>

      <div className="mt-5 border-t border-white/10 pt-5">
        <p className="mb-4 text-sm text-slate-400">Detect controls are sent to the engine as `gesture_events`. Show controls are local UI filters.</p>
        <div className="flex flex-wrap gap-2">
          <button className="button-secondary" onClick={() => props.onEnableAll(true)}>Detect all</button>
          <button className="button-secondary" onClick={() => props.onEnableAll(false)}>Detect none</button>
          <button className="button-secondary" onClick={() => props.onShowAll(true)}>Show all</button>
          <button className="button-secondary" onClick={props.onApplyConfig}>Apply</button>
          <button className="button-danger" onClick={props.onResetConfig}>Reset</button>
        </div>
      </div>

      <div className="mt-5 space-y-5">
        {(Object.keys(GESTURE_CATEGORIES) as GestureCategory[]).map((category) => {
          const definitions = gestureDefinitionsByCategory(category);
          return (
            <details key={category} className="rounded-2xl border border-white/10 bg-slate-950/50 p-4" open={false}>
              <summary className="cursor-pointer list-none">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <h3 className="font-bold text-white">{GESTURE_CATEGORIES[category].label}</h3>
                    <p className="text-sm text-slate-400">{GESTURE_CATEGORIES[category].description}</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button className="mini-button" onClick={(event) => { event.preventDefault(); props.onEnableAll(true, category); }}>Detect group</button>
                    <button className="mini-button" onClick={(event) => { event.preventDefault(); props.onEnableAll(false, category); }}>Mute group</button>
                    <button className="mini-button" onClick={(event) => { event.preventDefault(); props.onShowAll(true, category); }}>Show group</button>
                  </div>
                </div>
              </summary>
              <div className="mt-4 grid gap-3">
                {definitions.map((gesture) => (
                  <div key={gesture.name} className="grid gap-3 rounded-2xl border border-white/10 bg-black/20 p-3 lg:grid-cols-[minmax(0,1fr)_auto_auto_minmax(9rem,14rem)] lg:items-center">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="font-semibold text-white">{gesture.label}</p>
                        <code className="rounded-full bg-cyan-400/10 px-2 py-0.5 text-xs text-cyan-200">{gesture.name}</code>
                        <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-300">{props.eventCounts[gesture.name] ?? 0}</span>
                      </div>
                      <p className="mt-1 text-sm text-slate-400">{gesture.description}</p>
                    </div>
                    <Toggle checked={props.gestureEvents[gesture.name] ?? true} onChange={(checked) => props.onDetectionChange(gesture.name, checked)} label="Detect" />
                    <Toggle checked={props.visibleGestures[gesture.name] ?? true} onChange={(checked) => props.onVisibilityChange(gesture.name, checked)} label="Show" />
                    <input
                      className="input h-10"
                      value={props.gestureMapping[gesture.name] ?? ""}
                      onChange={(event) => props.onMappingChange(gesture.name, event.target.value)}
                      onBlur={props.onApplyConfig}
                      placeholder="Command mapping"
                    />
                  </div>
                ))}
              </div>
            </details>
          );
        })}
      </div>
    </details>
  );
}

function StreamPanel({ events, totalEvents }: { events: StreamEvent[]; totalEvents: number }) {
  const latest = events[0];
  return (
    <details className="group rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur">
      <summary className="flex cursor-pointer list-none items-start justify-between gap-4 [&::-webkit-details-marker]:hidden">
        <div>
          <h2 className="text-xl font-bold text-white">Live gesture stream</h2>
          <p className="text-sm text-slate-400">Showing {events.length} visible events from {totalEvents} received.</p>
        </div>
        <span className="inline-flex shrink-0 items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-bold uppercase tracking-wide text-cyan-100">
          <span className="group-open:hidden">Expand</span>
          <span className="hidden group-open:inline">Collapse</span>
          <span className="transition group-open:rotate-180">⌄</span>
        </span>
      </summary>

      <div className="mt-5 border-t border-white/10 pt-5">

        {latest && (
          <div className="mb-4 rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-200">Latest visible gesture</p>
            <div className="mt-2 flex flex-wrap items-end gap-3">
              <h3 className="text-2xl font-black text-white">{latest.gesture}</h3>
              <span className="text-sm text-slate-300">{latest.phase}</span>
              <span className="text-sm text-slate-300">confidence {formatPercent(latest.confidence)}</span>
              {latest.command && <span className="rounded-full bg-emerald-400/10 px-2 py-1 text-sm text-emerald-200">{latest.command}</span>}
            </div>
          </div>
        )}

        <div className="max-h-168 space-y-3 overflow-y-auto pr-1">
          {events.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-white/15 p-8 text-center text-slate-400">
              No visible gestures yet. Connect to the engine and perform a configured gesture.
            </div>
          ) : events.map((event) => <EventCard key={event.id} event={event} />)}
        </div>
      </div>
    </details>
  );
}

function EventCard({ event }: { event: StreamEvent }) {
  return (
    <article className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-bold text-white">{event.gesture}</h3>
            <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-300">{event.phase}</span>
            <span className="rounded-full bg-cyan-400/10 px-2 py-0.5 text-xs text-cyan-200">{formatPercent(event.confidence)}</span>
          </div>
          <p className="mt-1 text-sm text-slate-400">Received {formatTime(event.receivedAt)} · engine timestamp {formatNumber(event.timestamp)}</p>
        </div>
        {event.command && <span className="rounded-full bg-emerald-400/10 px-3 py-1 text-sm font-semibold text-emerald-200">{event.command}</span>}
      </div>
      <details className="mt-3">
        <summary className="cursor-pointer text-sm text-slate-400 hover:text-slate-200">JSON payload</summary>
        <pre className="mt-2 overflow-x-auto rounded-xl bg-black/40 p-3 text-xs text-slate-300">{JSON.stringify(stripClientFields(event), null, 2)}</pre>
      </details>
    </article>
  );
}

function ControlLog({ messages }: { messages: ControlMessage[] }) {
  return (
    <details className="group rounded-3xl border border-white/10 bg-white/6 p-5 shadow-xl shadow-slate-950/30 backdrop-blur">
      <summary className="flex cursor-pointer list-none items-start justify-between gap-4 [&::-webkit-details-marker]:hidden">
        <div>
          <h2 className="text-xl font-bold text-white">Control messages</h2>
          <p className="text-sm text-slate-400">Config acks and broker errors.</p>
        </div>
        <span className="inline-flex shrink-0 items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-bold uppercase tracking-wide text-cyan-100">
          <span className="group-open:hidden">Expand</span>
          <span className="hidden group-open:inline">Collapse</span>
          <span className="transition group-open:rotate-180">⌄</span>
        </span>
      </summary>
      <div className="mt-5 max-h-72 space-y-2 overflow-y-auto border-t border-white/10 pt-5">
        {messages.length === 0 ? (
          <p className="rounded-2xl border border-dashed border-white/15 p-4 text-sm text-slate-500">No control messages yet.</p>
        ) : messages.map((message) => (
          <details key={message.id} className="rounded-2xl border border-white/10 bg-slate-950/60 p-3">
            <summary className="cursor-pointer text-sm font-semibold text-white">{message.type} · {formatTime(message.receivedAt)}</summary>
            <pre className="mt-2 overflow-x-auto rounded-xl bg-black/40 p-3 text-xs text-slate-300">{JSON.stringify(message.payload, null, 2)}</pre>
          </details>
        ))}
      </div>
    </details>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (checked: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      aria-pressed={checked}
      onClick={() => onChange(!checked)}
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-semibold transition ${
        checked
          ? "border-emerald-300/30 bg-emerald-400/15 text-emerald-100"
          : "border-white/10 bg-white/5 text-slate-400"
      }`}
    >
      <span className={`h-2.5 w-2.5 rounded-full ${checked ? "bg-emerald-300" : "bg-slate-600"}`} />
      {label}
    </button>
  );
}

function StatusDot({ state }: { state: ConnectionState }) {
  const color = state === "connected" ? "bg-emerald-300" : state === "connecting" ? "bg-amber-300" : state === "error" ? "bg-rose-400" : "bg-slate-500";
  return <span className={`h-3 w-3 rounded-full ${color} shadow-lg`} />;
}

function usePersistentState<T>(key: string, fallback: T): [T, (value: T | ((current: T) => T)) => void] {
  const [state, setState] = useState<T>(fallback);
  const loadedRef = useRef(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(key);
      if (stored) {
        setState(mergeFallback(fallback, JSON.parse(stored)) as T);
      }
    } catch {
      // Ignore malformed local storage and keep defaults.
    } finally {
      loadedRef.current = true;
    }
  }, [fallback, key]);

  useEffect(() => {
    if (!loadedRef.current) {
      return;
    }
    window.localStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);

  return [state, setState];
}

function mergeFallback(fallback: unknown, stored: unknown): unknown {
  if (Array.isArray(fallback) || Array.isArray(stored)) {
    return stored ?? fallback;
  }
  if (isPlainObject(fallback) && isPlainObject(stored)) {
    const merged: Record<string, unknown> = { ...fallback };
    for (const [key, value] of Object.entries(stored)) {
      merged[key] = key in merged ? mergeFallback(merged[key], value) : value;
    }
    return merged;
  }
  return stored ?? fallback;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parseSocketMessage(data: unknown): Record<string, unknown> | null {
  if (typeof data !== "string") {
    return null;
  }
  try {
    const parsed = JSON.parse(data);
    return isPlainObject(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function isGestureEvent(payload: Record<string, unknown>): payload is GestureEventPayload {
  return typeof payload.gesture === "string" && typeof payload.phase === "string" && typeof payload.timestamp === "number";
}

function buildVideoOutputRequest(sourceId: string, token: string): Record<string, unknown> {
  const request: Record<string, unknown> = {
    type: "start_video_output",
    source_id: sourceId.trim() || DEFAULT_VIDEO_SOURCE_ID,
    include_frame: true,
    include_landmarks: true,
    include_features: false,
    jpeg_quality: VIDEO_JPEG_QUALITY,
  };
  const trimmedToken = token.trim();
  if (trimmedToken) {
    request.token = trimmedToken;
  }
  return request;
}

function buildBrowserVideoInputRequest(sourceId: string, label: string, token: string, deviceId: string): Record<string, unknown> {
  const request: Record<string, unknown> = {
    type: "start_video_input",
    source_id: sourceId,
    label,
    metadata: {
      client: "web_dashboard",
      device_id: deviceId || "default",
      source_prefix: BROWSER_INPUT_SOURCE_PREFIX,
    },
  };
  const trimmedToken = token.trim();
  if (trimmedToken) {
    request.token = trimmedToken;
  }
  return request;
}

function buildBrowserVideoFrameMessage(sourceId: string, data: string, index: number): Record<string, unknown> {
  return {
    type: "video_frame",
    source_id: sourceId,
    frame: {
      encoding: "jpeg_base64",
      data,
      timestamp: Date.now() / 1000,
      index,
    },
  };
}

function browserVideoConstraints(deviceId: string, fps: number): MediaTrackConstraints {
  const constraints: MediaTrackConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: clampBrowserInputFps(fps) },
  };
  if (deviceId) {
    constraints.deviceId = { exact: deviceId };
  }
  return constraints;
}

function captureBrowserVideoFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement): string | null {
  if (video.videoWidth <= 0 || video.videoHeight <= 0) {
    return null;
  }
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return dataUrlToBase64(canvas.toDataURL("image/jpeg", BROWSER_INPUT_JPEG_QUALITY));
}

function dataUrlToBase64(dataUrl: string) {
  const commaIndex = dataUrl.indexOf(",");
  return commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;
}

function stopMediaStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function createBrowserInputSourceId() {
  return `${BROWSER_INPUT_SOURCE_PREFIX}${randomSourceSuffix(14)}`;
}

function randomSourceSuffix(length: number) {
  const alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
  const crypto = globalThis.crypto;
  if (crypto?.getRandomValues) {
    const values = new Uint8Array(length);
    crypto.getRandomValues(values);
    return Array.from(values, (value) => alphabet[value % alphabet.length]).join("");
  }
  let suffix = "";
  while (suffix.length < length) {
    suffix += Math.random().toString(36).slice(2);
  }
  return suffix.slice(0, length);
}

function labelForBrowserCamera(devices: BrowserCameraDevice[], deviceId: string) {
  if (!deviceId) {
    return "default browser camera";
  }
  return devices.find((device) => device.deviceId === deviceId)?.label ?? "selected browser camera";
}

function normalizeVideoPreviewFrame(payload: Record<string, unknown>): VideoPreviewFrame | null {
  if (payload.type !== "video_frame") {
    return null;
  }

  const source = isPlainObject(payload.source) ? payload.source : {};
  const sourceId = stringOrNull(payload.source_id) ?? stringOrNull(source.id) ?? "unknown_source";
  const sourceLabel = stringOrNull(source.label) ?? sourceId;
  const framePayload = isPlainObject(payload.frame) ? payload.frame : null;
  const frame = normalizeVideoImageFrame(framePayload);
  const hands = Array.isArray(payload.hands)
    ? payload.hands.map(normalizeVideoHand).filter((hand): hand is VideoHand => hand !== null)
    : [];

  return {
    sourceId,
    sourceLabel,
    receivedAt: Date.now(),
    timestamp: numberOrNull(payload.timestamp),
    frame,
    hands,
    featureCount: Array.isArray(payload.features) ? payload.features.length : 0,
  };
}

function normalizeVideoImageFrame(payload: Record<string, unknown> | null): VideoImageFrame | null {
  if (!payload) {
    return null;
  }
  const data = stringOrNull(payload.data);
  if (!data) {
    return null;
  }
  const encoding = stringOrNull(payload.encoding);
  return {
    data,
    mimeType: stringOrNull(payload.mime_type) ?? mimeTypeForEncoding(encoding),
    encoding,
    width: numberOrNull(payload.width),
    height: numberOrNull(payload.height),
    index: numberOrNull(payload.index),
    timestamp: numberOrNull(payload.timestamp),
  };
}

function normalizeVideoHand(payload: unknown): VideoHand | null {
  const handPayload = isPlainObject(payload) ? payload : null;
  const landmarksPayload = handPayload ? handPayload.landmarks : payload;
  if (!Array.isArray(landmarksPayload)) {
    return null;
  }
  const landmarks = landmarksPayload.map(normalizeVideoPoint).filter((point): point is VideoPoint => point !== null);
  if (landmarks.length === 0) {
    return null;
  }
  return {
    landmarks,
    handedness: handPayload ? stringOrNull(handPayload.handedness) : null,
    score: handPayload ? numberOrNull(handPayload.score) : null,
    imageSize: handPayload ? normalizeImageSize(handPayload.image_size) : null,
  };
}

function normalizeVideoPoint(payload: unknown): VideoPoint | null {
  if (isPlainObject(payload)) {
    const x = numberOrNull(payload.x);
    const y = numberOrNull(payload.y);
    if (x === null || y === null) {
      return null;
    }
    return { x, y, z: numberOrNull(payload.z) ?? 0 };
  }
  if (Array.isArray(payload) && payload.length >= 2) {
    const x = numberOrNull(payload[0]);
    const y = numberOrNull(payload[1]);
    if (x === null || y === null) {
      return null;
    }
    return { x, y, z: numberOrNull(payload[2]) ?? 0 };
  }
  return null;
}

function normalizeImageSize(payload: unknown): [number, number] | null {
  if (!Array.isArray(payload) || payload.length !== 2) {
    return null;
  }
  const width = numberOrNull(payload[0]);
  const height = numberOrNull(payload[1]);
  return width !== null && height !== null ? [width, height] : null;
}

function videoDimensions(frame: VideoPreviewFrame | null): { width: number; height: number } {
  const frameWidth = frame?.frame?.width;
  const frameHeight = frame?.frame?.height;
  if (frameWidth && frameHeight) {
    return { width: frameWidth, height: frameHeight };
  }
  const handImageSize = frame?.hands.find((hand) => hand.imageSize)?.imageSize;
  if (handImageSize) {
    return { width: handImageSize[0], height: handImageSize[1] };
  }
  return { width: 1280, height: 720 };
}

function labelForVideoControl(type: string, payload: Record<string, unknown>): string {
  if (type === "video_output_started") {
    const source = isPlainObject(payload.source) ? payload.source : {};
    return `Streaming ${stringOrNull(source.label) ?? stringOrNull(payload.source_id) ?? DEFAULT_VIDEO_SOURCE_ID}`;
  }
  if (type === "source_selected") {
    return `Selected source ${stringOrNull(payload.source_id) ?? DEFAULT_VIDEO_SOURCE_ID}`;
  }
  if (type === "video_sources") {
    const count = Array.isArray(payload.sources) ? payload.sources.length : 0;
    return `${count} video sources available`;
  }
  return `Video control: ${type}`;
}

function labelForBrowserInputControl(type: string, payload: Record<string, unknown>): string {
  if (type === "video_input_started") {
    const source = isPlainObject(payload.source) ? payload.source : {};
    return `Registered ${stringOrNull(source.label) ?? stringOrNull(payload.source_id) ?? "browser camera input"}`;
  }
  if (type === "video_input_stopped") {
    return "Browser camera input stopped";
  }
  if (type === "video_sources_updated") {
    const count = Array.isArray(payload.sources) ? payload.sources.length : 0;
    return `${count} video sources available`;
  }
  return `Browser input control: ${type}`;
}

function mimeTypeForEncoding(encoding: string | null): string {
  return encoding === "png_base64" ? "image/png" : "image/jpeg";
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function buildStats(events: StreamEvent[], visibleEvents: StreamEvent[], gestureEvents: GestureToggleMap) {
  const countsByGesture: Record<string, number> = {};
  for (const event of events) {
    countsByGesture[event.gesture] = (countsByGesture[event.gesture] ?? 0) + 1;
  }
  const topGesture = Object.entries(countsByGesture).sort((first, second) => second[1] - first[1])[0]?.[0] ?? "";
  return {
    total: events.length,
    visible: visibleEvents.length,
    enabled: Object.values(gestureEvents).filter(Boolean).length,
    countsByGesture,
    topGesture,
  };
}

type TimelinePeak = {
  id: string;
  x: number;
  y: number;
  confidence: number;
  receivedAt: number;
};

type TimelineSeries = {
  gesture: string;
  label: string;
  color: string;
  path: string;
  peaks: TimelinePeak[];
  eventCount: number;
  latestConfidence: number;
  latestReceivedAt: number;
  isTopSeries: boolean;
};

function buildTimelineSeries(
  events: StreamEvent[],
  visibleGestures: GestureToggleMap,
  now: number,
  windowSeconds: number,
): TimelineSeries[] {
  const windowMs = clampChartWindowSeconds(windowSeconds) * 1000;
  const start = now - windowMs;
  const left = 58;
  const right = 984;
  const top = 32;
  const bottom = 352;
  const envelopeMs = Math.min(2500, Math.max(350, windowMs / 80));
  const envelopeReachMs = envelopeMs * 3;
  const sampleCount = Math.min(420, Math.max(120, Math.round(windowSeconds * 4)));
  const byGesture = new Map<string, StreamEvent[]>();

  for (const event of events) {
    if (event.receivedAt < start - envelopeReachMs || event.receivedAt > now) {
      continue;
    }
    if (!(visibleGestures[event.gesture] ?? true)) {
      continue;
    }
    const current = byGesture.get(event.gesture) ?? [];
    current.push(event);
    byGesture.set(event.gesture, current);
  }

  const series = Array.from(byGesture.entries()).map(([gesture, gestureEvents]) => {
    const sorted = [...gestureEvents].sort((first, second) => first.receivedAt - second.receivedAt);
    const visiblePeaks = sorted.filter((event) => event.receivedAt >= start && event.receivedAt <= now);
    const sampleTimes = timelineSampleTimes(start, now, sampleCount, visiblePeaks);
    const pathPoints = sampleTimes.map((time) => ({
      x: xForTime(time, start, now, left, right),
      y: yForConfidence(confidenceEnvelopeAt(time, sorted, envelopeMs), top, bottom),
    }));
    const peaks = visiblePeaks.map((event) => ({
      id: event.id,
      x: xForTime(event.receivedAt, start, now, left, right),
      y: yForConfidence(event.confidence, top, bottom),
      confidence: event.confidence,
      receivedAt: event.receivedAt,
    }));
    const latest = visiblePeaks.at(-1) ?? sorted.at(-1);
    return {
      gesture,
      label: labelForGesture(gesture),
      color: colorForGesture(gesture),
      path: smoothPathFromPoints(pathPoints),
      peaks,
      eventCount: visiblePeaks.length,
      latestConfidence: latest?.confidence ?? 0,
      latestReceivedAt: latest?.receivedAt ?? 0,
      isTopSeries: false,
    };
  });

  series.sort((first, second) => second.latestReceivedAt - first.latestReceivedAt);
  if (series[0]) {
    series[0] = { ...series[0], isTopSeries: true };
  }
  return series;
}

function xForTime(value: number, start: number, end: number, left: number, right: number) {
  const ratio = (value - start) / Math.max(end - start, 1);
  return left + (right - left) * Math.max(0, Math.min(1, ratio));
}

function yForConfidence(value: number, top: number, bottom: number) {
  const clamped = Math.max(0, Math.min(1, value));
  return bottom - (bottom - top) * clamped;
}

function timelineSampleTimes(start: number, end: number, sampleCount: number, events: StreamEvent[]) {
  const duration = Math.max(end - start, 1);
  const times = new Set<number>();
  for (let index = 0; index < sampleCount; index += 1) {
    times.add(start + (duration * index) / Math.max(sampleCount - 1, 1));
  }
  for (const event of events) {
    times.add(event.receivedAt);
  }
  return [...times].sort((first, second) => first - second);
}

function confidenceEnvelopeAt(time: number, events: StreamEvent[], envelopeMs: number) {
  let confidence = 0;
  for (const event of events) {
    const distance = Math.abs(time - event.receivedAt);
    if (distance > envelopeMs * 3) {
      continue;
    }
    const weight = Math.exp(-0.5 * (distance / envelopeMs) ** 2);
    confidence = Math.max(confidence, event.confidence * weight);
  }
  return Math.max(0, Math.min(1, confidence));
}

function smoothPathFromPoints(points: Array<{ x: number; y: number }>) {
  if (points.length === 0) {
    return "";
  }
  if (points.length === 1) {
    return `M ${formatCoordinate(points[0].x)} ${formatCoordinate(points[0].y)}`;
  }

  const commands = [`M ${formatCoordinate(points[0].x)} ${formatCoordinate(points[0].y)}`];
  for (let index = 1; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const midX = (current.x + next.x) / 2;
    const midY = (current.y + next.y) / 2;
    commands.push(
      `Q ${formatCoordinate(current.x)} ${formatCoordinate(current.y)} ${formatCoordinate(midX)} ${formatCoordinate(midY)}`,
    );
  }
  const last = points.at(-1);
  if (last) {
    commands.push(`L ${formatCoordinate(last.x)} ${formatCoordinate(last.y)}`);
  }
  return commands.join(" ");
}

function formatCoordinate(value: number) {
  return value.toFixed(2);
}

function labelForGesture(gesture: string) {
  return GESTURES.find((definition) => definition.name === gesture)?.label ?? gesture;
}

function colorForGesture(gesture: string) {
  const palette = [
    "#22d3ee",
    "#a78bfa",
    "#34d399",
    "#fbbf24",
    "#fb7185",
    "#60a5fa",
    "#f472b6",
    "#c084fc",
    "#2dd4bf",
    "#f97316",
    "#84cc16",
    "#e879f9",
  ];
  const hash = Array.from(gesture).reduce((total, character) => total + character.charCodeAt(0), 0);
  return palette[Math.abs(hash) % palette.length];
}

function clampChartWindowSeconds(value: number) {
  if (!Number.isFinite(value)) {
    return DEFAULT_CHART_WINDOW_SECONDS;
  }
  return Math.max(MIN_CHART_WINDOW_SECONDS, Math.min(MAX_CHART_WINDOW_SECONDS, Math.round(value)));
}

function clampEventLimit(value: number) {
  if (!Number.isFinite(value)) {
    return DEFAULT_EVENT_LIMIT;
  }
  return Math.max(MIN_EVENT_LIMIT, Math.min(MAX_EVENT_LIMIT, Math.round(value)));
}

function clampBrowserInputFps(value: number) {
  if (!Number.isFinite(value)) {
    return DEFAULT_BROWSER_INPUT_FPS;
  }
  return Math.max(MIN_BROWSER_INPUT_FPS, Math.min(MAX_BROWSER_INPUT_FPS, Math.round(value)));
}

function stripClientFields(event: StreamEvent): GestureEventPayload {
  const { id: _id, receivedAt: _receivedAt, ...payload } = event;
  return payload;
}

function createId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function labelForState(state: ConnectionState) {
  return state.replace(/^./, (letter) => letter.toUpperCase());
}

function formatTime(value: number) {
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(value);
}

function formatPercent(value: number) {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function formatNumber(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}
