"""MediaPipe-compatible 21-point hand landmark indices."""

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

PALM_INDICES = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

FINGER_TIPS = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}

FINGER_PIPS = {
    "thumb": THUMB_IP,
    "index": INDEX_PIP,
    "middle": MIDDLE_PIP,
    "ring": RING_PIP,
    "pinky": PINKY_PIP,
}

FINGER_MCP = {
    "thumb": THUMB_MCP,
    "index": INDEX_MCP,
    "middle": MIDDLE_MCP,
    "ring": RING_MCP,
    "pinky": PINKY_MCP,
}
