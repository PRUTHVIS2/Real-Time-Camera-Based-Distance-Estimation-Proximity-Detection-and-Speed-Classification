"""Windowed speed estimation, guards, classification, and voting (pure).

NOTE on Frame Rate Consistency:
    The parameter `min_points` in SpeedConfig must stay consistent with real camera FPS.
    For example, with default `window_s = 0.7` and real FPS ~10 FPS:
      - 5 points spans 0.5 s of trajectory (within the 0.7 s window).
    If the effective frame rate drops lower, `min_points` must be lowered or `window_s`
    must be extended so that at least `min_points` can accumulate within `window_s`.
"""

