## Palette Clamp RCA

### Evidence from `logs/video_compressor.log`

- `20:03:41 | ERROR | Palette generation failed`: FFmpeg rejects `palettegen=max_colors=285` because the valid range is `2-256`. The failure repeats across retries (`pal_4c8fc5af...png` and `_r1.png`) showing the same `Value 285.000000 ... out of range` message.
- Immediately afterwards, Stage 3c fallback re-encodes warn that outputs are **>10% larger than input** (e.g., `Output 29.37MB is >10% larger than input 12.69MB`), confirming the optimizer keeps feeding the same oversized palette budget back into FFmpeg.
- Segmentation inherits the invalid settings: when the workflow retries segment 1 with guardrail overrides, palette generation again fails (`Failed to generate palette`) because `_segment_palette_timeout` still delivers `max_colors=285`, causing the workflow to abort after two SIGINTs.

### Conclusion

`max_colors` values above 256 now flow through the Stage 2 re-encode and segmentation paths, violating FFmpeg’s palettegen constraints. We must clamp every palette budget to ≤256 (and preferably mirror that in config defaults) to prevent hard failures and the subsequent runaway retries that led to forced shutdown in this run.

