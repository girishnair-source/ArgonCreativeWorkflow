"""Media helpers: ffmpeg/ffprobe wrappers (through :mod:`procs`) plus pure-Python PNG/WAV writers.

The pure-Python writers let tests and offline/fake generation build valid media without any network access.
Functions that write into take folders refuse to overwrite existing files unless ``overwrite=True``
(take media are immutable artifacts).
"""

from __future__ import annotations

import json
import math
import struct
import wave
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .procs import CommandFailed, run


def _guard_out(out: Path, overwrite: bool) -> Path:
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------------------------
def probe(path: Path) -> dict[str, Any]:
    out = run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
         "-of", "json", str(path)],
        timeout=60,
    ).stdout
    return json.loads(out or "{}")


def duration_s(path: Path) -> float:
    value = probe(path).get("format", {}).get("duration")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def has_audio(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def video_size(path: Path) -> tuple[int, int]:
    for s in probe(path).get("streams", []):
        if s.get("codec_type") == "video":
            return int(s["width"]), int(s["height"])
    raise ValueError(f"no video stream in {path}")


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------
def extract_frame(video: Path, out: Path, *, at: str | float = "first", overwrite: bool = False) -> Path:
    """Write the first / last frame (or the frame at ``at`` seconds) of ``video`` to ``out`` (png/jpg)."""
    out = _guard_out(out, overwrite)
    if at == "first":
        run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-frames:v", "1", str(out)], timeout=120)
    elif at == "last":
        try:
            run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.5", "-i", str(video), "-update", "1", str(out)],
                timeout=120)
        except CommandFailed:
            run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-update", "1", str(out)], timeout=300)
    else:
        t = max(0.0, float(at))
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(out)],
            timeout=120)
    if not out.exists():
        raise RuntimeError(f"ffmpeg produced no frame for {video} at {at}")
    return out


def extract_wav(video: Path, out: Path, *, sample_rate: int = 16000, mono: bool = True,
                overwrite: bool = False) -> Path | None:
    """Extract the audio track as PCM WAV. Returns None when the media has no audio stream."""
    if not has_audio(video):
        return None
    out = _guard_out(out, overwrite)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1" if mono else "2",
         "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out)], timeout=300)
    return out


def mute_video(src: Path, out: Path, *, overwrite: bool = False) -> Path:
    out = _guard_out(out, overwrite)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "copy", "-an", str(out)], timeout=300)
    return out


def read_image_rgb(path: Path, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Decode any image/video frame to an RGB uint8 array of ``size`` (w, h) via ffmpeg rawvideo."""
    w, h = size
    proc = run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-vf", f"scale={w}:{h}",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], timeout=120, binary_output=True)
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    if data.size != w * h * 3:
        raise ValueError(f"could not decode {path} to {w}x{h} RGB")
    return data.reshape(h, w, 3)


# ---------------------------------------------------------------------------------------------
# Synthetic media (fakes, tests, offline demos)
# ---------------------------------------------------------------------------------------------
def make_test_clip(out: Path, *, duration_s: float = 4.0, size: tuple[int, int] = (270, 480), fps: int = 24,
                   tone_hz: float | None = 440.0, color: str | None = None, overwrite: bool = False) -> Path:
    """Synthetic H.264/AAC clip (testsrc2 or a solid colour) with an optional sine-tone audio track."""
    out = _guard_out(out, overwrite)
    w, h = size
    src = (f"color=c={color}:size={w}x{h}:rate={fps}:duration={duration_s}" if color
           else f"testsrc2=size={w}x{h}:rate={fps}:duration={duration_s}")
    base = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src]
    if tone_hz:
        base += ["-f", "lavfi", "-i", f"sine=frequency={tone_hz}:sample_rate=48000:duration={duration_s}"]
    for vcodec in (["-c:v", "libx264", "-preset", "ultrafast"], ["-c:v", "mpeg4", "-q:v", "5"]):
        argv = [*base, *vcodec, "-pix_fmt", "yuv420p"]
        if tone_hz:
            argv += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
        try:
            run([*argv, str(out)], timeout=300)
            return out
        except CommandFailed:
            continue
    raise RuntimeError("ffmpeg could not encode a synthetic test clip")


def write_png(path: Path, width: int = 64, height: int = 64, rgb: tuple[int, int, int] = (128, 128, 128),
              *, overwrite: bool = False) -> Path:
    """Minimal solid-colour PNG writer (no dependencies)."""
    path = _guard_out(path, overwrite)
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    path.write_bytes(png)
    return path


def write_tone_wav(path: Path, *, duration_s: float = 2.0, freq: float = 220.0, sample_rate: int = 16000,
                   amplitude: float = 0.3, harmonics: tuple[float, ...] = (1.0, 0.5, 0.25),
                   overwrite: bool = False) -> Path:
    """16-bit mono WAV with a harmonic tone (a crude 'voice' timbre for speaker-similarity tests)."""
    path = _guard_out(path, overwrite)
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    sig = np.zeros(n, dtype=np.float64)
    for k, a in enumerate(harmonics, start=1):
        sig += a * np.sin(2 * math.pi * freq * k * t)
    peak = float(np.max(np.abs(sig))) or 1.0
    pcm = (sig / peak * amplitude * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV as float32 mono in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw} in {path}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def make_test_png(
    path: Path,
    *,
    label: str = "",
    width: int = 64,
    height: int = 64,
    rgb: tuple[int, int, int] = (128, 128, 128),
    overwrite: bool = False,
) -> Path:
    """Convenience alias for :func:`write_png` accepting an optional label."""
    _ = label
    return write_png(path, width=width, height=height, rgb=rgb, overwrite=overwrite)


def write_wav(
    path: Path,
    *,
    duration_s: float = 2.0,
    freq_hz: float = 220.0,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> Path:
    """Convenience alias for :func:`write_tone_wav` accepting ``freq_hz``."""
    return write_tone_wav(path, duration_s=duration_s, freq=freq_hz, sample_rate=sample_rate, overwrite=overwrite)

