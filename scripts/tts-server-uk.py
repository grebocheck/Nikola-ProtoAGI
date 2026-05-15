"""OpenAI-compatible TTS server for Ukrainian, backed by Piper.

Exposes ``POST /v1/audio/speech`` (and ``GET /v1/voices``, ``GET /v1/models``)
so ProtoAGI's ``voice.py`` works unchanged — just point
``PROTOAGI_TTS_BASE_URL`` at this server. Native Piper output is 22 kHz mono
WAV; the server transcodes through ffmpeg into the requested
``response_format`` (opus/ogg/mp3/aac/flac/wav). Telegram ``sendVoice``
needs OGG/Opus, which is the default.

Why Piper UA: Coqui XTTS-v2 routes ``language=uk`` through Russian
phonemes, which is why the previous setup had a heavy Russian accent and
mis-stressed words. Piper's ``uk_UA-ukrainian_tts-medium`` is trained on
the robinhad/ukrainian-tts dataset with proper Ukrainian phonetics.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise SystemExit(
        "fastapi/pydantic not installed. Run: "
        "pip install fastapi uvicorn[standard] piper-tts"
        f"\nUnderlying error: {exc}"
    )

try:
    from piper.voice import PiperVoice
    from piper.config import SynthesisConfig
except ImportError as exc:
    raise SystemExit(
        "piper-tts not installed. Run: pip install piper-tts"
        f"\nUnderlying error: {exc}"
    )


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "config" / "tts" / "models"
DEFAULT_VOICE_MAP = ROOT / "config" / "tts" / "voice_map.json"
DEFAULT_MODEL_NAME = "uk_UA-ukrainian_tts-medium"


_FMT_TO_FFMPEG: dict[str, list[str]] = {
    "opus": ["-c:a", "libopus", "-b:a", "32k", "-application", "voip", "-f", "ogg"],
    "ogg":  ["-c:a", "libopus", "-b:a", "32k", "-application", "voip", "-f", "ogg"],
    "mp3":  ["-c:a", "libmp3lame", "-q:a", "4", "-f", "mp3"],
    "aac":  ["-c:a", "aac", "-b:a", "96k", "-f", "adts"],
    "flac": ["-c:a", "flac", "-f", "flac"],
}

_FMT_TO_MIME: dict[str, str] = {
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/wav",
}


class SpeechRequest(BaseModel):
    model: str = ""
    input: str
    voice: str = ""
    response_format: str = "opus"
    speed: float = 1.0


def _load_voice_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    voices = raw.get("voices") if isinstance(raw, dict) and "voices" in raw else raw
    return voices or {}


_LATIN_FALLBACK: dict[str, str] = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "г", "i": "і", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}


def _normalize_for_piper(text: str) -> str:
    """Piper's uk_UA-ukrainian_tts-medium uses ``phoneme_type=text`` with a
    lowercase-Cyrillic ID map, so capital letters and Latin characters get
    silently dropped during phonemization. Lowercase the input and replace
    Latin letters with a crude Cyrillic transliteration so brand names like
    "Piper" stay audible."""
    out: list[str] = []
    for ch in text.lower():
        repl = _LATIN_FALLBACK.get(ch)
        out.append(repl if repl is not None else ch)
    return "".join(out)


def _resolve_speaker(
    voice_map: dict[str, dict[str, Any]],
    speaker_id_map: dict[str, int],
    requested: str,
) -> tuple[int | None, float]:
    entry = (
        voice_map.get(requested)
        or voice_map.get(requested.lower())
        or {}
    )
    speaker_name = str(entry.get("speaker", "")).strip()
    length_scale = float(entry.get("length_scale", 1.0))
    speaker_id: int | None = None
    if speaker_name:
        if speaker_name in speaker_id_map:
            speaker_id = int(speaker_id_map[speaker_name])
        elif speaker_name.isdigit():
            speaker_id = int(speaker_name)
    elif speaker_id_map:
        first_name = next(iter(speaker_id_map))
        speaker_id = int(speaker_id_map[first_name])
    return speaker_id, length_scale


def _synthesize_wav(
    voice: PiperVoice,
    text: str,
    speaker_id: int | None,
    length_scale: float,
) -> bytes:
    buffer = io.BytesIO()
    syn_config = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=length_scale,
    )
    with wave.open(buffer, "wb") as wav:
        voice.synthesize_wav(text, wav, syn_config=syn_config)
    return buffer.getvalue()


def _transcode(wav_bytes: bytes, response_format: str) -> bytes:
    fmt = (response_format or "opus").lower().strip()
    if fmt in ("wav", "pcm", ""):
        return wav_bytes
    args = _FMT_TO_FFMPEG.get(fmt)
    if args is None:
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {fmt}")
    if shutil.which("ffmpeg") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ffmpeg not found in PATH. Either install ffmpeg or set "
                "PROTOAGI_TTS_RESPONSE_FORMAT=wav."
            ),
        )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"]
    proc = subprocess.run(cmd, input=wav_bytes, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace")[:400]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {detail}")
    return proc.stdout


def create_app(model_path: Path, voice_map_path: Path) -> FastAPI:
    if not model_path.exists():
        raise SystemExit(f"Piper model not found: {model_path}")
    config_path = Path(str(model_path) + ".json")
    if not config_path.exists():
        raise SystemExit(f"Piper model config not found: {config_path}")

    voice = PiperVoice.load(str(model_path), config_path=str(config_path))
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    speaker_id_map: dict[str, int] = raw_config.get("speaker_id_map") or {}
    voice_map = _load_voice_map(voice_map_path)

    app = FastAPI(title="ProtoAGI Ukrainian TTS (Piper)", version="1.0.0")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "protoagi-uk-tts",
            "backend": "piper",
            "model": model_path.stem,
            "endpoints": ["/v1/audio/speech", "/v1/voices", "/v1/models"],
        }

    @app.get("/v1/voices")
    def list_voices() -> dict[str, Any]:
        return {
            "model": model_path.stem,
            "speakers": speaker_id_map,
            "voice_map": voice_map,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"data": [{"id": model_path.stem, "object": "model"}]}

    @app.post("/v1/audio/speech")
    def synth(req: SpeechRequest) -> Response:
        text = (req.input or "").strip()
        if not text:
            return JSONResponse(status_code=400, content={"error": "empty input"})
        speaker_id, base_scale = _resolve_speaker(
            voice_map, speaker_id_map, req.voice or ""
        )
        speed = max(0.5, min(2.0, float(req.speed or 1.0)))
        length_scale = base_scale / speed
        normalized = _normalize_for_piper(text)
        try:
            wav_bytes = _synthesize_wav(voice, normalized, speaker_id, length_scale)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=500, content={"error": str(exc)})
        audio = _transcode(wav_bytes, req.response_format)
        mime = _FMT_TO_MIME.get((req.response_format or "opus").lower(), "application/octet-stream")
        return Response(content=audio, media_type=mime)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="ProtoAGI Ukrainian Piper TTS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_NAME}.onnx"),
        help="Path to .onnx Piper voice model",
    )
    parser.add_argument(
        "--voice-map",
        default=str(DEFAULT_VOICE_MAP),
        help="Path to persona voice map JSON",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install 'uvicorn[standard]'", file=sys.stderr)
        sys.exit(1)

    app = create_app(Path(args.model), Path(args.voice_map))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
