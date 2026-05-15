"""OpenAI-compatible voice transcription bridge.

Exposes ``POST /v1/audio/transcriptions`` on the same shape the OpenAI
client expects (multipart with ``file`` and ``model`` fields). Backed
by ``faster-whisper`` (CTranslate2) — Ukrainian quality is excellent
on ``large-v3`` / ``large-v3-turbo`` and acceptable on ``medium``.

ProtoAGI's voice client (``src/protoagi/telegram/voice.py``) talks to
this server; the bot's text reply pipeline already knows how to fall
through gracefully if it's not running.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
except ImportError as exc:
    raise SystemExit(
        "fastapi/uvicorn not installed. Run: "
        "pip install fastapi 'uvicorn[standard]' python-multipart faster-whisper"
        f"\nUnderlying error: {exc}"
    )

try:
    from faster_whisper import WhisperModel
except ImportError as exc:
    raise SystemExit(
        "faster-whisper not installed. Run: pip install faster-whisper"
        f"\nUnderlying error: {exc}"
    )


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "runs" / "voice-cache"


def create_app(
    *,
    model_size: str,
    device: str,
    compute_type: str,
    default_language: str | None,
    download_root: Path,
) -> FastAPI:
    download_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Loading faster-whisper model={model_size} device={device} "
        f"compute_type={compute_type} cache={download_root} ...",
        flush=True,
    )
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(download_root),
    )
    print("Whisper model ready.", flush=True)

    app = FastAPI(title="ProtoAGI Whisper bridge", version="1.0.0")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "protoagi-whisper",
            "model": model_size,
            "device": device,
            "compute_type": compute_type,
            "default_language": default_language,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        # The OpenAI client only needs the id field; we expose the
        # configured model under whatever alias the operator set in
        # PROTOAGI_VOICE_MODEL so the probe matches.
        return {"data": [{"id": model_size, "object": "model"}]}

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form(default=""),  # noqa: ARG001 — accepted for API parity
        language: str = Form(default=""),
        prompt: str = Form(default=""),
        response_format: str = Form(default="json"),  # noqa: ARG001
        temperature: float = Form(default=0.0),
    ) -> dict[str, Any]:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty audio file")
        lang = (language or default_language or "").strip() or None
        buffer = io.BytesIO(data)
        try:
            segments_iter, info = model.transcribe(
                buffer,
                language=lang,
                temperature=float(temperature),
                initial_prompt=(prompt or None),
                vad_filter=True,
            )
            # ``segments_iter`` is a generator; materialise it before we
            # join, otherwise it consumes lazily and we can't get a
            # length / debug info.
            segments = list(segments_iter)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"transcription failed: {type(exc).__name__}: {exc}",
            ) from exc
        text = "".join(seg.text for seg in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "duration": getattr(info, "duration", None),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="ProtoAGI Whisper bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper model id (tiny|base|small|medium|large-v3|large-v3-turbo|"
        "distil-large-v3 or a local path).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="CPU is the safe default — keeps the GPU free for gpt-oss-20b.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="int8 (CPU), float16 / int8_float16 (GPU). See faster-whisper docs.",
    )
    parser.add_argument(
        "--language",
        default="uk",
        help="Default language hint. Whisper still auto-detects per request "
        "when the client omits it.",
    )
    parser.add_argument(
        "--download-root",
        default=str(DEFAULT_CACHE_DIR),
        help="Where to cache CTranslate2 weights.",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install 'uvicorn[standard]'", file=sys.stderr)
        sys.exit(1)

    app = create_app(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        default_language=args.language,
        download_root=Path(args.download_root),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
