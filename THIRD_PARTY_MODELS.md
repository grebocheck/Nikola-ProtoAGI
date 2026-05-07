# Third-Party Models

Last checked: 2026-05-07.

This repository does not redistribute model weights. The local `models/`
directory is ignored by git except for `models/.gitkeep`; downloaded weights and
Hugging Face caches are operator-managed local files.

The Apache-2.0 license in this repository covers the project source code,
scripts, tests, and documentation. It does not relicense third-party model
weights, tokenizer files, GGUF conversions, vocoders, TTS models, or downloaded
runtime binaries. Check each upstream license before downloading, modifying, or
redistributing a model.

| Local use | Default reference | Upstream license/status |
| --- | --- | --- |
| Main chat/reasoning model | `models/gpt-oss-20b-MXFP4.gguf` | gpt-oss weights are described by OpenAI as Apache-2.0 and subject to the gpt-oss usage policy. The local GGUF file is not committed. |
| Optional vision model | `ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M` | GGUF conversion of Hugging Face's SmolVLM2-2.2B-Instruct. The upstream SmolVLM2 checkpoint is Apache-2.0. The local cache is not committed. |
| Optional embedding model | `CompendiumLabs/bge-m3-gguf:Q4_K_M` | Hugging Face lists the GGUF repository as MIT-licensed. The local cache is not committed. |
| Optional voice transcription | `whisper-large-v3` endpoint/operator choice | This project only calls an OpenAI-compatible `/audio/transcriptions` endpoint. If using OpenAI Whisper weights, verify the exact distribution's license first; OpenAI's Whisper GitHub repository states its code and model weights are MIT-licensed, while the Hugging Face `openai/whisper-large-v3` page lists Apache-2.0. |
| Optional TTS | `tts-local` endpoint/operator choice | Placeholder model name for a local `/audio/speech` endpoint. No TTS weights are bundled or licensed by this repository. |

Useful upstream references:

- OpenAI gpt-oss help: https://help.openai.com/en/articles/11870455-openai-open-weight-models-gpt-oss
- OpenAI gpt-oss model card: https://openai.com/index/gpt-oss-model-card/
- OpenAI gpt-oss repository: https://github.com/openai/gpt-oss
- SmolVLM2 upstream model card: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- SmolVLM2 GGUF repository: https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF
- bge-m3 GGUF repository: https://huggingface.co/CompendiumLabs/bge-m3-gguf
- OpenAI Whisper repository: https://github.com/openai/whisper
- OpenAI Whisper large-v3 on Hugging Face: https://huggingface.co/openai/whisper-large-v3

This file is a project hygiene note, not legal advice.
