#!/usr/bin/env python3
"""Render every per-slide text file in a directory to a WAV using Chatterbox TTS.

Reads `slide-NN.txt` files from the given build directory and writes
`slide-NN.wav` next to each one at 48 kHz, 16-bit, stereo. The voice is
either Chatterbox's built-in default voice or, if `--voice <path>` is
given, the voice cloned from a short reference audio clip.

Chatterbox is Resemble AI's open-source TTS (https://github.com/resemble-ai/chatterbox).
Install with `pip install chatterbox-tts`.

Long sentences cause Chatterbox to occasionally repeat or stutter, so this
driver splits each slide's text on sentence boundaries (period / question
mark / exclamation mark followed by whitespace) and synthesises each
sentence separately. The per-sentence WAVs are concatenated with a short
silence gap between them, which makes the result sound natural while
keeping each Chatterbox call short enough to be reliable.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", help="directory containing slide-NN.txt files")
    parser.add_argument(
        "--voice",
        default=None,
        help=(
            "optional path to a 5-15s reference audio clip "
            "(any common format) used to clone the speaker; "
            "if omitted, Chatterbox's default voice is used"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="torch device for inference (default: auto-detect)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="emotion intensity, 0.0 (flat) to 1.0 (extreme); default 0.5",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.3,
        help=(
            "classifier-free guidance weight, 0.0 to 1.0; default 0.3 "
            "(lower than the Chatterbox default of 0.5; reduces repetition "
            "and stutter on technical content)"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help=(
            "sampling temperature, 0.0 (deterministic) to 1.0+ (more "
            "expressive); default 0.7 (lower than the Chatterbox default "
            "of 0.8; further reduces repetition risk)"
        ),
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=48000,
        help="output sample rate (Hz); default 48000",
    )
    parser.add_argument(
        "--no-chunk-sentences",
        dest="chunk_sentences",
        action="store_false",
        default=True,
        help=(
            "synthesise each slide as a single chunk instead of splitting on "
            "sentence boundaries (default: chunk; per-sentence chunking is "
            "more reliable but adds small inter-sentence pauses)"
        ),
    )
    parser.add_argument(
        "--sentence-gap-seconds",
        type=float,
        default=0.18,
        help="silence between sentences when chunking; default 0.18 s",
    )
    return parser.parse_args()


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Sentence splitter: split on .!? followed by whitespace, keeping the
# punctuation attached to the preceding sentence. The script-side authoring
# rule keeps sentences short, so this regex stays simple and does not try
# to handle abbreviations like "e.g." (we have none in the script).
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text)]
    return [p for p in parts if p]


def synthesise(model, text: str, *, chunk_sentences: bool,
               sentence_gap_seconds: float, gen_kwargs: dict):
    """Run Chatterbox on `text`, optionally chunking on sentence boundaries.

    Returns a torch tensor shaped (channels, samples) at `model.sr`.
    """
    import torch  # type: ignore

    if not chunk_sentences:
        wav = model.generate(text, **gen_kwargs)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        wav = model.generate(text, **gen_kwargs)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav

    parts = []
    n_gap = max(1, int(sentence_gap_seconds * model.sr))
    for i, sentence in enumerate(sentences):
        wav = model.generate(sentence, **gen_kwargs)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        parts.append(wav)
        if i < len(sentences) - 1:
            gap = torch.zeros(
                wav.shape[0], n_gap,
                dtype=wav.dtype, device=wav.device,
            )
            parts.append(gap)
    return torch.cat(parts, dim=-1)


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)

    try:
        from chatterbox.tts import ChatterboxTTS  # type: ignore
    except ImportError:
        print(
            "chatterbox-tts is not installed. Install with: pip install chatterbox-tts",
            file=sys.stderr,
        )
        return 2

    try:
        import torch  # type: ignore  # noqa: F401
        import torchaudio  # type: ignore
    except ImportError:
        print(
            "torch and torchaudio are required for chatterbox-tts. "
            "Install with: pip install torch torchaudio",
            file=sys.stderr,
        )
        return 2

    print(f"loading Chatterbox on device={device}")
    model = ChatterboxTTS.from_pretrained(device=device)

    build_dir = Path(args.build_dir)
    txts = sorted(build_dir.glob("slide-*.txt"))
    if not txts:
        print(f"no slide-*.txt files in {build_dir}", file=sys.stderr)
        return 2

    for txt_path in txts:
        text = txt_path.read_text().strip()
        if not text:
            print(f"skipping {txt_path.name}: empty text")
            continue

        gen_kwargs = dict(
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
        )
        if args.voice:
            gen_kwargs["audio_prompt_path"] = args.voice

        n_sent = len(split_sentences(text)) if args.chunk_sentences else 1
        print(
            f"synthesising {txt_path.name}: "
            f"{len(text)} chars, {n_sent} sentence chunk(s)"
        )
        wav = synthesise(
            model, text,
            chunk_sentences=args.chunk_sentences,
            sentence_gap_seconds=args.sentence_gap_seconds,
            gen_kwargs=gen_kwargs,
        )

        if model.sr != args.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=model.sr, new_freq=args.target_sr
            )
            wav = resampler(wav)

        # Up-mix mono to stereo so ffmpeg's later -ac 2 step is a no-op.
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        out_path = txt_path.with_suffix(".wav")
        wav_cpu = wav.cpu()
        try:
            import soundfile as sf  # type: ignore

            # soundfile expects (frames, channels) int16 or float; we provide
            # float32 in [-1, 1] and let soundfile clip into PCM_16.
            sf.write(
                str(out_path),
                wav_cpu.transpose(0, 1).numpy(),
                args.target_sr,
                subtype="PCM_16",
            )
        except ImportError:
            torchaudio.save(
                str(out_path),
                wav_cpu,
                args.target_sr,
                encoding="PCM_S",
                bits_per_sample=16,
            )
        duration = wav.shape[-1] / args.target_sr
        print(f"wrote {out_path.name}: {duration:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
