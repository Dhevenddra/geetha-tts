# Geetha — Tamil text to speech

[![Geetha, an open voice for Tamil](https://geethalabs.com/assets/img/og.jpg)](https://geethalabs.com)

**Listen in the browser: [geethalabs.com](https://geethalabs.com)**

Geetha is a small fine-tune (a LoRA adapter) of
[Indic-Speak by Bodhan AI](https://huggingface.co/bodhan-ai/indic-speak) for Tamil. It
speaks in two studio voices, **anitha** and **arun**, can be steered into six emotions,
and can take a breath before a sentence.

This repository is the code behind it: data preparation, training, the benchmark, the
listening tests and the release file. The model itself, the audio samples and every raw
benchmark output are on Hugging Face: **https://huggingface.co/Dhevenddra/geetha-tts**.

This is a first project by a small team learning in public. It builds on other people's
work, the numbers are modest, and the limits are written next to the results. If you
find a mistake, please open an issue; we would rather know. The code was written with
the help of an AI coding assistant.

## What it does, as measured

304 Tamil sentences, one take each, two automatic judges. Character error rate in %,
lower is better, 95% intervals in brackets.

| System | Usable clips | Indic-Transcribe-core | IndicConformer-600M |
| --- | ---: | ---: | ---: |
| Geetha | 303 / 304 | 2.15 [1.87, 2.44] | 2.85 [2.52, 3.21] |
| Indic-Speak (the base) | 284 / 304 | 1.95 [1.51, 2.58] | 2.60 [2.06, 3.31] |
| Rumik-OSS-1 | 304 / 304 | 5.68 [4.99, 6.40] | 8.33 [7.55, 9.17] |
| snorTTS-Indic-v0 | 304 / 304 | 5.14 [4.50, 5.81] | 8.76 [7.87, 9.70] |

- Geetha is as clear as its base model, not clearer. What the fine-tune changed is
  reliability (it stops when the sentence ends) and how the two voices sound.
- It makes fewer errors than the two other open Tamil models we could run. They speak
  in different voices at their own settings, so this compares systems, not recipes.
- The six emotion styles are audible: in 24 blind pairs one listener picked the styled
  take every time. One listener is not a panel.
- The breath is inserted, not learned. A breath in the middle of a sentence is
  experimental.
- English words and digits inside Tamil text are still the weak spot.
- It is slow: about 40 seconds for a 7-second sentence on a Tesla T4.

The model card has the full list, including what did not work.

## Use it

You need access to the gated base model on Hugging Face first.

```python
from inference import Geetha          # release/inference.py

tts = Geetha("Dhevenddra/geetha-tts")
wav = tts.speak("நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.", speaker="anitha")
tts.save("out.wav", wav)              # float32 mono, 24 kHz
```

### How to write the text

| You want | You write |
| --- | --- |
| Anitha or Arun | `speaker="anitha"` or `speaker="arun"` |
| An emotion | `style="HAPPY"` (also `SAD`, `ANGER`, `FEAR`, `SURPRISE`, `DISGUST`) |
| A breath before speaking | start the text with `[breath]` |
| Numbers, money, dates | write them as digits: `₹350`, `15/11/2026`, `20%` |

Write spoken Tamil if you want it to sound spoken. A style works best when the sentence
itself carries the feeling. Use commas and full stops; that is where the voice pauses.

## What is in this repository

| Folder | What it holds |
| --- | --- |
| `release/` | `inference.py`, the number-to-words normaliser, the breath library, the model card |
| `geetha/text`, `geetha/audio`, `geetha/data` | text normalisation, audio checks, SNAC token layout |
| `geetha/train` | the two-stage LoRA training |
| `geetha/scoring`, `geetha/eval` | CER, bootstrap intervals, leak checks, validity gate |
| `kaggle/` | every free-GPU job we ran, as it ran, including the ones that failed |
| `scripts/` | dataset builders, the benchmark, the blind listening pages and their scoring |
| `ops/` | the guard rails around the paid GPU runs (total spend: ₹724) |
| `tests/` | unit tests |

To run the unit tests: `pip install -e ".[test]"`, then `pytest`. They need no GPU and no data.

Paths in the scripts point at one Windows machine (`E:\Geetha\...`). They document what
was run; they will need editing to run elsewhere.

## Reproduce the benchmark

The prompt list, both judges' transcripts, 912 clips and the scoring script are
in the dataset [Dhevenddra/geetha-tts-eval](https://huggingface.co/datasets/Dhevenddra/geetha-tts-eval). `scripts/score_bench_v2.py` turns the transcripts into
the table above.

## Thanks

- [Bodhan AI](https://huggingface.co/bodhan-ai) for Indic-Speak, the model this is built on.
- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the Rasa and IndicVoices-R corpora and
  the IndicConformer judge.
- The authors of Rumik-OSS-1 and snorTTS-Indic-v0 for releasing open Tamil models we
  could compare against.
- Google's FLEURS for test sentences, and SNAC and Vocos for the codec and decoder.

## Licence

The model weights are released under the base model's licence, the Indic Open Model
License v1.0 (see the model page). The code licence is in `LICENSE`.
