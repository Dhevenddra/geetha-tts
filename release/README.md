<!-- DRAFT 16 Nov 2026. Lines marked TODO are not yet measured or not yet decided. Nothing
     in this file is published until the operator has decided the claim (D-058). -->

# Geetha — Tamil text to speech, two studio voices

Geetha is a LoRA adapter on [`bodhan-ai/indic-speak`](https://huggingface.co/bodhan-ai/indic-speak),
fine-tuned for Tamil. It speaks in two voices, **anitha** and **arun**, can be steered
into six emotions, and can put an audible breath at the start of a sentence.

Listen in the browser at **[geethalabs.com](https://geethalabs.com)**. The code is at
[github.com/Dhevenddra/geetha-tts](https://github.com/Dhevenddra/geetha-tts).

**Built on Indic-Speak by Bodhan AI.** Geetha is released under the same licence as its
base, the Indic Open Model License v1.0 (`Bodhan_AI_Open_Model_License.md`, included).
You need access to the gated base model to run it.

## What it is, and what it is not

Stated as measured. Every number below is our own run, single shot, with fixed seeds;
the transcripts and audio behind them are released in the companion dataset [`Dhevenddra/geetha-tts-eval`](https://huggingface.co/datasets/Dhevenddra/geetha-tts-eval).

- **It is intelligible, and more so than the two open Tamil models we could run.**
  304 Tamil-script sentences (120 from the FLEURS test split; 100 everyday and
  emotional sentences and 84 harder ones written for this project), one take
  each, character error rate in % with 95% intervals, two ASR judges:

  | System | Usable clips | Indic-Transcribe-core | IndicConformer-600M |
  | --- | ---: | ---: | ---: |
  | Geetha | 303 / 304 | 2.15 [1.87, 2.44] | 2.85 [2.52, 3.21] |
  | Indic-Speak (the base) | 284 / 304 | 1.95 [1.51, 2.58] | 2.60 [2.06, 3.31] |
  | Rumik-OSS-1 | 304 / 304 | 5.68 [4.99, 6.40] | 8.33 [7.55, 9.17] |
  | snorTTS-Indic-v0 | 304 / 304 | 5.14 [4.50, 5.81] | 8.76 [7.87, 9.70] |

  CER is over each system's usable clips. On the sentences both systems rendered,
  Geetha is 3.5 / 5.5 points better than Rumik-OSS-1 and 3.0 / 5.9 better than
  snorTTS-Indic-v0 (all intervals exclude zero). The rivals speak in their own voices
  at their own cards' settings.
  On the earlier 41 mixed Tamil-English prompts every system is far worse: Geetha
  14.3 / 14.7 %, the base 13.3 / 15.0 %, Rumik-OSS-1 17.9 / 21.3 %, snorTTS-Indic-v0
  26.6 / 32.3 % (references for that set were drafted by us and not checked by a
  second reader).
- **On intelligibility it is level with its base model.** The fine-tune did not make
  the words clearer: Geetha minus base is +0.23 [-0.39, +0.67] and +0.26 [-0.44, +0.84]
  points on the 283 sentences both rendered.
- **It finishes its sentences more reliably than the base.** In one take the base ran
  on to the 22-second cap without stopping on 20 of 304 sentences; Geetha did on 1.
  Both rivals stopped every time.
- **One listener preferred it to the base, blind.** 16 plain sentence pairs, same text,
  same seeds, order hidden: Geetha 14, base 0, no difference 2. With a breath added:
  Geetha 10, base 3, no difference 3. This is one listener (the project's operator) and
  8 sentences per voice. It is not a listening panel.
- **The same listener preferred it to both open rivals.** 24 benchmark sentences,
  order hidden, level-matched: Geetha 12, Rumik-OSS-1 0; Geetha 11, snorTTS-Indic-v0 1.
  The listener knows Geetha's two voices, so this was not blind to the system. He
  chose on which voice he would rather listen to, and noted that snorTTS-Indic-v0
  sounds like a real person recorded on a phone microphone. This is a preference for
  Geetha's voices and studio sound, not evidence that Geetha is the more lifelike.
- **Emotion works with guidance, and only Geetha survives it.** Under style guidance
  3.5 the base model's CER rises to 33.9 / 34.5 %; Geetha's stays at 4.4 / 6.4 % (30
  paired clips). Without guidance Geetha is **not** more expressive than the base: the
  pitch shift is equal and Geetha's pitch range is narrower. Emotion is clearest when
  the sentence itself carries the feeling.
- **The styles are audible.** 24 blind pairs, same voice, same sentence, same seed, one
  take with the style and one plain; the listener was told the feeling and asked which
  take had more of it. He picked the styled take 24 times out of 24, 4 of 4 for each
  of the six styles. The sentences were worded to carry their feeling, and this is
  again one listener. His note: anitha stays herself under every style; arun's
  emotions are as clear but his voice can drift a little from how he normally sounds.
- **The breath is inserted, not learned.** The model does not breathe on request in
  these two voices: the studio recordings they come from hold no breaths, and two
  fine-tunes on breath-tagged rows did not change that. `[breath]` at the start of a
  sentence writes one of the model's own breaths (9 for anitha, 18 for arun, stored as
  codec tokens) into the stream, and the model carries on from it. A breath in the
  middle of a sentence is experimental and off by default. We tried two ways of
  placing it and the listener preferred neither (3 against 4, 5 no difference), and
  a mid-sentence breath did not beat no breath at all (6 against 6). Some land well
  and some sound pasted in; which one you get depends on the breath and the sentence.
- **No voice cloning, no laughter or sighs, no other languages tested.**
- Predicted-MOS scores are not reported: none is validated for Tamil.

## Use

Ask for access to the gated base model, log in with `hf auth login`, then fetch the
three small files that make up the recipe:

```
pip install torch transformers peft snac soundfile huggingface_hub
hf download Dhevenddra/geetha-tts inference.py tamil_text.py breaths.json --local-dir geetha
cd geetha
```

```python
from inference import Geetha

tts = Geetha("Dhevenddra/geetha-tts")       # downloads the adapter and the base model
wav = tts.speak("நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.", speaker="anitha")
wav = tts.speak("[breath] இத பத்தி நாம நாளைக்கு பேசலாம்.", speaker="arun", style="SAD")
tts.save("out.wav", wav)          # float32 mono, 24 kHz
```

`inference.py` is the whole recipe and is not optional: the adapter was trained on one
prompt layout and is attached unmerged. Loaded any other way it still speaks, at the
base model's quality. Styles: `CONV` (default), `HAPPY`, `SAD`, `ANGER`, `FEAR`,
`SURPRISE`, `DISGUST`. Digits, dates, times, rupees and percentages are read as Tamil
words. Long text is split into sentences. A take that is silent or never ends is thrown
away and re-rolled (`retries`, default 2); the benchmarks use `retries=0`.

### How to write the text

The text you pass is plain Tamil, written the way you want it said. Three things
change the result: the `speaker`, the `style`, and one optional `[breath]` tag.

| You want | You write |
| --- | --- |
| Anitha, neutral | `tts.speak("நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.", speaker="anitha")` |
| Arun, neutral | `tts.speak("நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.", speaker="arun")` |
| Happy | `tts.speak("எனக்கு வேலை கிடைச்சிருச்சு, ரொம்ப சந்தோஷமா இருக்கு!", speaker="anitha", style="HAPPY")` |
| Sad | `tts.speak("அவங்க போனதுக்கு அப்புறம் வீடே வெறிச்சோடி இருக்கு.", speaker="anitha", style="SAD")` |
| A breath before speaking | `tts.speak("[breath] சரி, நான் ஒண்ணு சொல்றேன், கவனமா கேளு.", speaker="arun")` |
| Numbers, money, dates | `tts.speak("விலை ₹350, 15/11/2026 வரைக்கும் 20% தள்ளுபடி.", speaker="anitha")` |

- Write spoken Tamil if you want it to sound spoken (`இருக்கு`, not `இருக்கிறது`).
  Formal written Tamil is read formally.
- A style works best when the sentence itself carries the feeling. `style="ANGER"` on
  a weather report gives a weather report.
- One `[breath]` per sentence, at the start. Do not write other tags (`[laugh]`,
  `[sigh]`): the model was not trained on them and will try to read them.
- Use punctuation. Commas and full stops are where the voice pauses.
- Keep English words to a minimum, or write them in Tamil script (`மீட்டிங்`).

Speed: on a Tesla T4 a 7-second sentence takes about 40 seconds. This is not a
real-time model on that card. No faster card was measured.

## How it was made

- Base: `bodhan-ai/indic-speak` at revision `76e0814f`. LoRA rank 64 on the attention
  and MLP projections and the (tied) token embeddings.
- Stage A: 62,348 clips, 126.5 hours of Tamil speech, 1.08 epochs (9 GPU-hours).
  Two corpora, both CC BY 4.0, both from AI4Bharat:
  [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) (about 79
  hours, many speakers, used only as anonymous training signal; its test split is
  not used) and [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) (about 48
  hours, the two studio speakers).
- Stage B: the two studio voices, 16,530 clips, 20.3 distinct hours, 0.53 epoch
  (3 GPU-hours). Rasa only.
- Total paid compute: ₹724.
- Test sentences were checked against every training transcript by normalised text;
  none is in training.

## Known limits

- English words and digits inside Tamil text are the weak spot for every system we
  tested, Geetha included.
- In a single take, 1 of 304 benchmark sentences never ended (0.3%); none was silent.
  `speak()` re-rolls such takes by default.
- The two voices are studio read speech. Geetha sounds like a clear reader, not like a
  person talking in a room.
- The codec (SNAC, decoded with Vocos) costs intelligibility by itself, before any
  model is involved.

## Reproduce the numbers

Everything behind the table is in the dataset [`Dhevenddra/geetha-tts-eval`](https://huggingface.co/datasets/Dhevenddra/geetha-tts-eval):

- `bench_ta_v2.json`: the 304 sentences, their source and the voice each was given.
- `transcripts/bench_<system>.json`: per clip, whether it was usable and why not, its
  length, and what each of the two judges heard. Also the decoding settings, seeds
  and model revisions of that run.
- `clips/<system>__<id>.wav`: every clip, usable or not, for Geetha, the base and
  snorTTS-Indic-v0. Rumik-OSS-1's audio is not redistributed here because its licence
  is non-commercial; its transcripts and scores are included, and the kernel in `code/`
  regenerates the audio.
- `bench_v2.md`, `bench_v2.json`: the scores, including per-clip error rates.
- `code/`: the two Kaggle kernel files that rendered and transcribed the clips, and
  `score_bench_v2.py`, which turns the transcripts into the table.
- `listening/`: the pairs, keys and answers of the two listening tests.

Seed for sentence *i* is 1234 + 100·*i*. Each system ran once per sentence.
