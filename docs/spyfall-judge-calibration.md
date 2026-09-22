# Spyfall Leak-Judge Calibration

The Spyfall leak judge awards a round to the Spy whenever the pinned judge
model returns a leak probability at or above the configured
`judge_leak_threshold`. A threshold is only meaningful for the exact model it
was calibrated against. This document defines the calibration protocol, the
artifact every deployment must be able to point at, and the privacy trade-off
operators accept when enabling the judge.

## Protocol

1. Freeze a labeled corpus for the pinned model — currently
   `typesafe/jev-1.13` through OpenRouter's System One/Decisions endpoint.
   The corpus is offline and versioned; entries never change after labeling.
2. Run every corpus entry through the same request the plugin sends: the
   stable `LEAK_RUBRIC` instructions, the labeled secret location, and the
   candidate text as a single `noul` question.
3. Score each entry against a deterministic exact-name baseline (a
   case-insensitive match of the location in the text) in addition to the
   human label.
4. Select the deployment threshold from the resulting table. Thresholds are
   deliberately deployment-configurable: there is no universal pass rate, and
   every live configuration sets `judge_leak_threshold` explicitly. A
   configuration must never rely on an implicit default tuned for a different
   model, and the judge must never silently fall back to a rolling model
   alias (IDs starting with `~` are rejected at configuration time) or an
   application-level fallback model.

## Corpus composition

A calibration corpus must include, at minimum:

- direct mentions of the secret location;
- unambiguous paraphrases and identifying cultural references;
- legitimate non-leaking answers that share vocabulary with the location;
- negation and quotation ("don't ask me about the airport", quoting a
  location name without asserting it);
- deliberately misleading text;
- multilingual text, if multilingual play is intended; and
- attempts to manipulate the judge (instructions embedded in candidate text).

## Required artifact fields

The calibration artifact records, for the selected threshold:

- corpus version and the rubric version (the `LEAK_RUBRIC` constant);
- sample size;
- true positives, false positives, true negatives, and false negatives,
  against both the human labels and the exact-name baseline;
- repeat consistency across identical repeated requests;
- p50 and p95 end-to-end latency;
- provider error rate (timeouts, rate limits, malformed responses); and
- the selected threshold value itself.

Keep the artifact with the experiment it justifies. Re-calibrate before
changing the model ID, the rubric, or the payload shape.

## Privacy disclosure

Enabling the judge sends game secrets off the local machine. Each judged
question or answer produces one request containing the selected secret
location, the rubric, and the candidate public text. OpenRouter receives that
request and forwards it to TypeSafe, the sole provider for this model; both
services therefore see the location and the candidate text.

Local traces keep the same payloads. Credential-shaped fields are redacted
recursively by the event writer, but game secrets are not: `judge_request`
payloads, `judge_response` decisions, and raw provider payloads stay
unredacted so runs remain forensically analyzable. Treat local traces, and the
OpenRouter-to-TypeSafe request path, as sensitive artifacts.
