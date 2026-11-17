# GTA VI Twitch CCU Forecaster — honest generalizer

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aikbani/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aikbani/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day (2026-11-19), from everything
recorded up to the end of its first hour. Entered in the make.fun bounty
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).
Python standard library only; runs offline in the grader's sandbox.

## What it does

`main.py` reads one case (`/input/<id>.in`), writes one forecast
(`/output/<id>.out`): five quantiles (5/25/50/75/95%) of Twitch concurrent
viewers in the "Grand Theft Auto VI" category for each of hours 2–24 UTC.
Score is mean pinball loss on the log scale; a prize needs a lower loss
than the best house entry.

## Method — generalize, don't memorize

The practice leaderboard is dominated by entries that memorize the four
practice curves (loss ~0.0002). The final case is a genuinely new curve —
the real GTA VI launch day — so a memorized curve cannot transfer. This
entry is built to generalize from past big launches instead:

1. **Kernel-weighted log-ratio quantiles.** Each comparable launch in the
   case file contributes its release-day curve in log-ratio form
   `ln(viewers_h / viewers_1)`, weighted by a Gaussian kernel on
   `ln(hour-1 viewers)` (bandwidth 1.0): launches whose opening hour looked
   like this launch's count most.
2. **Local-linear log-ratio regression** on `[1, ln(hour-1), ln(pre-48h
   median viewers)]`, kernel-weighted the same way: the category's
   pre-release level in the last 48h before launch carries the hype signal,
   and anchoring on it is what separates the launches whose opening hour
   alone says little (a late unlock) from the ones it says a lot about.

The two members' log-medians are combined 1:2 (kernel 1, local-linear 2)
into one ensemble median. Quantiles around it come from forecaster 1's
weighted empirical ratio distribution, shrunk by 0.8 in log space (a
calibration factor fitted on leave-one-launch-out backtests, not on the
practice cases). The spread widens over the day because the ratio
distribution does.

Deliberately **not** used: the case's `streams` / `categories_hourly`
fields (null in every practice case, so no honest backtest can validate
them), and no practice-case answer is stored anywhere in this program.

## Backtest (honest, held-out)

- **Practice cases** (scored against their true SullyGnome curves; each
  case's hour-1 print verified exact): mean pinball loss **0.173**, vs the
  house persistence baseline's 0.387.
- **Leave-one-launch-out** over the 38 comparable launches in the kit:
  mean **0.403** over all 37 held-out launches, **0.262** over the 28 with
  an opening hour ≥ 10k viewers — the regime GTA VI is in (New Zealand,
  Australia and Europe all unlock before hour 1 ends).

The practice score is no guide to the final; the backtest only shows the
forecaster generalizes to unseen launches instead of memorizing them.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` in python:3.13-slim, stdlib
only, no network. Deterministic: same input bytes give the same output
bytes, every run.

## License

MIT — see [LICENSE](LICENSE).
