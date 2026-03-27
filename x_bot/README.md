# X Bot

This package posts automated `MATCH PREVIEW` posts for upcoming 2026 World Cup matches.

## What it does

* selects matches whose post time is due
* target post time is `48 hours before 00:00 UTC` on the match date
* generates tweet text from the same public match and probability files used by the website
* renders a PNG preview card
* optionally posts to X with media attached
* records sent posts in `state/sent-posts.json` to avoid duplicates

## Install

Run from `x_bot/`:

```bash
npm install
```

## Usage

Dry run:

```bash
npm run dry-run
```

Post to X:

```bash
npm run post
```

The bot automatically loads credentials from `x_bot/creds` if present. Format:

```bash
X_API_KEY=...
X_API_KEY_SECRET=...
X_ACCESS_TOKEN=...
X_ACCESS_TOKEN_SECRET=...
```

## Scheduling

Run this script hourly from cron or your job runner. It checks a 1-hour due window and posts anything due in that window.

Example cron:

```cron
5 * * * * cd /path/to/repo/x_bot && /usr/bin/env npm run post >> /tmp/world-cup-x-bot.log 2>&1
```

## One-off test post

For a controlled first live post, target a specific match id:

```bash
npm run dry-run -- --match-id=4
npm run post -- --match-id=4
```

## Notes

* Group-stage matches post 90-minute win/draw/win probabilities only.
* Knockout and qualifier matches post both `After 90` and `To qualify`.
* Placeholder teams such as `UEFA Path C winner` are skipped until resolved team names exist.
* The image renderer uses local flag PNGs and current website output files in `web/public/`.
