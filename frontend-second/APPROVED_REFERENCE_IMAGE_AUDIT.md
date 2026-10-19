# Approved 10-image reference comparison

The separate approved PNGs are conceptual design references, not screenshots of a running product. Their route meanings come from [`09_REFERENCE_IMAGE_MAP.md`](../OpenAgent_HTML_Reference/docs/original-blueprint/09_REFERENCE_IMAGE_MAP.md). They are therefore useful for judging composition, hierarchy, spacing, panel density and design direction, but do not support a meaningful exact-pixel score against application captures.

## Route mapping and dimensions

The sheet maps each image to the current light desktop capture as follows:

| Ref | Approved image | Target route | Approved PNG | Target capture | Size delta (reference minus target) |
|---|---|---|---:|---:|---:|
| 01 | `01_home.png` | Home | 1448×1086 | 1440×900 | +8×+186 px |
| 02 | `02_chat.png` | Chat | 1448×1086 | 1440×900 | +8×+186 px |
| 03 | `03_agents.png` | Agents | 1448×1086 | 1440×900 | +8×+186 px |
| 04 | `04_research.png` | Research | 1448×1086 | 1440×900 | +8×+186 px |
| 05 | `05_files.png` | Files | 1448×1086 | 1440×900 | +8×+186 px |
| 06 | `06_automations.png` | Automations | 1448×1086 | 1440×900 | +8×+186 px |
| 07 | `07_calendar.png` | Calendar | 1448×1086 | 1440×900 | +8×+186 px |
| 08 | `08_models.png` | Models | 1448×1086 | 1440×900 | +8×+186 px |
| 09 | `09_settings.png` | Settings | 1448×1086 | 1440×900 | +8×+186 px |
| 10 | `10_codeq.png` | CodeQ | 1448×1086 | 1440×900 | +8×+186 px |

These are image-dimension differences, not measured CSS panel offsets. The approved concept PNGs use a taller viewport and may depict illustrative or unavailable content. The image map identifies representative structural targets such as a 252 px sidebar, 58 px top bar, and right-side panels around 340–400 px; it also cautions against copying invented details from the artwork. This sheet does not claim that those measurements were independently extracted from each conceptual PNG.

## Generated artifacts

- Side-by-side sheet, rendered without scaling or cropping: `.qa/approved-reference-10-screen-sheet.png`
- Machine-readable mapping, intrinsic dimensions and method: `.qa/approved-reference-10-screen-sheet.json`
- Generator: [`scripts/create-approved-reference-sheet.mjs`](scripts/create-approved-reference-sheet.mjs)

Images share a top-left origin on the sheet. Each is drawn at its intrinsic dimensions. A visible outline shows each image boundary; the reference PNG's unused right and bottom area remains blank where the target capture is narrower or shorter. No pixel-match percentage is computed.

Run from `frontend-second/`:

```powershell
node scripts/create-approved-reference-sheet.mjs
```

Optional environment variables select another set without modifying source images or the implementation: `OPENAGENT_APPROVED_REFERENCE_ROOT`, `OPENAGENT_APPROVED_TARGET_ROOT`, and `OPENAGENT_APPROVED_SHEET_OUTPUT`.

## Limits and interpretation

The result makes route correspondence and capture bounds easy to inspect. It is not a matched-state, same-viewport pixel regression: the conceptual images show illustrative content and different viewport dimensions from the 1440×900 target captures. Do not resize these assets and report the resulting similarity as pixel fidelity. For exact visual regression, compare the live HTML reference and React route under the same viewport, theme, browser, scroll position and equivalent state data; retain the approved PNGs as additional composition and style authority.
