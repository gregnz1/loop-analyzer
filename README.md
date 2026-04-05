# Loop Analyzer

Reusable analyzer for reviewing Dexcom Clarity CSV exports together with Tidepool XLSX exports.

## What it does
- Aligns Clarity glucose data with Tidepool insulin, food, and settings data
- Compares recent and previous windows
- Supports settings-change mode
- Produces markdown report, HTML dashboard, CSV summaries, and PNG plots

## Main files
- `loop_tuneup_analyzer_v3.py`
- `notes_template.txt`
- `run_loop_tuneup.txt`

## Standard review
```bash
python loop_tuneup_analyzer_v3.py \
  --clarity-csv Clarity_Export.csv \
  --tidepool-xlsx TidepoolExport.xlsx \
  --outdir analysis_out \
  --recent-start 2026-03-06 \
  --recent-end 2026-04-04 \
  --previous-start 2026-02-04 \
  --previous-end 2026-03-05 \
  --notes-file notes_template.txt
```

## Settings-change mode
```bash
python loop_tuneup_analyzer_v3.py \
  --clarity-csv Clarity_Export.csv \
  --tidepool-xlsx TidepoolExport.xlsx \
  --outdir analysis_change \
  --change-datetime "2026-04-05 16:00" \
  --pre-days 14 \
  --post-days 14 \
  --notes-file notes_template.txt
```

## Outputs
- `report.md`
- `dashboard.html`
- CSV summaries
- PNG plots

## Notes
- New settings only appear after a fresh Tidepool export.
- Training-day tagging works best when manual training dates are supplied.
