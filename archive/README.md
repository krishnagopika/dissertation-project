# Archive

Superseded files, kept for provenance. Nothing here is referenced by the
dissertation or by any script in `src/`.

## results_tables/
Earlier passes of `src/evaluation/evaluate_all.py`. **The current table is
`results_new/test_all/test_all_v7_windowed_attnraw.json`** — every number in the
write-up that comes from a re-scoring pass comes from that file.

- `test_all.json` … `test_all_v5.json` — scored windowed runs (`k0`–`k4`) without
  re-applying the training-time context window, so those cells describe a
  different model. `attnraw_*` runs were additionally fed the masked-mean cache
  instead of `meld_pooled_raw/attention`.
- `test_all_v6_windowed.json` — windowing fixed, `attnraw_` cache still wrong.
- `v7` fixes both and passes the bc-LSTM validation gate
  (`src/evaluation/check_windowed_eval.py`): 62/62 comparable cells reproduce the
  test metrics written at training time.

Superseded write-up material (old figures, the interim report) lives under
`report/_archive/`, which is outside version control along with the rest of
`report/`.

## code_backups/
`evaluate_all.py.bak` — pre-fix copy, before the context-window and `attnraw`
cache corrections.
