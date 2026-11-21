# QUIRKS: errors reproduced on purpose, findings, and the only allowed deviations

This package re-implements the authors' code **bug-for-bug**. Every entry below is reproduced deliberately and
marked in the source with `# QUIRK(Qnn)`. Tests in `tests/quirks/` and `tests/golden/` enforce them.
"Observed" records what happened in the smoke run of 2026-11-21 (Windows 10, Python 3.11, TF 2.15.1, pandas 2.2.3).

## 1. Preserved quirks

| ID | What happens (original location) | Observed in smoke run |
|---|---|---|
| Q01 | `pre_select`: the skip-list contains `pump` and `buy`, so only `target` can match as a keyword. Returns at the first match, checking crypto, then exchange, then keyword (`keyword_filter.py:L22-L39`). | unit-tested |
| Q02 | Only messages with `from_id == None` and `_ == "Message"` are kept. The `len(message_list)>0` check sits outside the `.pkl` branch (`keyword_filter.py:L52-L67`). | ported |
| Q03 | `message_fg`: `load_obj(file)` has no directory prefix, so it opens the file relative to the cwd and raises `FileNotFoundError` with the default config (`message_fg.py:L40-L42, L64`). | FileNotFoundError (test) |
| Q04 | `message_fg` writes `./preselect_sample`, but the classifiers read `./PreSelection/preselect_sample` (`message_fg.py:L44`, `train_classifier.py:L26`). | path mismatch (test) |
| Q05 | Extended keyword list. Signals are counted after stop-word removal with three independent `if`s. `on_time` minutes `[58,59,0,1,2,28..32]`. Tabs stay in `message`, which corrupts the TSV (`message_fg.py:L54-L107`). | ported |
| Q06 | Emoji, URL and punctuation removal with the exact regexes and punctuation string (`message_fg.py:L11-L30`). | byte-identical to the original functions (test) |
| Q07 | `word_tokenize` + NLTK English stop-words (`message_fg.py:L32-L38`). | identical to the original (test) |
| Q08 | Benchmark: 3 string classes (`0`/`1`/`2`). TF-IDF fitted on **all** rows before `train_test_split(0.3, rs=42)` (leakage). Weighted metrics (`train_classifier.py`). | ported |
| Q09 | Predictor: `2→0`, left merge, fill `-1`. TF-IDF fitted on train + predict rows together. The `threshold` variable is ignored and the literal `0.25` is applied. `reset_index` concat keeps an extra `index` column (`predic_classifier.py`). | threshold override has no effect (test) |
| Q10 | Sessions split when the gap is `>=` **12h**. Only sessions with `pred_pump_cnt > 1` are kept (`sess_aggregate.py`). | **reproduces the shipped session pickles exactly** for all 160 reproducible channels (G2) |
| Q11 | `PD_label` reads `pump_attack_browsed.txt` (not shipped, so FileNotFoundError) but writes `*_test.txt` files. Window is -5/+6 messages. `?`/`r` delta arithmetic (`PD_label.py`). | FileNotFoundError (test) |
| Q12 | Message labeler: min date 2021-01-01 (pytz UTC), `##` skips the channel, unknown input is written as `0` (`pump_message_label.py`). | ported |
| Q13 | `PDlog_clean`: coin+date fill, then a fill from the previous row via `df.loc[i-1]` (KeyError when row 0 is `?`), then defaults BTC/binance. | KeyError on crafted input (test); see Q37 |
| Q14 | `kline_collect_Binance`: only the **previous** month is downloaded (31d/15d rule). The pump month is never downloaded. Bare `except` (`kline_collect_Binance.py`). | only `…-2019-05/06/07` previous-month zips (network test) |
| Q15 | `download_file`: `blocksize` is undefined without `content-length`, and only `HTTPError` is caught. The existence check is cwd-relative (`utility.py:L109-L149`). | ported |
| Q16 | `download_monthly_klines`: the checksum download sits outside the date filter (`utility.py:L9-L45`). | ported |
| Q17 | `data_process` walks `Data/data` (nothing there), reads `Data/data/unzip`, and writes `Data/data/concat` (`data_process.py`). | **0 concat files** |
| Q18 | Positive features: `timestamp_unix` in ms. Klines are read from `<repo>/Binance/data/concat` (another mismatch). After-pump block has no `try`. Pre-pump anchor `idx-30`. `*_scale` columns unused (`pos_sample_process.py`). | 1003/1003 binance rows without kline features |
| Q19 | `pre_pump_statistics`: 73 hourly buckets, volume-weighted mid price (`pos_sample_process.py:L24-L65`). | identical to the original function (test) |
| Q20 | Window features: `return = p[0]/p[w]-1`, but `price = p[w-1]`. `*_avg`/`*_sum` only for w>1. Partial writes possible. | unit-tested |
| Q21 | CoinGecko lookup key uses the **pump date**, while the data was fetched at date-3. Zero market cap and falsy social values are skipped. Output goes to `FeatGeneration/pump_sample_raw.csv`. | 1208/1335 rows matched the JSON |
| Q22 | `gen_feature` reads `feature/pump_sample_raw.csv`, while the positive stage writes `pump_sample_raw.csv`. | path mismatch (test) |
| Q23 | `neg_sample_process`: all flags False. (a) Required list read, or built on failure. (b) Previous month **never** downloaded (`date` used instead of `last_month_date`). (c) unzip/concat roots mismatched. (d) gecko branch uses `symbol2coinId` only. (e) negatives = every other listed symbol with a concat file. (f) **result never saved**. | (b) test; (f) nothing written (test) |
| Q24 | `gen_feature` reads `feature/neg_sample_raw.csv` (`na_values="None"`), which is never produced. | build-dataset fails with FileNotFoundError by default |
| Q25 | Mean imputation **per class** (positives and negatives separately: label leakage). Filters binance/BTC/`pre_1h_price` notna. | ported |
| Q26 | z-score: columns absent from negatives use positives only; all others use pos+neg **including test rows** (leakage). | ported |
| Q27 | Id maps reserve id 0 for `"0"`, others follow `np.unique` order. | ported |
| Q28 | Sequence feature string is `"".join(values)`, **no delimiter**. Negatives inherit the sequence of the pump at the same (channel, time). | `"0.5"+"2.0" → "0.52.0"` (test) |
| Q29 | Single split at 1620579621000 ms. Unseeded shuffle (now seeded, D5). Output written to `FeatGeneration/`, but training reads `feature/`. | path mismatch (test) |
| Q30 | Data loader splits feature strings **per character** and calls `string_to_number` on each. Any `.`, `-` or `e` raises at runtime. Fixed `n_coin=378` / `n_seq_feat=147`. Frozen word2vec coin table. | StringToNumber error on `"0.5"` (test) |
| Q31 | Models: DNN `hidden3` and `dropout_rate` unused. SNN/SNNTA use `[:,:,-9:]` of the char features. `tf.Variable` inside the length clip. `maxlen=50` hard-coded. | DNN/SNN/SNNTA build and train |
| Q32 | Training ends on **any** exception (`print(e); break`), including end-of-data and a partial last batch. Save interval comes from `sample_num=132314`. | "End of sequence" printed, loop ends normally |
| Q33 | Evaluation: checkpoints 15–30 only, and a missing checkpoint breaks the loop. `HitRatio_calculation` builds `DataFrame(dtype=int)`. Under the authors' pandas 1.x this **truncates every probability to 0** (verified on pandas 1.5.3), so all candidates tie and the ranking follows data order. HR@50 is printed as "HitRate@30". Groups with fewer than k candidates raise IndexError. | AUC 0.966 but HR@1 = 0 on the fixture |
| Q34 | `optimization.py` imported but never used. | verbatim copy |
| Q35 | `tmp.py`: only 16 hard-coded symbols, fetched at date-3 under the pump-date key, never saved. | opt-in, not run |
| Q36 | Telegram fetcher: `Pump_channel2` only, entity `line[13:-1]`, broadcast channels only, limits 500/500000. | opt-in, not run |

## 2. Findings about the authors' data (discovered while reproducing)

- **Q37 The shipped `PD_logs_cleaned.txt` was hand-edited after `PDlog_clean.py` ran.** Running the shipped code on the shipped `PD_logs_raw.txt` reproduces it except for:
  - 20 exchange names lower-cased (`Yobit` to `yobit`, `Binance` to `binance`, `Hotbit`, `Bittrex`). Five others were left inconsistent (`UNISWAP`, `CREX24`, `Poocoin`, `BITMART`).
  - 1 coin renamed (`Ong` to `ONG`, row 357).
  - The last raw row (`1249251103 219 LOOM yobit BTC 2020-05-19 17:00:03`) dropped.

  The test G1 pins these exact differences.
- **Q38 The shipped raw Telegram data is incomplete.**
  - 15 of the 213 channels in `pred_pump_message.csv` (5,106 of 36,696 predicted messages) have no raw pickle.
  - 12 of the 172 shipped session files, and 6 channels of the P&D log, belong to those channels.
  - So the literal `aggregate-sessions` run stops with FileNotFoundError.
  - The paper reports 715 channels, but only 512 pickles are shipped.
- **Binance audit** (`pnd verify-binance --limit 50`, pump month + previous month from data.binance.vision):
  - 50/50 archives exist.
  - 45/50 logged pumps show a ≥5% spike within 60 minutes (median peak/pre ratio 1.39, peak usually in the first minute).
  - The CoinGecko `price_btc` (date-3) agrees with Binance VWAP within a median 2.0% relative error.
  - None of the first 50 symbols is still TRADING on Binance spot today.
  - Run without `--limit` for the full log.

## 3. Allowed deviations (the only ones)

| ID | Deviation | Why |
|---|---|---|
| D1 | Output directories are created when missing. | The authors' machine already had them. |
| D2 | Telegram credentials come from `TELEGRAM_API_ID` / `TELEGRAM_API_HASH`. | No hard-coded secrets. |
| D3 | The non-existent `Statistics.utility` import becomes `pnd.data.binance.archive`. `pos_sample_process` import becomes `pnd.features.market_stats`. | The original could not import. |
| D4 | The unused module-level `CoinGeckoAPI().get_coins_list()` calls are not executed. | Binance-only data decision; the result was unused. |
| D5 | Global `seed` (default 1234) for python/numpy/TF and `tf.data.shuffle`. `seed: null` restores unseeded behaviour. The original hard-coded `random_state`s (42, 123, 10, 0) are unchanged. | Reproducibility. |
| D6 | Progress `print`s become `logger.info`. Interactive prompts are byte-identical. | Clean code. |
| D7 | Files opened without close now use `with`. | Same content. |
| D8 | `tf.flags` becomes Typer options with the same names and defaults. The `run_eval.py` flag-deletion hack becomes the `evaluate` command. | Clean CLI. |
| D9 | Unused imports without side effects are dropped (except `emoji`, `urlextract`). Unused argparse helpers are not ported. | Clean code. |
| D10 | New session pickles reference `pnd.data.telegram.models`. Old ones load via `LegacyUnpickler`. | Package layout. |
| D11 | The stop-word list is built once per call instead of once per token. | Same result, faster. |
| D12 | `astype(int)` / `dtype=int` use **int64**. | The authors ran on Linux/macOS (int = int64). On Windows with numpy<2, int = int32 overflows ms timestamps and raises. |
| D13 | Text files are read and written as **UTF-8**. | The authors' default locale. Windows' cp1252 cannot read `crypto_symbol`. |
| D14 | `HitRatio`'s `DataFrame(dtype=int)` emulates pandas-1.x semantics (numeric columns cast with truncation, byte columns kept). | pandas ≥ 2.0 raises instead; this reproduces what the authors' run computed. |
| D15 | Vocabulary, keyword and stop-word membership tests use sets instead of lists; the NLTK stop-word list is loaded once. | Identical results (pinned by a test); message features drop from about 20 min to about 2.5 min on the 1.03 M pre-selected messages. |

## 4. Paper vs. code (documentation only; the code is implemented)

| Item | Paper | Code (implemented) |
|---|---|---|
| Message classifier | RF (Table 1) | GBDT |
| Threshold | 0.2 | 0.25 |
| Session gap | 24h | 12h |
| Split | train/val/test at 2021-01-19 and 2021-05-10 | train/test at 1620579621000 ms (2021-05-09 17:00:21 UTC) |
| Windows x | {1,3,6,12,24,48,60,72} | [1,3,6,12,24,36,48,60,72] |
| Labeled messages | 5,050, two classes | 5,146 lines: 3654 `0` / 847 `1` / 630 `2` / 15 `?` |
| Metric | HR@1,3,5,10,20,30 | HR@1,3,5,10,20,**50** printed as "30" (and computed on truncated scores, Q33) |
| Channels | 715 | 512 raw pickles shipped (Q38) |
