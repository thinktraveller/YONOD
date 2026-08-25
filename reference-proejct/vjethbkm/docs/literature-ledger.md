# VJETHBKM literature ledger

Last updated: 2026-08-25

This ledger records the evidence used to reproduce Zotero item `VJETHBKM`.
It intentionally stores citations, checksums, access notes, and reproduction
decisions instead of copying copyright-sensitive article files into Git.

## Source records

| id | type | source | identifier | access status | redistribution |
|---|---|---|---|---|---|
| vjethbkm-main | article | ACS Publications | https://doi.org/10.1021/jacs.6c02213 | Open Access landing page checked on 2026-08-25 | Article is ACS Open Access under CC BY 4.0 according to the ACS landing page; keep local PDFs out of Git unless license and size are rechecked. |
| vjethbkm-crossref | metadata | Crossref | https://doi.org/10.1021/jacs.6c02213 | Metadata queried on 2026-08-25 | Metadata can be stored as citation facts. |
| vjethbkm-si | supporting information | ACS Publications / Crossref | https://doi.org/10.1021/jacs.6c02213.s001 | SI DOI found through Crossref on 2026-08-25 | Store DOI and target mapping; do not commit downloaded PDF/SI by default. |
| vjethbkm-data-code | data and code | ETH Research Collection | https://doi.org/10.3929/ethz-c-000800856 | DOI recorded from article data availability statement; direct package inspection still pending | Commit only manifest/checksum/schema unless license and size allow data files. |
| vjethbkm-zaihub | Zotero / Z-AIHub | Zotero key `VJETHBKM` | local item key | Previous main-thread Z-AIHub attachment-level reading is available in the conversation history; this builder turn could not re-open Z-AIHub because no callable Z-AIHub/Zotero tool was exposed after tool discovery | Use previous Z-AIHub document IDs as provenance; keep raw PDFs/SI out of Git by default. |

## Verified article facts

| fact | evidence source | reproduction impact |
|---|---|---|
| Title: "Yield Smarter, Not Harder: Good Practices for Machine Learning of Reaction Outcomes" | ACS landing page and Crossref metadata | Names this reproduction target and report heading. |
| Authors: Idil Ismail, Gregory A. Landrum, Sereina Riniker | ACS landing page and Crossref metadata | Used in citation and report metadata. |
| Journal/date: Journal of the American Chemical Society, published online 2026-07-22 | ACS landing page and Crossref metadata | Used for citation and versioned literature audit. |
| DOI: `10.1021/jacs.6c02213` | ACS landing page and Crossref metadata | Primary article identifier. |
| Supporting information DOI: `10.1021/jacs.6c02213.s001` | Crossref metadata | Primary SI identifier for later PDF/chunk audit. |
| Data and source code are stated to be available at ETH Research Collection DOI `10.3929/ethz-c-000800856` | ACS data/code availability statement | Primary target for data and code retrieval. |
| Benchmarked reaction classes include Buchwald-Hartwig amination, Suzuki-Miyaura coupling, and SLAP | ACS abstract and plan audit | Drives dataset manifest placeholders and split design. |
| Main methodological themes include descriptor complexity, component-wise generalization, external validation, and asymmetric yield distributions | ACS abstract/introduction and SI contents | Drives the stage A/B/C implementation plan. |

## Evidence provenance layers

The evidence base is split into two layers so later work can tell which facts
came from a direct Zotero/Z-AIHub reading and which facts were rechecked in the
current builder turn.

| layer | status | evidence details | implementation use |
|---|---|---|---|
| Previous main-thread Z-AIHub attachment reading | Available from inherited conversation history, not re-executed in this builder turn | Zotero item key `VJETHBKM`; main article `documentId=doc_9c01f86b3093`, 11 pages, 43 chunks; SI `documentId=doc_a270fcc24f77`, 13 pages, 37 chunks | Treats the descriptor/model/split/result map as attachment-level evidence, while still requiring official-file checksums before numerical reproduction claims. |
| Current builder-turn public-source recheck | Completed on 2026-08-25 | ACS/Crossref verified article DOI `10.1021/jacs.6c02213`, SI DOI `10.1021/jacs.6c02213.s001`, title, authors, publication venue/date, Open Access note, and data/code DOI `10.3929/ethz-c-000800856` | Confirms citation metadata and public resource identifiers used by manifests and reports. |
| Current builder-turn Z-AIHub access | Blocked by current tool surface | Tool discovery for `zaihub`, `Z-AIHub`, `Zotero`, `VJETHBKM`, `document chunks`, and `knowledge base` did not expose a callable Z-AIHub/Zotero document reader | Do not claim a fresh Z-AIHub read in this turn; keep the limitation visible. |

## Previous Z-AIHub reading summary

The previous main-thread Z-AIHub reading found the following reproduction-relevant
items in the main article and SI. This section records the provenance so the
implementation can be guided by attachment-level evidence without pretending
that the current builder turn re-read the chunks.

| topic | previous Z-AIHub evidence | implementation consequence |
|---|---|---|
| Descriptor families | DFT, SOAP, PhysChem, count-based Morgan fingerprints, and OHE were identified from the article/SI | Stage A starts with OHE/Morgan/PhysChem; DFT/SOAP remain registered high-cost extensions. |
| Model matrix | RF was the primary model; LightGBM, KNN, and Ridge were supplementary comparisons | Stage A implements RF first; configs reserve later extension slots. |
| Evaluation protocol | 5 x 5 CV, MAE, RMSE, R2, and Kendall tau were identified | Smoke benchmark already outputs these metrics; full 5 x 5 awaits official data. |
| Generalization tests | 0D/1D/2D component splits were identified | Split manifest design must preserve component identity and leakage checks. |
| External validation | BH2 external validation with 187 records was identified | BH2 must remain a true external holdout when official files are available. |
| Imbalance/reweighting | Yield imbalance analysis and reweighting experiments were identified; previous reading noted no statistically significant improvement from reweighting | Stage C should report global metrics, yield-bucket metrics, and the trade-off rather than assuming reweighting helps. |
| Numerical SI targets | Table S3 was previously read as containing exact four-dataset metrics | Exact values still need to be materialized into a tracked target table before claiming numerical reproduction. |

## Supporting information targets

The ACS page exposes the SI contents list. The first audit pass maps these
sections to implementation work without assuming full numerical values yet.

| SI section | topic | implementation target |
|---|---|---|
| S1 | Reaction yield datasets | `data/manifest/sources.yaml`, `schema.yaml`, `dataset_stats.csv` generation. |
| S2.1 | HTE datasets | Dataset registry and component-column mapping. |
| S2.2.1 | DFT-derived descriptors | Mark as high-cost/precomputed-preferred feature source. |
| S2.2.2 | SOAP | Mark as extension feature source requiring ASE/DScribe and geometry assumptions. |
| S2.2.3 | 2D physicochemical descriptors | Stage A descriptor baseline using RDKit descriptors where available. |
| S2.2.4 | Count-based Morgan fingerprints | Stage A descriptor baseline using count Morgan features. |
| S2.2.5 | One-hot encoding | Stage A categorical component baseline. |
| S2.3 | RF regression training details | Stage A primary model baseline. |
| S2.4 | Other ML algorithms | Stage B/C model extension registry. |
| S2.5 | Comparing model performance | Metrics registry: MAE, RMSE, R2, Kendall tau, plus runtime. |
| S3 | Additional RF cross-validation results | Stage B target table mapping. |
| S4 | Classical ML algorithm comparison | Stage C model matrix target. |
| S5 | Reweighting experiments | Stage C imbalance and high-yield analysis. |

## Z-AIHub audit note

The current Codex session exposed literature search tools but did not expose a
direct Z-AIHub/Zotero document reader after searching for `zaihub`, `Z-AIHub`,
`Zotero`, `VJETHBKM`, `document chunks`, and `knowledge base`.

This is an access limitation of the current tool surface, not a decision to
skip the evidence workflow. Because prior main-thread Z-AIHub evidence exists,
the first reproducible implementation therefore:

- records public article/SI/data-code identifiers;
- records the prior Z-AIHub document IDs and chunk counts separately from the
  current public-source recheck;
- keeps placeholder fields for future local attachment hashes;
- treats figure/table numerical values as pending materialization into tracked
  files, even when the previous thread read them;
- allows smoke-stage engineering work to proceed without claiming full paper
  reproduction.

## Local-file manifest placeholders

| expected local file | status | sha256 |
|---|---|---|
| `data/raw/articles/VJETHBKM-main.pdf` | not committed; pending user-provided or connector-fetched file | pending |
| `data/raw/articles/VJETHBKM-si.pdf` | not committed; pending user-provided or connector-fetched file | pending |
| `data/raw/ethz-c-000800856/` | not committed; pending official package inspection | pending |
