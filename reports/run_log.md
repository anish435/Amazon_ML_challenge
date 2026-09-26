## Step 1 Preprocessing Execution & Validation Report
**Date/Time**: 2026-09-26 16:14:27
**Total Preprocessing Runtime**: 181.45s (3.02 min)

### Hardware Environment
- **CPU Logical Cores**: 20
- **Total RAM**: 15.64 GB (Available: 1.09 GB)
- **Disk Space (D:)**: 34.7 GB free of 94.94 GB

### 1. Parquet Verification & Column Cardinality
| File | Raw Rows | Prep Rows | Match | Size (MB) | Time (s) | Null/Empty Counts |
|---|---:|---:|:---:|---:|---:|---|
| `train_s1.parquet` | 2,206,821 | 2,206,821 | PASS | 281.89 | 10.79 | postal: 2059562 null / 0 empty; house_nums: 178598 null / 0 empty |
| `train_s2.parquet` | 5,034,616 | 5,034,616 | PASS | 679.58 | 35.67 | addr_raw: 168967 null / 0 empty; addr_norm: 0 null / 168967 empty; postal: 4665473 null / 0 empty; house_nums: 736670 null / 0 empty |
| `train_s3.parquet` | 5,285,603 | 5,285,603 | PASS | 700.89 | 39.67 | addr_raw: 175916 null / 0 empty; addr_norm: 0 null / 175916 empty; postal: 4899870 null / 0 empty; house_nums: 740371 null / 0 empty |
| `test_s1.parquet` | 1,732,544 | 1,732,544 | PASS | 224.71 | 11.88 | postal: 1656910 null / 0 empty; house_nums: 122803 null / 0 empty |
| `test_s2.parquet` | 4,887,273 | 4,887,273 | PASS | 680.67 | 41.07 | addr_raw: 129408 null / 0 empty; addr_norm: 0 null / 129408 empty; postal: 4639347 null / 0 empty; house_nums: 531360 null / 0 empty |
| `test_s3.parquet` | 5,082,316 | 5,082,316 | PASS | 686.7 | 42.37 | addr_raw: 136098 null / 0 empty; addr_norm: 0 null / 136098 empty; postal: 4826264 null / 0 empty; house_nums: 546769 null / 0 empty |

### 2. Normalization Quality by Country Partition
| Split | Country | Total Rows | Raw Non-ASCII (%) | Prep Non-ASCII (%) | With Postal (%) | With House Nums (%) |
|---|---|---:|---:|---:|---:|---:|
| TRAIN S1 | India | 883,188 | 554 (0.06%) | 0 (0.00%) | 2,386 (0.27%) | 805,661 (91.22%) |
| TRAIN S1 | US | 1,323,633 | 0 (0.00%) | 0 (0.00%) | 144,873 (10.95%) | 1,222,562 (92.36%) |
| TEST S1 | France | 259,452 | 101,001 (38.93%) | 0 (0.00%) | 1,039 (0.40%) | 258,249 (99.54%) |
| TEST S1 | India | 809,986 | 465 (0.06%) | 0 (0.00%) | 2,068 (0.26%) | 738,768 (91.21%) |
| TEST S1 | US | 663,106 | 0 (0.00%) | 0 (0.00%) | 72,527 (10.94%) | 612,724 (92.40%) |

### 3. S1 Duplicate Rate (name_core + addr_norm)
Rate of S1 entities sharing an identical `(name_core, addr_norm)` with another S1 row within the same country partition, alongside individual name and address collision rates.
| Split | Country | S1 Entities | Duplicate (Name+Addr) Rows | Dup (Name+Addr) Rate | Dup name_core Rate | Dup addr_norm Rate |
|---|---|---:|---:|---:|---:|---:|
| TRAIN | India | 883,188 | 2 | **0.00%** | 53.29% | 4.88% |
| TRAIN | US | 1,323,633 | 2 | **0.00%** | 46.61% | 5.62% |
| TEST | France | 259,452 | 59 | **0.02%** | 47.44% | 11.73% |
| TEST | India | 809,986 | 2 | **0.00%** | 52.57% | 4.77% |
| TEST | US | 663,106 | 2 | **0.00%** | 38.86% | 4.04% |

### 4. Before / After Transformation Examples (20 Samples)
| Country | Raw Name | Normalized Name | Core Name | Raw Address | Normalized Address | Postal | House Nums |
|---|---|---|---|---|---|---|---|
| France | << Team Ecole | team ecole | team ecole | 175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine | 175 blvd du president franklin roosevelt bordeaux nouvelle aquitaine | - | 175 |
| France | Maison de Santé Generation | maison de sante generation | maison de sante generation | 30 Rue Louis Thénard, Saint-Nazaire, Pays de la Loire | 30 rue louis thenard saint nazaire pays de la loire | - | 30 |
| France | Saint-Herblain Societe SARL | saint herblain societe sarl | saint herblain | 5 Impasse Jean Baptiste Clément, Saint-Herblain, Pays de la Loire | 5 imp jean baptiste clement saint herblain pays de la loire | - | 5 |
| France | Securite Darts Sport SARL | securite darts sport sarl | securite darts sport | 21 Chemin des Réunis, Nantes, Pays de la Loire | 21 che des reunis nantes pays de la loire | - | 21 |
| France | École primaire Sainte Pierre | ecole primaire sainte pierre | ecole primaire sainte pierre | 22 RUE Descartes, Hauts-de-France, Calais | 22 rue descartes hauts de france calais | - | 22 |
| France | Rotary Sport SASU | rotary sport sasu | rotary sport | 7 Place de Suède, Lille, Hauts-de-France | 7 pl de suede lille hauts de france | - | 7 |
| France | Bouliste (France) Amicale SARL | bouliste france amicale sarl | bouliste france amicale | 267 Rue de la Renaudière, Nantes, Pays de la Loire | 267 rue de la renaudiere nantes pays de la loire | - | 267 |
| India | Red Perfect Trading | red perfect trading | red perfect trading | Mirzapur, Ews 12, Uttar Pradesh, Mirzapursadar, Awas Vikas Colony | mirzapur ews 12 uttar pradesh mirzapursadar awas vikas colony | - | 12 |
| India | Nandlal Kisan LLP | nandlal kisan llp | nandlal kisan | D-61 Ifs Apartmentmayur Vihar I, New Delhi, East Delhi, Delhi | d 61 ifs apartmentmayur vihar i new delhi east delhi delhi | - | 61 |
| India | Om Constructions Pvt Ltd | om constructions pvt ltd | om constructions | Karauli, Rajasthan, Karauli, Pani Ki Tanki Ke Pas Choubepada | karauli rajasthan karauli pani ki tanki ke pas choubepada | - | - |
| India | Roongta Sangh | roongta sangh | roongta sangh | Bhubaneswar, Sub Plot No.-L6/29, Mahodadhi Bhawan (Next To Iter College) Panchasakha Nagar, Dumuduma, Khordha, Plot No.-780, Orissa | bhubaneswar sub plot no l6 29 mahodadhi bhawan next to iter college panchasakha nagar dumuduma khordha plot no 780 orissa | - | l6 29 780 |
| India | Anupama Estates Pvt Ltd | anupama estates pvt ltd | anupama estates | 26, Ram Nagar, Behind M.C Quarters Near Jagadhri Gate, Ambala City, Haryana | 26 ram nagar behind m c quarters nr jagadhri gate ambala city haryana | - | 26 |
| India | Siliguri Media Pvt Ltd | siliguri media pvt ltd | siliguri media | S No. 42/2/3, H No. B/5, Santkrupa Niwas, Ramnagar, Pune City, Pune, Maharashtra | s no 42 2 3 h no b 5 santkrupa niwas ramnagar pune city pune maharashtra | - | 42 2 3 5 |
| US | Zephay Labs Inc | zephay labs inc | zephay labs | 2621 Cotten Road, Tyler, TX | 2621 cotten rd tyler tx | - | 2621 |
| US | Vision Partners Corp | vision partners corp | vision partners | IA, Iowa City, 1064 Newton Rd, Unit 11 | ia iowa city 1064 newton rd unit 11 | - | 1064 11 |
| US | Cure Seafood | cure seafood | cure seafood | 1325 Brooklyn Walk, Issaquah, WA | 1325 brooklyn walk issaquah wa | - | 1325 |
| US | Blinny Gordon Apex Rate | blinny gordon apex rate | blinny gordon apex rate | 100 Hartsdale Avenue, Unit Apartment TPE, Greenburgh, NY | 100 hartsdale ave unit apt tpe greenburgh ny | - | 100 |
| US | Garcia Mercury Corp | garcia mercury corp | garcia mercury | 600 SE 2nd St, Pelican Rapids, MN | 600 se 2nd st pelican rapids mn | - | 600 2nd |
| US | Delta Partners Inc | delta partners inc | delta partners | 2202 Brandywood Drive, Murfreesboro, TN | 2202 brandywood dr murfreesboro tn | - | 2202 |
| US | Midwest Alliance LLC | midwest alliance llc | midwest alliance | 1045 Lazy Acres Road, Protem, MO | 1045 lazy acres rd protem mo | - | 1045 |

## Step 2 Pilot Findings & Speed Bottleneck Audit (Pre-Redo)
**Date/Time**: 2026-09-26 18:55:00
- **Pilot Setup**: 500 India + 500 US queries against full candidate pools (4.13M India, 6.18M US).
- **Throughput Measured**:
  - Channel A (Combo char 3-gram, top-30): ~14.5 q/s
  - Channel B (Name char 3-gram, top-20): ~32.0 q/s
  - Channel C (Addr char 3-gram, top-15): ~22.5 q/s
  - Channel D (Key Join): ~2,500 q/s
  - Effective Multi-Channel Throughput: ~8.5 q/s
- **500 India Recall**:
  - Channel A: 92.65%, B: 54.56%, C: 81.39%, D: 47.89%
  - Union A+B: 94.35%, Union A+B+C: 94.77%, Union A+B+C+D: 94.95%
  - Candidate density: ~54.9 cands/entity
- **Root Cause of Bottleneck**:
  1. No max_df pruning: top trigrams present in 1.95M candidates (248M non-zeros in candidate matrix).
  2. Single-threaded query transformation and under-utilized cores (average CPU ~5%).
  3. Character-level n-grams for address and combo caused extreme matrix density.
- **Action Plan**: Pivot to full-core utilization (20 cores), parallel TF-IDF transforms via joblib, max_df=0.02, word-level TF-IDF for combo and address, vectorized Polars key-join, and GPU dense embeddings (if available).

## Step 2 (REDO) Fast Blocking — Benchmark & Gate Decision
**Date/Time**: 2026-09-26 19:46:55
**Configuration**: 2,000 Validation queries per country from buckets 80–89 against FULL candidate pools (4.13M India, 6.18M US).

### 1. Channel Performance & Candidate Density
| Country | Channel A (Word Combo) | Channel B (Char Name) | Channel C (Word Addr) | Channel D (Polars Key) | Union A+B | Union A+B+C | Union A+B+C+D | Mean Cands/S1 | p90 Cands/S1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **India** | 92.57% (465 q/s) | 46.20% (612 q/s) | 81.67% (490 q/s) | 47.43% (2,500 q/s) | 93.43% | **93.98%** | 94.14% | **51.4** | 57.0 |
| **US** | 97.84% (425 q/s) | 60.22% (488 q/s) | 84.89% (635 q/s) | 54.89% (2,500 q/s) | 98.18% | **98.45%** | 98.47% | **51.7** | 58.0 |
| **WEIGHTED** | 95.73% | 54.61% | 83.60% | 51.91% | 96.28% | **96.66%** | 96.73% | **51.6** | - |

### 2. Gate Decision & Channel Selection
- **Gate 1: Union Recall $\ge 95\%$**: **PASSED** (96.66% weighted recall across India and US; threshold: $\ge 95\%$)
- **Gate 2: $\le 80$ Candidates/Entity**: **PASSED** (51.6 candidates/entity; threshold: $\le 80$)
- **Channel Selection Rules**: Adding Channel D to Union A+B+C adds only **+0.08%** recall (< 1% threshold). Per prompt rules, **Channel D is dropped**. Channels **A (Word Combo)**, **B (Char Name)**, and **C (Word Addr)** are selected.
- **Gate 3: Test Runtime $\le 2$ Hours**: **FAILED** (Real test runtime across all 1,732,544 test S1 queries was 11,861s = 3h 17m 41s: France 04m 47s, India 02h 05m 47s, US 01h 07m 07s. Blocking proceeded to full completion with 100% S1 coverage and 88,679,129 candidate pairs).

### 3. Audit of 20 Missed True Matches (Sample)
| Country | S1 ID | S1 Raw Name | S1 Raw Address | Cand ID | Candidate Raw Name | Candidate Raw Address | Failure Diagnostic |
|---|---|---|---|---|---|---|---|
| India | `S1-818165388` | Eastern Infotech Private Limited | D-4, 12D, Sector -82, Noida, Gautam Buddha Nagar, Uttar Pradesh | `S2-897365524` | ईस्टर्न इंफोटेक प्राइवेट लिमिटेड | D-4, NOIDA, GAUTAM BUDDHA NAGAR, Uttar Pradesh | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| India | `S1-407554487` | Spine Sangh | 609, Sushant Lok Phase 1, Gurugram, Gurgaon, Haryana | `S3-619134355` | Spine Sfh | - | Candidate or query has missing/empty address string |
| India | `S1-324165385` | Super Care Private Limited | House No 234 Kh. No 11/23/2 & 11/24/2, Ground Floor Vardhman Enclave Blk-F Village Karala Landmark Near Water Tank, Delhi, North West Delhi, Delhi | `S2-417501373` | सुपर केयर प्राइवेट लिमिटेड | H.NO #36 HOUSE NO 234 KH. NO 11/23/2 & 11/24/2, DELHI | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| India | `S1-431101282` | North Builders Private Limited | Gopala Krishna Complex C/O 91 Springboard Business No 45/3 Residency Road, Bangalore, Karnataka | `S3-353215454` | ನಾರ್ತ್ ಬಿಲ್ಡರ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | Gopala Krishna Complex C/o 091 Springboard Business No 45/3 Residency Road, Bangalore, KA | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| India | `S1-431101282` | North Builders Private Limited | Gopala Krishna Complex C/O 91 Springboard Business No 45/3 Residency Road, Bangalore, Karnataka | `S3-936355823` | @Northbuilders | Gopala Krishna Complex C/o 091 Springboard Business No 45/3 Residency Road, Bangalore, KA | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| India | `S1-217771786` | Indo Foods Limited | 132 District Center, Meera Chowk, Sriganganagar, Sriganga Nagar, Rajasthan | `S2-391326010` | इंडो फूड्स लिमिटेड | H.NO 132 DISTRICT CENTER, SAWAI MADHOPUR, SRIGANGA NAGAR, राजस्थान | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| India | `S1-475902387` | Star Products Pvt Ltd | S.No.37/9/4, Bijalinagar, Kalepadal, Hadapsar, Pune, Maharashtra | `S2-418968972` | स्टार प्रोडक्ट्स प्रा. लि. | S.NO.7/9/4, PUNE, Maharashtra | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| India | `S1-549037888` | Megha Farmers Private Limited | Flat No. 209, B-Block, Foyer City Apartments, Doddathogur, Electronics City 3Rd Cross, Green House Layout, Bangalore South, Bangalore, Karnataka | `S3-206771300` | meghafarmers.com | Bangalore South, KA, Door No 209 | Substantial spelling divergence in street name |
| India | `S1-684819088` | Khusboo Charitable Trust | Ward No. 5, Gram - Ghoti Lalburra, Balaghat, Madhya Pradesh | `S3-225010094` | Iriecto | Ward No. 5, Balaghat, MP | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| India | `S1-871207380` | Gayatri Services Center | Delhi, Plot No. D-4, West Delhi, Kh. No. 104/14/1 Gali No.5, Rajapuri, Uttam Nagar | `S3-220532536` | Gayatri Services  Cntre | Block B-569- Plot No. D-4, Divreportingcircle, West Delhi, DL | Substantial spelling divergence in street name |
| US | `S1-979786585` | Sneed Health Center LLC | 450 M Street, Unit 4, Casper, WY | `S3-368582854` | Veoonyxaria | 0450 M Street, Wyoming, Casper, Unit 4 | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| US | `S1-979786585` | Sneed Health Center LLC | 450 M Street, Unit 4, Casper, WY | `S3-882900633` | Sneed Htalh Center LLC | - | Candidate or query has missing/empty address string |
| US | `S1-42920781` | Interstate Institute PLLC | 1009 15th Street, Mayfield, KY | `S3-980705045` | Interstateinstitutecom | Fifteenth St, Mayfield DP, Kentucky | Substantial spelling divergence in street name |
| US | `S1-172424185` | Rocky Safety Inc | 14 Rabbit Run Road, Rainsville, AL | `S2-425945826` | ROCKY SAIFY INC | - | Candidate or query has missing/empty address string |
| US | `S1-687933681` | Urban Nails | Fl Second Floor, PA, 5442 Shannondell Drive, Lower Providence Township | `S2-824821517` | NURBAN.COM | 5442 SHANNONDELL DR, NORRISTOWN, PA | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| US | `S1-86314785` | Regional Services LLC | 21 Family Lane, Hempstead, NY | `S2-171927863` | Regional Sevnihsc Sevnihsc LLC | 21 FAMILY LANE, PMB 5517, LEVITTWN TOWNSHIP, NY | Substantial spelling divergence in street name |
| US | `S1-767678483` | Schwab Complete Iron Works LLC | 11576 Old Richmond Road, VA, Pittsylvania County | `S3-121652984` | Fayeflux | 91576 Old Richmond Rd, Pittsylvania County, Virginia | Legal trade alias / DBA mismatch without shared 3-gram prefix |
| US | `S1-123036387` | Pinnacle, Corp | Dublin, Unit 168, 5043 Tuttle Crossing Boulevard, OH | `S3-825307433` | Pinnacle, Córp | - | Candidate or query has missing/empty address string |
| US | `S1-531743085` | Rocky Coalition Inc | 33548 Nichols Road, Londonderry, OH | `S2-106838314` | Rocky Cialiton-Inc | 3354 NICHOLS RD, LODONDERRY, OH | Substantial spelling divergence in street name |
| US | `S1-792214586` | Smart Wellness Technologies Inc | 49 Greenlawn Road, Brookhaven, NY | `S3-171009861` | Gildhalo | Greenlawn Rd, Sound Beach, New York | Legal trade alias / DBA mismatch without shared 3-gram prefix |

## Step 2 Full Multi-Channel Production Blocking Report
**Date/Time**: 2026-09-26 23:48:20
**Channels**: A (Word Combo top-30), B (Char Name top-20), C (Word Addr top-15). Channel D dropped per Gate.

### 1. Dataset Partition Statistics & S1 Entity Coverage
| Split | Country | Total S1 | S1 with Cands | S1 Zero Cands | Coverage (%) | Blocked Pairs | Mean Cands/S1 | p90 Cands/S1 | Max Cands | Runtime | Peak RSS | Parquet Size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TEST | France | 259,452 | 259,449 | 3 | **100.00%** | 14,065,048 | 54.2 | 60.0 | 65 | 04m 47s | 2074.0 MB | 236.6 MB |
| TEST | India | 809,986 | 809,986 | 0 | **100.00%** | 41,115,377 | 50.8 | 57.0 | 65 | 02h 05m 47s | 2908.5 MB | 725.4 MB |
| TEST | US | 663,106 | 663,106 | 0 | **100.00%** | 33,498,704 | 50.5 | 57.0 | 65 | 01h 07m 07s | 1923.7 MB | 598.7 MB |
| TRAIN | India | 60,030 | 60,030 | 0 | **100.00%** | 3,078,306 | 51.3 | 57.0 | 65 | 10m 24s | 560.7 MB | 52.2 MB |
| VAL | India | 8,004 | 8,004 | 0 | **100.00%** | 411,096 | 51.4 | 57.0 | 65 | 02m 37s | 598.6 MB | 5.8 MB |
| HOLDOUT | India | 4,002 | 4,002 | 0 | **100.00%** | 205,662 | 51.4 | 57.0 | 65 | 02m 08s | 514.1 MB | 2.8 MB |
| TRAIN | US | 89,970 | 89,970 | 0 | **100.00%** | 4,645,060 | 51.6 | 58.0 | 65 | 18m 28s | 720.1 MB | 81.5 MB |
| VAL | US | 11,996 | 11,996 | 0 | **100.00%** | 620,042 | 51.7 | 58.0 | 65 | 04m 32s | 742.4 MB | 9.1 MB |
| HOLDOUT | US | 5,998 | 5,998 | 0 | **100.00%** | 310,045 | 51.7 | 58.0 | 65 | 03m 17s | 502.0 MB | 4.4 MB |

- **Total TEST S1 Verified**: **1,732,544** (Matches raw test_source1.tsv exactly: France 259,452 + India 809,986 + US 663,106 = 1,732,544).
- **Total TEST Pairs Generated**: **88,679,129** candidate pairs.

### France Unlabeled Test Candidate Channel Breakdown
- **Channel A (Word Combo)**: 7,781,903 candidate pairs
- **Channel B (Char Name)**: 5,073,262 candidate pairs
- **Channel C (Word Addr)**: 3,887,138 candidate pairs

### 2. Storage & System Status
- **Total Disk Used by data/blocks/**: **1716.6 MB**
- **Free Disk Space on D: Drive**: **32.33 GB**


### 3. Full Validation (20,000 S1) & Holdout (10,000 S1) Recall Audit
Exact ground-truth match recall on blocked candidates across validation and holdout splits:

| Split | Country | S1 Entities | Ground Truth Matches | Channel A (Combo) | Channel B (Name) | Channel C (Addr) | **Union (A+B+C)** | Candidates / S1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **VAL** | India | 8,004 | 27,828 | 92.63% | 46.69% | 81.52% | **94.09%** | 51.4 |
| **VAL** | US | 11,996 | 41,229 | 97.71% | 60.72% | 84.08% | **98.37%** | 51.7 |
| **VAL** | **Weighted** | **20,000** | **69,057** | **95.68%** | **55.11%** | **83.06%** | **96.66%** | **51.6** |
| **HOLDOUT** | India | 4,002 | 13,872 | 92.10% | 46.14% | 81.63% | **93.60%** | 51.4 |
| **HOLDOUT** | US | 5,998 | 20,707 | 97.62% | 59.97% | 84.16% | **98.30%** | 51.7 |
| **HOLDOUT** | **Weighted** | **10,000** | **34,579** | **95.41%** | **54.44%** | **83.15%** | **96.42%** | **51.6** |

- **Validation Recall**: **96.66%** ($\ge 95\%$ target exceeded).
- **Holdout Recall**: **96.42%** ($\ge 95\%$ target exceeded).
- **Candidate Volume**: Strictly bounded at **51.6 candidates/entity** (mean) and **57.0–58.0** (p90), well below the 80 max constraint.

## Step 3 Feature Engineering Execution Report
**Date/Time**: 2026-09-27 00:58:09
**Total Features**: 38 float32 features
**Overall Speed**: 19,130 pairs/sec at 20 threads (Total runtime: 484.6s = 8.08 min)

### 1. Feature Generation Throughput by Split & Country
| Split | Country | Pairs | Time (s) | Pairs/Sec | Parquet Size (MB) | Peak RSS (MB) | NaN/Inf Count |
|---|---|---:|---:|---:|---:|---:|---:|
| VAL | India | 411,096 | 12.5s | 32,991 | 35.9 MB | 1685.9 MB | 0/0 |
| HOLDOUT | India | 205,662 | 6.9s | 29,604 | 17.8 MB | 1664.3 MB | 0/0 |
| TRAIN | India | 3,078,306 | 116.7s | 26,387 | 277.6 MB | 2139.4 MB | 0/0 |
| VAL | US | 620,042 | 19.3s | 32,102 | 53.0 MB | 1632.0 MB | 0/0 |
| HOLDOUT | US | 310,045 | 8.2s | 37,689 | 26.2 MB | 1843.8 MB | 0/0 |
| TRAIN | US | 4,645,060 | 154.2s | 30,131 | 410.3 MB | 2395.8 MB | 0/0 |

### 2. Extrapolation for 88.7M Test Pairs
- **Estimated Test Generation Time**: **0.78 hours** (~46.9 min at ~31,484 pairs/sec)
- **Estimated Test Features Disk Size**: **7.67 GB** (based on 92.8 bytes/pair compressed)
- **Free Disk Space Available**: **31.46 GB** (Sufficient: 7.67 GB << 31.46 GB free)


## Step 4 Training & Model Evaluation Report
**Date/Time**: 2026-09-27 01:32:59
**Selected Model Architecture**: Stage 2 (xgboost_gpu)
**Optimal Threshold (tau)**: **0.67**

### 1. Model Benchmark: LightGBM CPU vs XGBoost GPU (1M Sample)
| Library | Device / Threads | Training Time | Validation AUC | Selected |
|---|---|---:|---:|:---:|
| LightGBM | CPU (20 threads) | 37.46s | 0.99873 | NO |
| XGBoost | GPU (NVIDIA RTX 3050 CUDA) | 15.26s | 0.99881 | YES |

### 2. Stage 1 vs Stage 2 Validation Comparison
- **Stage 1 (Pair Model, 38 features)**: Val Macro F0.5 = **0.93994** at tau = 0.67 (Val AUC: 0.99908, 2-Fold OOF AUC: 0.99901)
- **Stage 2 (Context Model, 44 features)**: Val Macro F0.5 = **0.94359** at tau = 0.67 (Val AUC: 0.99922)
- **Context Model Gain**: **+0.00365** macro F0.5.
- **Greedy 1:1 Exact Ties**: 0 on validation, 0 on holdout.

### 3. Final Performance Metrics (Validation vs Holdout)
| Split | Country | Macro F0.5 | Precision | Recall | % Empty Pred | % True Singletons | Avg Pred Matches | Avg True Matches |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Validation** | **OVERALL** | **0.94359** | 0.9773 | 0.8977 | 6.71% | 5.59% | 3.17 | 3.45 |
| Validation | India | 0.92977 | - | - | - | - | - | - |
| Validation | US | 0.95281 | - | - | - | - | - | - |
| **Holdout** | **OVERALL** | **0.94528** | 0.9781 | 0.8969 | 6.47% | 5.48% | 3.17 | 3.46 |
| Holdout | India | 0.92981 | - | - | - | - | - | - |
| Holdout | US | 0.95560 | - | - | - | - | - | - |

### 4. Tau Threshold Sensitivity Curve (Validation Macro F0.5)
| Tau | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.98 | **Best (0.67)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Macro F0.5** | 0.93957 | 0.94274 | 0.94329 | 0.94069 | 0.92985 | 0.91449 | 0.88607 | **0.94359** |

### 5. Top 15 Feature Importances
| Rank | Feature | Importance |
|---|---|---:|
| 1 | `p1_rank` | 0.6 |
| 2 | `p1_rank_src` | 0.1 |
| 3 | `score_a` | 0.1 |
| 4 | `p1_diff_best` | 0.1 |
| 5 | `s1_n_cand_above_half` | 0.0 |
| 6 | `p1_diff_second` | 0.0 |
| 7 | `s1_best_p1` | 0.0 |
| 8 | `house_num_conflict` | 0.0 |
| 9 | `n_channels` | 0.0 |
| 10 | `rank_a` | 0.0 |
| 11 | `house_num_missing_cand` | 0.0 |
| 12 | `house_num_overlap` | 0.0 |
| 13 | `addr_norm_token_set_ratio` | 0.0 |
| 14 | `name_core_partial_ratio` | 0.0 |
| 15 | `name_norm_partial_ratio` | 0.0 |

### 6. Error Analysis: 10 False Positives (Sample)
| Country | S1 ID | S1 Name | S1 Address | Cand ID | Cand Name | Cand Address | Score | Diagnostic Reason |
|---|---|---|---|---|---|---|---|---|
| US | `S1-213984` | PW Frontier Globa LLC | 1700 Glenwood Drive, Moore, OK | `S2-990700183` | Martin Martin Frontier Globa LLC | 1700 GLENWOOD DRIVE, MOORE, OK | 0.722 | High orthographic string similarity on common brand or partial address collision |
| India | `S1-418986` | Jain Foundation Limited | S-35/1, Gns Plaza, Site Iv, Kasna Road, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh | `S3-733686614` | Jain Management | S-35/12, Gns Plaza, Site Iv, Kasna Road, Gautam Buddha, Greater Noida, UP | 0.710 | High orthographic string similarity on common brand or partial address collision |
| India | `S1-626781` | Premier Raj Hospitality Private Limited | Bangalore, Karnataka, Joly Paul, 3Rd Floor, St, Marys Nagar, Bommanahalli, Managammanapalya, No 9/5/5, Bangalore | `S2-858287433` | ಪ್ರೀಮಿಯರ್ ರಾಜ್ ಮೀಡಿಯಾ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | DOOR NO G-9/5/12, JOLY PAUL, 3RD FLOOR, ST, MARYS NAGAR, BOMMANAHALLI, MANAGAMMANAPALYA, BANGALORE, ಕರ್ನಾಟಕ | 0.819 | High orthographic string similarity on common brand or partial address collision |
| US | `S1-1070985` | Classic Information | 2718 Bowdish Road, WA, Spokane Valley, Unit G-338 | `S3-326564447` | Classic Information Corp | #2731 Bowdish Road, NULL, Spokane Valley, Washington | 0.856 | High orthographic string similarity on common brand or partial address collision |
| India | `S1-2261981` | Nagaya Realty Private Limited | 28, Bazarpeth, 1St Floor, Near Meenatai Thakarey Hall, Bhiwandi, Bhiwandi, Thane, Maharashtra | `S2-405787580` | Nagayaer Realty-Private | www.nagayaerr.com | NO 922 30, BAZARPETH, 1ST FLOOR, NEAR MEENATAI THAKAREY HALL, BHIWANDI, BHIWANDI, Maharashtra | 0.770 | High orthographic string similarity on common brand or partial address collision |
| India | `S1-3373087` | QY Communication LLP | Sno. 48/1/2 Nr.Kate Puram, Haveli, Pune, Maharashtra | `S2-36458485` | QY Communication L.L.P. | None | 0.945 | High orthographic string similarity on common brand or partial address collision |
| US | `S1-3698981` | Lower Utilities | 955 Oak Dale Drive, Xenia, OH | `S2-381384424` | Lower Utilities [Co.] | 95 OAK DALE DR, XENIA, OH | 0.982 | High orthographic string similarity on common brand or partial address collision |
| India | `S1-3879585` | Meenendra Renewables Pvt Ltd | Dcb-218, 2F, Dlf Cyber City, Bhubaneswar, Odisha, Bhubaneswar, Khordha, Orissa | `S3-148629982` | Meennendra Renewables Pvt Ltd | Dcb-219, 2F, Dlf Cyber City, Bhubaneswar, Odisha, Pur Bhubaneswar, Bhubaneswar, OD | 0.832 | High orthographic string similarity on common brand or partial address collision |
| India | `S1-4353789` | Target (india) | H.No. 76/A-1, Balkeshwar Colony, Agra, Uttar Pradesh | `S3-958322160` | Target Buildwell | H.no 76/A-3, Balkeshwar Colony, Delhi, Agra, UP | 0.957 | High orthographic string similarity on common brand or partial address collision |
| India | `S1-4573884` | Acg Industries Group Public Limited | 225/A, 15Th Cross, 9Th Main, 6Th Sector, Bangalore South, Bangalore, Karnataka | `S3-618185293` | Acg Industries Group | None | 0.844 | High orthographic string similarity on common brand or partial address collision |

### 7. Error Analysis: 10 False Negatives (Sample)
| Country | S1 ID | S1 Name | S1 Address | Cand ID | Cand Name | Cand Address | Diagnostic Reason |
|---|---|---|---|---|---|---|---|
| India | `S1-22633084` | Premier Foundation Limited | Gurgaon, Sushant Lok Phase 1 Near Park Plaza, Gurgaon, Haryana, C-329 | `S2-680658950` | प्रीमियर फाउंडेशन लिमिटेड | C-329, GURGAON, Haryana | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| India | `S1-22633084` | Premier Foundation Limited | Gurgaon, Sushant Lok Phase 1 Near Park Plaza, Gurgaon, Haryana, C-329 | `S3-912074059` | प्रीमियर फाउंडेशन लिमिटेड | No. 616 C-329, Gurgaon, HR | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| US | `S1-66222580` | Signature Aspac | 136 Patriot Way, Poca, WV | `S3-600441546` | Signature Áspac | None | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| India | `S1-337333581` | First Builders Private Limited | Mohali, Punjab, Scf 31 Phase-2 | `S3-914926798` | ਫਸਟ ਬਿਲਡਰਜ਼ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ | Scf 31 Pahse-2, Mohali, ਪੰਜਾਬ | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| US | `S1-302789684` | Teamsters Local 491 | 1301 3rd Street, Ozark, MO | `S3-129177983` | Teamsters Local 491 LP | None | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| US | `S1-856138680` | Bernier Certified Pharmaceutical Inc. | 5712 Truelson Drive, Fort Worth, TX | `S2-429878196` | BERNIER PHARMACEUTICAL INC. SERVICES | 5866 TRUELSON DR, NULL, FORT WORTH, TX | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| India | `S1-424774284` | New Delhi Aerospace Private Limited | 19 Narang Colony Janakpuri, New Delhi, Delhi | `S2-237416502` | New Delhi  Aer0space Private | None | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| India | `S1-388364080` | YJ Solar Limited | Falt No 301, Plot No 420, 3Rd Flr, Sapphire Vistas Deepthisri Nagar, Madeenaguda, Hafeezpet, Hyderabad, Rangareddy, Telangana | `S2-281306972` | Orbivantage | FALT NO G-301, RANGAREDDY, HYDERABAD, Telangana | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| US | `S1-105397585` | Desert Xsolla | 200 Jackson Drive, Berryville Town, VA | `S3-297617221` | lridrex | 200 Jackson Drive, Berryville CITY, Virginia | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |
| US | `S1-409328987` | Value Textile Industries Inc | 126 Oak Street, South Dayton, NY | `S2-392838403` | Value Textile Industries | None | Severe abbreviation, legal entity divergence, or candidate pruned below threshold |

### 8. Part Timings & Hardware Utilization
- **data_load**: 0.67s (0.01 min)
- **benchmark**: 63.61s (1.06 min)
- **stage1_training**: 268.63s (4.48 min)
- **stage2_training**: 139.35s (2.32 min)
- **evaluation**: 1.97s (0.03 min)
- **save_artifacts**: 0.19s (0.00 min)
- **Peak RSS**: **3995.0 MB**

