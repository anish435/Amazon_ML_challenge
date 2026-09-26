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
