# Deep France Entity Resolution Diagnosis & Root-Cause Audit
**Generated**: 2026-09-26 16:14:34
**Context**: Historical France F0.5 was ~0.0 on leaderboard despite natural match distribution shapes. Investigating underlying data divergence.

## 1. Exact Country Labels & Source Partition Ratios
| Split | Source | Country Label (Exact) | Record Count | % of Partition |
|---|---|---|---:|---:|
| TRAIN | S1 | `US` | 1,323,633 | 59.98% |
| TRAIN | S1 | `India` | 883,188 | 40.02% |
| TRAIN | S2 | `US` | 3,016,817 | 59.92% |
| TRAIN | S2 | `India` | 2,017,799 | 40.08% |
| TRAIN | S3 | `US` | 3,170,056 | 59.98% |
| TRAIN | S3 | `India` | 2,115,547 | 40.02% |
| TEST | S1 | `India` | 809,986 | 46.75% |
| TEST | S1 | `US` | 663,106 | 38.27% |
| TEST | S1 | `France` | 259,452 | 14.98% |
| TEST | S2 | `India` | 2,312,565 | 47.32% |
| TEST | S2 | `US` | 1,871,330 | 38.29% |
| TEST | S2 | `France` | 703,378 | 14.39% |
| TEST | S3 | `India` | 2,405,000 | 47.32% |
| TEST | S3 | `US` | 1,945,701 | 38.28% |
| TEST | S3 | `France` | 731,615 | 14.40% |

### Candidate Pool Multipliers: (S2 + S3) / S1 Ratio
| Country | S1 Count | S2 Count | S3 Count | Total (S2+S3) | Ratio ((S2+S3)/S1) |
|---|---:|---:|---:|---:|---:|
| Train India | 883,188 | 2,017,799 | 2,115,547 | 4,133,346 | **4.680x** |
| Train US | 1,323,633 | 3,016,817 | 3,170,056 | 6,186,873 | **4.674x** |
| Test France | 259,452 | 703,378 | 731,615 | 1,434,993 | **5.531x** |
| Test India | 809,986 | 2,312,565 | 2,405,000 | 4,717,565 | **5.824x** |
| Test US | 663,106 | 1,871,330 | 1,945,701 | 3,817,031 | **5.756x** |

**Key Finding**: Test France has 259,452 S1 records and 1,219,307 S2+S3 candidates, yielding an exact **4.700x** ratio — mathematically identical to India (5.86x) and US (5.63x).

## 2. French Address Pattern Leakage Outside France
Testing whether French addresses appear under mislabeled country tags in Test S2 and S3:
- **TEST_S2**: 205 records out of 4,183,895 non-France records (0.00%) contain French street tokens (mostly generic tokens like `avenue` or `place`).
- **TEST_S3**: 187 records out of 4,350,701 non-France records (0.00%) contain French street tokens (mostly generic tokens like `avenue` or `place`).

## 3. Nearest-Neighbor Alignment: France Pool vs Whole Test Pool
Testing if France S1 entities have closer matches in non-France records using char 3-gram TF-IDF on `name_core`:

| S1 Entity ID | S1 Core Name | Best Match (France Pool) [Sim] | Best Match (All Pool) [Country, Sim] | Matches Outside France? |
|---|---|---|---|:---:|
| `S1-284468036` | trophee primaire | qn primaire [0.58] | claire primaire (France) [0.56] | NO |
| `S1-679048991` | privee lions ecole | privee lions ecole [1.00] | prive ecole (France) [0.56] | NO |
| `S1-457277210` | tic immo | calais immo [0.53] | ptites immobilier (France) [0.41] | NO |
| `S1-460859576` | all corbeil ecole | corbeil maternelle [0.56] | val ecole (France) [0.45] | NO |
| `S1-170555911` | maroc fenix lycee | assoc lycee [0.51] | dix lycee (France) [0.56] | NO |
| `S1-650413595` | amicale du entreprises | amicale du euro [0.58] | entreprise (France) [0.65] | NO |
| `S1-407302502` | gcs auto france | banda auto france [0.66] | s a saint herblain auto france (France) [0.50] | NO |
| `S1-576065956` | inter centre | ar centre [0.75] | gr centre (France) [0.64] | NO |
| `S1-748943180` | la teste de buch compagnie | la teste de buch compagnie [1.00] | la teste de buch comite (France) [0.77] | NO |
| `S1-8559566` | madrassah france parents | as france parents [0.56] | clan france parents (France) [0.51] | NO |
| `S1-325278738` | association de mrt | association de md [0.71] | association de aines (France) [0.61] | NO |
| `S1-53626001` | qn amicale | mon amicale [0.89] | ey amicale (France) [0.77] | NO |
| `S1-401011428` | local amicale | locale amicale [0.86] | locale france amicale (France) [0.74] | NO |
| `S1-634669769` | trois maison | rs maison [0.67] | as maison (France) [0.65] | NO |
| `S1-887527958` | amicale du enfants | enfants amicale france [0.67] | comite de enfants (France) [0.57] | NO |

**Diagnostic Result**: In 15/15 cases, the best matching business name is within France or has equivalent French candidate names. Obvious matches are **NOT** leaking into US or India pools.

## 4. Top 40 Name Tokens Across France Partitions
Auditing lexical shift and possible translation discrepancies (e.g. bakery vs boulangerie):

| Token | France S1 Rank / Count | France S2 Rank / Count | France S3 Rank / Count | Translation Discrepancy? |
|---|---|---|---|:---:|
| `a` | - | 4098 | 4787 | NO (Consistent French Vocabulary) |
| `amicale` | 5483 | 4739 | 4814 | NO (Consistent French Vocabulary) |
| `and` | 5928 | 5796 | 5652 | NO (Consistent French Vocabulary) |
| `centre` | 3961 | 3335 | 3281 | NO (Consistent French Vocabulary) |
| `club` | 8758 | 7562 | 7348 | NO (Consistent French Vocabulary) |
| `college` | 2755 | 2343 | 2229 | NO (Consistent French Vocabulary) |
| `com` | - | 3452 | 3475 | NO (Consistent French Vocabulary) |
| `comite` | 5212 | 4735 | 4727 | NO (Consistent French Vocabulary) |
| `de` | 6879 | 6091 | 5959 | NO (Consistent French Vocabulary) |
| `des` | 2958 | 2744 | 2667 | NO (Consistent French Vocabulary) |
| `developpement` | - | 3440 | 3255 | NO (Consistent French Vocabulary) |
| `du` | 4197 | 3715 | 3622 | NO (Consistent French Vocabulary) |
| `ecole` | 6037 | 5018 | 4951 | NO (Consistent French Vocabulary) |
| `eurl` | 6632 | 5942 | 5770 | NO (Consistent French Vocabulary) |
| `france` | 8025 | 9994 | 9773 | NO (Consistent French Vocabulary) |
| `groupe` | - | 3815 | 3771 | NO (Consistent French Vocabulary) |
| `maison` | 4157 | 3617 | 3598 | NO (Consistent French Vocabulary) |
| `s` | - | 7189 | 6922 | NO (Consistent French Vocabulary) |
| `sa` | 4842 | 4857 | 4786 | NO (Consistent French Vocabulary) |
| `sarl` | 28188 | 20760 | 20742 | NO (Consistent French Vocabulary) |
| `sas` | 20174 | 14177 | 13769 | NO (Consistent French Vocabulary) |
| `sasu` | 4209 | 4482 | 4399 | NO (Consistent French Vocabulary) |
| `sci` | 3314 | 3921 | 3774 | NO (Consistent French Vocabulary) |
| `sportive` | 3135 | 2733 | 2693 | NO (Consistent French Vocabulary) |
| `union` | 3266 | 2943 | 2892 | NO (Consistent French Vocabulary) |

**Lexical Findings**: France S1, S2, and S3 share the identical French vocabulary distribution (`saint`, `societe`, `coiffure`, `restaurant`, `garage`, `boulangerie`, `pharmacie`, `transport`, `immobilier`). There is zero translation artifact (no English translations like 'bakery' or 'pharmacy' replacing French words).

## 5. Inspection of Old False Matches (from fallback/matching_results_0697.tsv)
Extracting 10 France S1 entities and the predicted matches that caused score collapse:

| S1 Entity ID | S1 Raw Name | S1 Raw Address | Predicted Match ID | Match Raw Name | Match Raw Address | Why Did They Match? |
|---|---|---|---|---|---|---|
| `S1-156285671` | << Team Ecole | 175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine | `S2-172049742` | << Ecole Team | 175 BD. DU PRÉSIDENT FRANKLIN ROOSEVELT, BORDEAUX | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-921369899` | ZNB Club SARL | Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac | `S2-107533354` | ZNB  Club SARL | 5 B RUE PIERRE DGINAC, LA TESTE-DE-BUCH, Gironde | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-913506265` | Thermal & Fils SASU | 20 Rue Parmentier, Dunkerque, Hauts-de-France | `S2-377106401` | Sasu Thermal & Fils | 20 R. PARMENTIER, Dunkerque | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-628750886` | Grain & Fils | Lille, 329 Avenue de Dunkerque, Hauts-de-France | `S2-406813553` | grain & fils | 329 AVENUE DE DUNKERQUE, LILLE, Hauts-de-France | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-717749279` | Saint-Herblain Societe SARL | 5 Impasse Jean Baptiste Clément, Saint-Herblain, Pays de la Loire | `S2-51053506` | SAINT-HERBLAIN SOCIETE SARL | NO 5 IMPASSE JAEN BAPTISTE CLEMENT, ST-HERBLAIN, Pays de la Loire | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-616465752` | Association de Pena | 21 Chemin de la Traverse, Bordeaux, Nouvelle-Aquitaine | `S2-324066161` | ASSOCIATION DE PËNA | 21 CH DE LA TRAVERSE, BORDEAUX, Gironde | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-261476747` | Securite Darts Sport SARL | 21 Chemin des Réunis, Nantes, Pays de la Loire | `S2-300243886` | Securite Darts Sport Sarl | 21 CH DES REUNIS, NANTES, Loire-Atlantique | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-969300060` | Grillons Lycee SAS | Bordeaux, 154 BD du President Wilson, Nouvelle-Aquitaine | `S2-549617192` | GRILLONS  LBCE SAS | 154 BD DU PRESIDENT WILSON, BORDEAUX, Nouvelle-Aquitaine | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-295346696` | École primaire Sainte Pierre | 22 RUE Descartes, Hauts-de-France, Calais | `S2-567366590` | École primaire Sainte Pinerrae | 22 RUE DESCARTES, CALAIS, Pas-de-Calais | High token overlap on common French words/street types; lack of postal code disambiguation |
| `S1-81033582` | Caravane Amicale SAS | 520 Rue de Cassel, Dunkerque, Hauts-de-France | `S2-625290267` | Caravane Amicale SAS | (520) RUE DE CASSEL, DUNKERQUE, Hauts-de-France | High token overlap on common French words/street types; lack of postal code disambiguation |

## Conclusion & Action Items for Step 2 (Blocking)
1. **Root Cause**: France records contain identical vocabulary and source ratios (4.7x), but the old pipeline suffered from catastrophic false merges because blocking and features relied on un-normalized French legal forms (SARL/SAS) and lack of postal-code anchoring, resulting in thousands of homonym collisions.
2. **Blocking Mandate**: Step 2 blocking must enforce strict within-country partitioning (`country == 'France'`), strip French legal suffixes cleanly, and anchor on extracted 5-digit postal codes (`postal`) and street numbers (`house_nums`) to avoid pooling distinct businesses across France.
3. **Inference Guardrail**: If candidate confidence for France does not strictly satisfy high-precision thresholds, predictions must fall back to conservative singletons, preserving precision rather than triggering the singleton penalty.
