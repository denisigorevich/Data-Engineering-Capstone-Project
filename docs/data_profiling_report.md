# One-Time Data Profiling & Data Quality Assessment

## Tools & Setup
- **Google BigQuery**: Used for SQL-based profiling on `vehicles` dataset (~249K rows)
- **Google Cloud Dataplex**: One-time on-demand profiling scan over cleaned BigQuery asset
- **Sampling**: 10% (Dataplex) + full scan (BigQuery)
- **Profiling Goals**: Assess data completeness, consistency, validity, and uniqueness before moving to rules-based checks

---

## Data Completeness (Null Count)
| **Column**         | **Null Count** | **% Null**   |
|--------------------|----------------|--------------|
| `county`           | 249,202        | 100.00% ❌   |
| `VIN`              | 103,221        | 41.43% ⚠️    |
| `size`             | 141,039        | 56.61% ⚠️    |
| `condition`        | 67,706         | 27.17%       |
| `drive`            | 35,352         | 14.19%       |
| `paint_color`      | 49,583         | 19.89%       |
| `year`             | 1,118          | 0.45%        |
| `manufacturer`     | 9,524          | 3.82%        |
| `model`            | 2,603          | 1.04%        |
| *(All others)*     | `<1% or none`  | Acceptable   |

---

## Validity Checks

| **Column**     | **Check**                      | **Finding** |
|----------------|--------------------------------|-------------|
| `price`        | Max = 3.7B                     | ❌ Outlier |
| `odometer`     | Max = 10M                      | ⚠️ High |
| `year`         | Min = 1900                    | ⚠️ Improbable |
| `VIN`          | Only 145,046 of 249,202 are 17 characters | ⚠️ 58.2% valid |

---

## Categorical Cardinality
| **Column**       | **# of Unique Values** | **Comments**            |
|------------------|------------------------|-------------------------|
| `type`           | 13                     | High variety            |
| `paint_color`    | 12                     | Expected                |
| `fuel`           | 5                      | OK                      |
| `transmission`   | 3                      | OK                      |
| `condition`      | 6                      | OK                      |
| `drive`          | 3                      | OK                      |
| `size`           | 4                      | OK                      |

---

## Descriptive Statistics

### `price`
- **Min**: 0
- **Max**: 3,736,928,711 ❌
- **Avg**: 66,179
- **StdDev**: 12,397,444 ❌ → suggests extreme skew

### `odometer`
- **Min**: 0
- **Max**: 10,000,000 ⚠️
- **Avg**: 104,241
- **StdDev**: 211,056 → likely contains extreme outliers

### `year`
- **Min**: 1900 ⚠️
- **Max**: 2022 ✅

---

## VIN Integrity
- **Valid VINs (17 chars)**: 145,046
- **Invalid or missing**: 104,156 → **41.77% of all records**

---

## Summary of Identified Issues

| **Issue**                      | **Type**         | **Action Plan**                                                                 |
|-------------------------------|------------------|----------------------------------------------------------------------------------|
| `county` fully null           | Completeness     | Drop column from final dataset                                                |
| `VIN` missing or invalid      | Validity         | Flag or exclude from VIN-based joins or lookups                               |
| Extreme `price` & `odometer`  | Validity/Outliers| Cap or flag values > reasonable thresholds (e.g., price > $250K, odo > 1M)     |
| Invalid `year` values         | Validity         | Exclude values < 1981 as pre-VIN era, or set to null                          |
| Missing `size`, `paint_color` | Completeness     | Impute with 'unknown' or exclude from modeling depending on use case          |
