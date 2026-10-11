# Research Proposal (Bachelor Thesis)

**Program:** Informatics and Computer Engineering, profile “Information Systems Engineering”

---

## 1. Topic (working title)

A comparative analysis of weekly demand forecasting methods and courier slot planning against the manual plan in a food delivery service.

## 2. Introduction and research problem

In food delivery services, weekly courier shift planning is often done manually or with simple heuristics. Planning errors show up as two opposite effects:

- **undercoverage:** not enough own couriers → some orders go to external coverage (in our case — deliveries via Yandex);
- **overcoverage:** too many couriers → idle time and **guarantee payments** (a minimum hourly/shift pay if earnings from orders fall below the guarantee).

The thesis problem: whether the pipeline “one-week demand forecast → automatic slot calculation” can achieve a better balance of these effects than the manual plan, and how to measure this fairly on historical data.

## 3. Research question and hypothesis

**Question.** Does automatic courier slot planning based on a weekly demand forecast yield a better trade-off between undercoverage (Yandex share) and idle time (guarantee pay) than the manual plan and simple forecasting baselines?

**Hypothesis.** A plan based on an ML forecast (CatBoost), when varying the slot buffer, Pareto-dominates or is no worse than the manual plan and heuristics: at the same (or lower) guarantee level, Yandex share is lower and courier-hours are not higher. A negative result (no improvement) is also a valid research outcome.

## 4. Novelty and contribution

**Contribution type:** empirical / applied (a comparative study on real anonymized data), with an optional methodological research extension.

**Contribution in one sentence.**  
An empirical comparison of the pipeline “weekly demand forecast → courier slot calculation” with the manual plan on real delivery data, using a calibrated under/overcoverage simulation (Yandex / guarantee) and a buffer trade-off curve; as a research option — a quantile buffer instead of a fixed safety margin.

The “forecast + staff/slot planning” link is long established in the literature; the novelty is not inventing that link, but **comparing it on the operational data of a specific network** with business proxies (external coverage and guarantee pay) and explicit simulator calibration on the manual plan.

## 5. State of the Art

Below are key directions and **what they do not do** relative to our setting.

1. **Yu, Lan, Mao (2023).** Short-term demand prediction for on-demand food delivery with attention-based ConvLSTM. *Systems*.  
   Short-term ODFD demand forecasting (about one hour), strong on accuracy.  
   **Does not:** weekly shift-planning horizon; converting forecast into slots; comparison with a manual plan; Yandex/guarantee metrics.
2. **Ulmer, Savelsbergh (2020).** Workforce scheduling in the era of crowdsourced delivery. *Transportation Science*.  
   Scheduling under crowdsourced/ad-hoc couriers.  
   **Does not:** our own restaurant-network setting with a weekly manual slot plan; simulation of external coverage and guarantee pay as business comparison metrics.
3. **Behrendt, Savelsbergh, Wang (2023).** A prescriptive machine learning method for courier scheduling on crowdsourced delivery platforms. *Transportation Science*.  
   Prescriptive ML for courier schedules under demand/ad-hoc forecasts.  
   **Does not:** comparison with the actual manual plan of a specific dark kitchen / restaurant delivery network; under/over simulator calibration on historical guarantee pay and external coverage.
4. **Zhang, Zhang, Baldacci, Tang (2026).** Workforce planning for meal deliveries with ad-hoc drivers: a distributionally robust contextual optimization approach. *EJOR*.  
   Tactical workforce planning (in-house + ad-hoc) under uncertainty.  
   **Does not:** a simple reproducible pipeline “baseline forecasts vs boosting → slots → buffer curve” for a bachelor-level comparative study on our metrics.
5. **Hyndman, Athanasopoulos.** *Forecasting: Principles and Practice* (OTexts).  
   Standard baselines (naive / seasonal naive) and time-series evaluation protocol.  
   **Does not:** the domain setting of courier slots and delivery operating costs.

**Review takeaway:** demand forecasting and staff scheduling are studied separately; our gap remains — **a fair comparison of an auto plan with the manual plan on real delivery data via a calibrated Yandex/guarantee simulation and a trade-off curve**.

A negative result (the auto plan is not better than the manual plan under the chosen criterion) is also a research result.

## 6. Goal and objectives

**Goal.** Develop and experimentally compare weekly demand forecasting methods and automatic courier slot formation with the manual plan on technical and operational metrics.

**Objectives:**

1. Prepare an anonymized dataset (locations, segments, orders, slots, Yandex, guarantee).
2. Implement forecasting methods and the forecast-to-slots conversion.
3. Build and calibrate an under/overcoverage simulator.
4. Run the comparison under a rolling / hold-out protocol with a buffer curve.
5. Record conclusions, limitations, and (optionally) a quantile buffer.

## 7. What is done (core and optional)

### Core of the work

1. **Demand forecasting** on a **one-week** horizon by locations (~150) and **time segments** (lunch / dinner / block_06_12, etc.).
2. **Converting the forecast into slots** (car / bike) with capacity, safety buffer, and operational rules.
3. **Comparison of forecasting methods:**
   - seasonal naive;
   - average over N=4 weeks in the same segment;
   - linear regression on features;
   - CatBoost (the main working model in the current pipeline);
   - **the actual manual plan** as an operational baseline (not a forecast, but the slots that were actually scheduled).
4. **Simulation of business metrics** for any slot plan (see Section 8).
5. **Trade-off curve** across different fixed buffers.

### Optional (not core)

- a pilot / before–after comparison on live data if the company provides a window;
- a **quantile buffer** (slots sized to a forecast quantile, e.g. p80) instead of a fixed % margin — a research option.

## 8. Yandex and guarantee simulation

In history, **actual** Yandex share and guarantee pay exist only for the **manual plan** that was really in place. For a model plan they must be obtained by simulation.

### 8.1. Undercoverage model (Yandex proxy)

At the **time-segment** level:

- take actual segment orders;
- take the slot-plan capacity (how many orders the slots can serve);
- if `orders > capacity(slots)` → treat the difference as going to external coverage (Yandex).

Caveat: in rare cases couriers can be moved from another location; the base model does not capture this (a conservative/simplified proxy).

### 8.2. Idle-time model (guarantee proxy)

There is a minimum courier pay per hour (then per shift as the sum of hours).  
If earnings from delivered orders on the slot/shift < guarantee → top-up = `guarantee − earned`.  
For a model plan: overlay **actual orders** on **model slots** and compute total top-up.

Courier earnings used for guarantee calculation are personal data. Only **aggregates without courier identifiers** are used (sums / rates by location–segment–week); raw per-courier pay is never stored or published in the thesis.

### 8.3. Simulator calibration (mandatory check)

Before comparing plans:

1. Take **actual manual slots** + **actual orders**.
2. Compute `simulate → Yandex_sim, guarantee_sim`.
3. Compare with `Yandex_fact, guarantee_fact` from the DB.
4. **Acceptance threshold (fixed in advance):** relative error of totals ≤ **10%** for Yandex volume and for guarantee pay.
5. Calibration is done on weeks **outside the final test set**.
6. If the threshold is not met — refine the simulation model first; only then evaluate model plans.

Also run a **sensitivity check** on the capacity parameter (vary orders-per-slot / capacity and report how Yandex and guarantee estimates change).

A base for this already exists: a delivery/lateness simulation script; within the thesis, Yandex and guarantee are added to the same pipeline.

## 9. How it is evaluated

### Technical metrics

- orders: MAE, RMSE, WAPE;
- slots: bias, under/over relative to need, slot WAPE.

### Operational metrics (after simulation)

- share/volume of external coverage (Yandex), normalized (per order / per 1,000 orders);
- guarantee payments, normalized;
- courier-hours.

### Success criterion (fixed in advance)

Primary criterion: **Pareto dominance**.

A plan A is better than plan B if, at the **same (or lower) guarantee level**, A has a **lower Yandex share**, and **courier-hours are not higher**.

We still build a trade-off curve over buffers (and optionally quantiles) to show the frontier; ranking vs the manual plan and baselines uses Pareto dominance only.  
Closeness to the manual plan is **not** a success criterion.

### Evaluation protocol

- **Locations:** about 150.
- **History:** orders for roughly the last 2 years.
- **Parameter tuning only:** 80/20 time split (or an inner validation window) for choosing hyperparameters / buffer candidates.
- **Final evaluation protocol (main):** **rolling-origin**, step = 1 week, forecast horizon = 1 week; the final test segment is untouched until final runs.
- **Frozen before final test:** CatBoost artifact and buffer/slot rules from the current pipeline are fixed before the final test runs (no retuning on test).
- **Comparison unit for reporting:** location–week metrics aggregated to location-level summaries for inference.
- **Statistics:** location–weeks are not independent (the same location repeats across weeks). Bootstrap is done **by location** (~150 independent units). Confidence intervals on metric differences use this location bootstrap. The comparison criterion is Pareto dominance as above.

## 10. Data, NDA, company permission, and limitations

The work uses **anonymized** operational data from a delivery service.  
In the thesis text: aggregates and relative metrics; no raw location IDs, courier IDs, personal data, or commercially sensitive absolute amounts (as agreed).  
For guarantee pay: only aggregates without courier identifiers (see §8.2).

**Written company permission** to use the data in the thesis: **requested, expected by 31 October 2026**. A chat screenshot from the company confirming permission is acceptable as written confirmation.  
Practical thesis work with the data starts after permission is obtained (until then — method, review, experiment design, and code preparation on the already available pipeline as agreed).

**Limitations (fixed in the proposal):**
1. The simulator is calibrated and validated **only on historical manual plans**; model plans may differ substantially from them, so simulated Yandex/guarantee for model plans are an extrapolation.
2. CatBoost and buffer/slot rules were developed by the author on the same operational setting; they are **frozen before final test** to reduce self-evaluation bias, but this does not remove all risk of overfitting to past practice.
3. The manual planner may have used information **not present in the dataset** (promotions, local events, one-off agreements), so the manual plan is not a pure function of the observed features.
4. Cross-location courier borrowing is rare and not modeled in the base simulator; capacity sensitivity is reported separately.

## 11. Month-by-month plan

| Month | Content |
| ----- | ------- |
| **October 2026** | Topic approval; company permission (by 31.10); submit **Ch.1–2 by 1 November**; simulator/protocol design |
| **November 2026** | **Data preparation** (anonymized extracts, joins, guarantee/Yandex aggregates); simulator calibration on non-test weeks; baseline forecasts |
| **December 2026** | Freeze model/rules; full rolling-origin comparison; buffer Pareto curve; first results |
| **January 2027** | Refinement (quantile buffer — if time allows); capacity sensitivity; Methodology/Results text |
| **February 2027** | Full draft; supervisor revisions |
| **March–April 2027** | Finalization, formatting, defense prep (deadlines per university calendar) |

## 12. References (starter list)

1. Yu X., Lan A., Mao H. Short-Term Demand Prediction for On-Demand Food Delivery with Attention-Based Convolutional LSTM. *Systems*, 2023, 11(10), 485.
2. Ulmer M.W., Savelsbergh M. Workforce Scheduling in the Era of Crowdsourced Delivery. *Transportation Science*, 2020, 54(4), 1113–1133.
3. Behrendt A., Savelsbergh M., Wang H. A Prescriptive Machine Learning Method for Courier Scheduling on Crowdsourced Delivery Platforms. *Transportation Science*, 2023, 57(4), 889–907.
4. Zhang J., Zhang Y., Baldacci R., Tang J. Workforce planning for meal deliveries with Ad-Hoc drivers: A distributionally robust contextual optimization approach. *European Journal of Operational Research*, 2026, 330(2), 427–443.
5. Hyndman R.J., Athanasopoulos G. *Forecasting: Principles and Practice*. OTexts (online textbook).
6. Dai H., Liu P. Workforce planning for O2O delivery systems with crowdsourced drivers. *Annals of Operations Research*, 2020, 291, 219–245.

---
