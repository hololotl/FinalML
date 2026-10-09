## 1. Topic

A study of demand forecasting methods and automatic courier slot planning in a food delivery service.

## 2. Research question

Can order forecasting and automatic slot planning improve demand coverage compared with manual planning: reduce the share of external coverage (deliveries via Yandex) and the volume of guarantee payments while controlling the courier-hours pool?

**Working hypothesis.** An automatic slot plan based on an ML demand forecast achieves a better undercoverage / overcoverage balance than manual allocation or simple heuristics (averages over past periods).

## 3. What is done

The work is carried out on anonymized operational data from a delivery service (location identifiers and commercially sensitive absolute figures are not disclosed; aggregates and relative metrics are used in the text).

The practical part includes a pipeline:

1. **Demand forecasting** on a one-week horizon by locations and time segments (using order history, opening hours, seasonality/calendar, and other features).
2. **Converting the forecast into courier slots** (car / bike) with operational constraints: slot capacity, safety buffer, rules for low-activity locations, etc.
3. **Comparison of approaches:**
   - baseline: manual/heuristic plan (average over N weeks / actually assigned slots);
   - ML forecast + slot calculation;
   - planning policy variants (different buffer / capacity).

This is research, not only deployment: methods, metrics, and a comparison protocol are fixed on historical periods (and, if data are available, before/after or pilot vs manual plan).

## 4. How it is evaluated

### Technical metrics (forecast and slot quality)
- for orders: MAE, RMSE, WAPE;
- for slots: bias, under/over coverage, slot WAPE.

### Operational / business metrics (planning effect)
- share of orders that went to external coverage (Yandex) — a proxy for courier shortage;
- guarantee payments / courier idle time — a proxy for excess coverage;
- courier-hours and normalized indicators (per order / per 1,000 orders), so that comparison across periods is fair.

**Success criterion:** a reduction in the Yandex share and/or guarantee pay with a comparable or smaller courier-hours pool relative to the baseline / manual plan.

## 5. Data limitations and ethics

Data are anonymized; the thesis uses aggregates and relative values. Raw location identifiers, personal data, and commercially sensitive absolute amounts are not published
