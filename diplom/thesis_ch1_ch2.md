# Bachelor Thesis

**Program:** Informatics and Computer Engineering  
**Profile:** Information Systems Engineering  

**Topic:** A comparative analysis of weekly demand forecasting methods and courier slot planning against the manual plan in a food delivery service  

---

## Abstract

In food delivery services, weekly courier shift planning is often performed manually or with simple heuristics. Planning errors will lead to undercoverage (some orders may go to external coverage, e.g. via Yandex) or overcoverage (courier idle time and guarantee payments when earnings fall below minimum hourly or shift pay).

The goal of this work will be to develop and experimentally compare weekly demand forecasting methods and automatic courier slot calculation with the actual manual plan on anonymized operational data from a restaurant network (approximately 150 locations).

The planned pipeline is “order forecast → conversion into slots (car / bike) → simulation of operational metrics.” For model plans, external coverage share and guarantee pay will be obtained through a simulator to be calibrated on historical manual plans (a relative total-error threshold of at most 10% will be fixed in advance). The study will compare forecasting baselines (seasonal naive, four-week average, linear regression), a CatBoost model, and the actual manual plan. The primary success criterion will be Pareto dominance: at the same or lower guarantee level, lower Yandex share with courier-hours not higher; a buffer trade-off curve will also be reported.

The expected contribution is an empirical comparison on real delivery data with under/overcoverage business proxies; a negative result (no improvement over the manual plan) will also be treated as a valid research outcome.

**Keywords:** demand forecasting, shift planning, courier logistics, food delivery, CatBoost, simulation, Pareto.

---

# Chapter 1. Introduction

## 1.1. Relevance and context

Food delivery markets rely on information systems that link orders, kitchen operations, courier capacity, and delivery-time promises [Mao et al., 2022]. For a restaurant network with its own or mixed courier fleet, a key tactical task is **planning work slots on a one-week horizon**: how many couriers (and which vehicle types) each location needs in each time segment [Mandal et al., 2025; Zhang et al., 2026].

Planning quality directly affects operating costs and service. Under **undercoverage**, demand exceeds own courier capacity; some orders are served externally—in this setting, Yandex deliveries, used as a proxy for insufficient own capacity [Auad et al., 2024]. Under **overcoverage**, too many couriers are scheduled relative to realized demand; couriers idle and the company pays **guarantee top-ups** to reach minimum hourly or shift pay [Dai and Liu, 2020].

Automating planning from demand forecasts and slot-conversion rules is a typical **information systems engineering** problem: designing a data pipeline, models, business rules, and evaluation of the solution [Behrendt et al., 2023].

## 1.2. Object and subject of the research

**Object of research:** the process of weekly planning of courier work slots in a food delivery service (demand estimation, slot calculation, and comparison with the manual plan).

**Subject of research:** methods and algorithms for weekly demand forecasting and automatic conversion of forecasts into courier slots, and methods for evaluating them against the manual plan on technical and operational metrics.

## 1.3. Research problem

In practice, weekly slots are often built **manually** or with **simple heuristics** (past weeks, expert adjustments) [Ulmer and Savelsbergh, 2020]. This is hard to scale to many locations and does not guarantee a consistent trade-off between external coverage and idle time.

The research problem is: **whether the pipeline “one-week order forecast → automatic courier slot calculation” can achieve a better balance of undercoverage and overcoverage than the manual plan, and how to measure this fairly on historical data**, given that alternative (model) plans in the past have no observed Yandex or guarantee outcomes.

## 1.4. Knowledge gap (brief)

Demand forecasting in delivery and courier/workforce scheduling are well developed separately (Chapter 2). To the best of our knowledge, after a structured literature search (Section 2.2), few studies jointly address: a **one-week** horizon, a **restaurant network with an actual manual slot plan**, **empirical comparison of automatic vs manual planning**, and **operational proxies** (external coverage and guarantee) with **simulator calibration** on the manual plan. This thesis will target that gap in an applied comparative design.

## 1.5. Research question and hypothesis

**Research question.** Will automatic courier slot planning based on a weekly demand forecast yield a better trade-off between undercoverage (share of orders sent to external coverage) and overcoverage (guarantee payments / idle time) than the manual plan and simple forecasting baselines?

**Working hypothesis.** A plan based on an ML forecast (CatBoost), when varying the slot buffer, will Pareto-dominate or be no worse than the manual plan and heuristics: at the same or lower guarantee level, Yandex share will be lower and **courier-hours will not be higher**. A negative result (no improvement) will also be a valid outcome.

## 1.6. Goal and objectives

**Goal.** To develop and experimentally compare weekly demand forecasting methods and automatic courier slot formation with the manual plan on technical and operational metrics.

**Objectives:**

1. Prepare an anonymized dataset (locations, time segments, orders, slots, aggregates for external coverage and guarantee).
2. Implement forecasting methods and forecast-to-slots conversion (car / bike) with slot capacity, safety buffer, and operational rules.
3. Build and calibrate an under/overcoverage simulator; verify reproduction of actual Yandex and guarantee on the historical manual plan.
4. Conduct comparison under a rolling-origin protocol with a buffer trade-off curve; apply a pre-fixed Pareto dominance criterion.
5. Formulate conclusions, limitations, and (optionally) study a quantile buffer instead of a fixed margin.

## 1.7. Novelty and contribution

**Contribution type:** empirical / applied comparative study on real anonymized data, with an optional methodological extension (quantile buffer).

**Contribution in one sentence.** An empirical comparison of the pipeline “weekly demand forecast → courier slot calculation” with the manual plan on delivery data, with calibrated under/overcoverage simulation (Yandex / guarantee) and a buffer trade-off curve.

The link “forecast + staff/slot planning” is established in operations research and prescriptive analytics [Bertsimas and Kallus, 2020; Zhang et al., 2026]; novelty here is **comparison on operational data from a specific network** with business proxies and explicit simulator calibration on the manual plan.

## 1.8. Planned approach (overview)

The study will be designed as a **comparative experiment** on historical data.

**Core:**

1. One-week demand forecast across ~150 locations and **time segments** (lunch, dinner, block_06_12, etc.).
2. Conversion into slots (vehicle type, capacity, buffer, rules for low-activity locations).
3. Forecasting methods: seasonal naive; four-week segment average; linear regression; CatBoost; **actual manual plan** as operational baseline.
4. Simulation for any slot plan: capacity shortage → Yandex proxy; earnings below guarantee → guarantee proxy (aggregates without courier identifiers).
5. Trade-off curve over buffer values; ranking by **Pareto dominance** only.

**Optional:** before–after pilot; quantile buffer (e.g. p80).

Detailed design, data, and protocols will be presented in the **Methodology** and **Implementation** chapters.

## 1.9. Expected results and significance

**Technical metrics (planned):** MAE, RMSE, WAPE for orders; bias, under/over, slot WAPE for slots [Hyndman and Athanasopoulos, 2021].

**Operational metrics (planned, after simulation):** normalized external coverage and guarantee; courier-hours.

**Success criterion (pre-fixed):** Pareto dominance vs manual plan and baselines; closeness to the manual plan is **not** a success criterion.

Practical significance: support decisions on manual vs automatic slot planning. For Information Systems Engineering, a reproducible forecast-and-planning pipeline is also a project outcome.

## 1.10. Data, ethics, permission, and limitations

Work will use **anonymized** operational data. The thesis will report aggregates and relative metrics; raw location and courier identifiers and sensitive absolute amounts will not be published as agreed. Earnings for guarantee analysis will be used only in **aggregate** form without courier identifiers.

**Company permission (status as of November 2026):** written permission to use data in the thesis was **requested** before 31 October 2026; confirmation is **still pending** (update this sentence once the signed approval or an agreed written confirmation is received). Empirical work on company data will start only after permission is obtained.

**Plan B if permission is denied or delayed:** (i) limit published results to method design, expanded literature review, and **offline backtests** on fully anonymized aggregates already approved for internal use, if any; (ii) use **public meal-delivery or demand datasets** only for the **forecasting** submodule, while noting that **comparison with the actual manual slot plan** requires company data—if permission fails, the research question may need to be narrowed (to be agreed with the supervisor); (iii) document operational metrics as **methodology** without disclosing company-specific levels.

**Main limitations (fixed in advance):**

1. The simulator will be calibrated on **historical manual** plans; Yandex/guarantee for model plans are extrapolations.
2. CatBoost and buffer rules are developed by the author on the same setting; they will be **frozen before the final test**, but overfitting risk remains.
3. Manual planners may use information **outside the dataset** (promotions, events).
4. Rare cross-location courier moves are not modeled in the base simulator; capacity sensitivity will be reported separately.

## 1.11. Thesis structure

- **Chapter 1** — Introduction.
- **Chapter 2** — Literature review and related work.
- **Chapter 3** — Methodology (research design, metrics, evaluation protocol, ethics).
- **Chapter 4** — Implementation (data pipeline, forecasting, slot engine, simulator).
- **Chapter 5** — Evaluation (experiments, results, discussion).
- **Chapter 6** — Conclusion.
- **Appendices** — if needed: pipeline diagrams, configuration tables.

---

# Chapter 2. Literature Review and Related Work

## 2.1. Purpose of the chapter

This chapter synthesizes **demand forecasting**, **meal/last-mile delivery operations**, **workforce and shift planning**, and **forecast-to-capacity links** in service operations. For each close line of work we summarize: research question, data, evaluation, main results, and limitations. The goal is to justify the chosen methods and metrics and to position this thesis relative to the **closest published work**.

## 2.2. Literature search strategy and coverage

**Databases and queries (October 2026).** Searches were run in **Scopus**, **IEEE Xplore**, and **Google Scholar** with strings such as:  
(`"meal delivery" OR "food delivery" OR "last-mile"`) AND (`workforce` OR `staffing` OR `shift planning` OR `courier scheduling`) AND (`forecast*` OR `demand`); plus `"on-demand food delivery"` AND `forecasting`; `"workforce planning"` AND `crowdsourced`.

**Screening (preliminary, draft stage):**

| Stage | Count |
|--------|------:|
| Records retrieved (deduplicated) | ~185 |
| Title/abstract screened | ~62 |
| Full texts assessed for relevance | ~28 |
| Sources cited in this draft | **22** |

**Inclusion:** English publications relevant to delivery/meal logistics, demand forecasting, or courier/workforce planning with a clear method or experiment. **Exclusion:** unrelated domains, papers without recoverable bibliographic metadata.

The reference list will be expanded during the thesis period; backward/forward search from **Zhang et al. [2026]** and **Mandal et al. [2025]** was used to capture the 2023–2026 meal-delivery workforce cluster.

## 2.3. Demand forecasting: competitions, boosting, and foundation models

### 2.3.1. M5 and gradient boosting in forecasting

The **M5 Forecasting – Accuracy** competition [Makridakis et al., 2022] showed that **gradient boosted trees** (especially LightGBM ensembles with lag/calendar features and recursive/direct strategies) strongly outperform classical statistical benchmarks on large-scale hierarchical retail demand. This supports using **boosting** as a serious forecasting baseline in operational demand pipelines—not only deep learning.

**Implication for this thesis:** CatBoost will be the main ML forecaster; seasonal naive and moving averages serve as simple baselines [Hyndman and Athanasopoulos, 2021].

### 2.3.2. CatBoost

**Prokhorenkova et al. [2018]** introduce CatBoost: ordered boosting and ordered target statistics for categorical features, reducing prediction shift/target leakage common in standard gradient boosting.

| Aspect | Content |
|--------|---------|
| Question | How to improve GBDT accuracy with high-cardinality categoricals without leakage? |
| Data | Multiple public ML benchmarks (not delivery-specific) |
| Evaluation | Prediction error vs XGBoost/LightGBM |
| Main result | Strong empirical gains on heterogeneous tabular data |
| Limitations | Not a forecasting paper; no shift/slot planning |

**Implication:** CatBoost fits location/segment categorical features in our weekly panel; it does not replace a dedicated forecasting architecture but is standard for structured operational panels.

### 2.3.3. Short-term ML demand in food delivery

**Yu et al. [2023]** (At-ConvLSTM) forecast **hourly** ODFD demand with spatial–temporal deep learning on city-scale data.

| Aspect | Content |
|--------|---------|
| Question | Short-horizon ODFD demand accuracy |
| Data | Shenzhen ODFD |
| Evaluation | MAE/RMSE vs DL/statistical baselines |
| Main result | Best accuracy among compared models |
| Limitations | Hourly horizon; no weekly slots; no manual-plan comparison |

### 2.3.4. Foundation models (Chronos, TimesFM) — planned role

**Chronos** [Ansari et al., 2024] and **TimesFM** [Das et al., 2024] are pretrained time-series foundation models with strong **zero-shot** performance on many public series.

**Why they are not the primary forecaster in this thesis (planned):**

1. **Granularity and covariates:** Our panel is location × time-segment × week with operational features (open hours, segment type, vehicle split). Foundation models in the cited work target mostly **univariate** or patch-based series; integrating rich categorical operational features is less straightforward than CatBoost on a feature table.
2. **Horizon and alignment:** We forecast **one week ahead** on business segments aligned with slot rules; zero-shot models would require careful alignment of context length and segment aggregation; this is **planned as optional baseline**, not core, due to thesis scope and reproducibility on company panel data.
3. **Fair comparison:** M5 and industry practice justify **boosting + classical baselines** as the main comparison arm; foundation models will be discussed as **future work** or optional experiment if compute and permission allow.

## 2.4. Meal delivery operations and capacity

### 2.4.1. Mao et al. [2022] — data and research agenda

| Aspect | Content |
|--------|---------|
| Question | What operational data exist for meal platforms and what research is needed? |
| Data | Hangzhou platform, Jul–Aug 2015 (orders, drivers, weather, traffic) |
| Evaluation | Descriptive + research agenda (not a single optimization metric) |
| Main result | Public dataset; identifies dispatch, routing, staffing questions |
| Limitations | Platform (not restaurant network); no weekly slot vs manual comparison |

### 2.4.2. Auad et al. [2024] — capacity and demand management

| Aspect | Content |
|--------|---------|
| Question | How do fleet size, deadlines, depot sharing, and **service-area restriction** affect meal delivery capacity? |
| Data | Stylized MIP scenarios (1–2 depots), experimental |
| Evaluation | Required fleet size, order coverage |
| Main result | Temporary demand restriction (smaller service area) can sharply reduce required fleet |
| Limitations | Aggregators/crowd model; not restaurant-owned weekly slot planning |

**Link to thesis:** External coverage (Yandex) and guarantee are **downstream** of capacity mismatch; our simulator will operationalize mismatch at segment level.

### 2.4.3. Zhao et al. [2025] — stochastic scheduling and routing (OMD)

| Aspect | Content |
|--------|---------|
| Question | Scheduling/routing policy with **mixed in-house and crowdsourced** drivers under uncertainty |
| Data | **Real** Meituan OMD data: orders and drivers, 3–30 July 2017 (Section **5.1**, *Numerical experiments*, EJOR 323(1), pp. 147–151) |
| Evaluation | Policy performance vs alternatives; fleet-size bounds |
| Main result | MDP-based policies; continuous approximation for in-house fleet size |
| Limitations | **Operational** routing/scheduling, not **tactical weekly slot planning** vs human planner; no guarantee/external-coverage proxies |

### 2.4.4. Ulmer et al. [2021] — restaurant meal delivery problem

Dynamic pickup-and-delivery with deadlines and random ready times [Ulmer et al., 2021]—foundational for **meal delivery logistics**, but focused on **routing/dispatch**, not weekly shift design for a fixed restaurant network.

### 2.4.5. Routing in food delivery

**Yildiz and Savelsbergh [2019]** and **Steever et al. [2019]** study high-quality or **dynamic courier routing** for meal/food delivery—again **routing**, not forecast-to-weekly-slots.

## 2.5. Workforce planning and shift scheduling (2023–2026 cluster)

### 2.5.1. Zhang et al. [2026] — **closest optimization-oriented work**

| Aspect | Content |
|--------|---------|
| Question | Tactical hiring of **in-house** drivers vs **ad-hoc** drivers under uncertain demand and weather (contextual features) |
| Data | Real-world inspired instances; DRCO framework |
| Evaluation | Total cost (fixed + variable + penalty for unfulfilled orders) |
| Main result | Two-stage distributionally robust contextual optimization; Benders decomposition |
| Limitations | **Optimization** of hiring/delivery assignment, not an **empirical comparison** with a **documented manual weekly slot plan**; metrics differ from Yandex/guarantee proxies; ad-hoc/platform setting vs our owned/restaurant network |

**Positioning:** Zhang et al. [2026] is the **nearest published work** on **meal-delivery workforce planning under uncertainty**. This thesis **complements** it: less focus on proving optimality of a DRCO model, more on **reproducible forecasting baselines + slot rules + calibrated simulation** compared to **actual manual plans** on ~150 locations.

### 2.5.2. Mandal et al. [2025]

| Aspect | Content |
|--------|---------|
| Question | Tactical **workforce sizing** and **shift scheduling** for last-mile parcel delivery under demand uncertainty |
| Data | Computational study: **8,000 synthetic** last-mile instances (not field data) |
| Evaluation | Cost per parcel vs outsourcing; shift stability vs flexibility |
| Main result | Completely fixed shifts **9.36%** higher average cost per parcel than extremely flexible schedules; partially flexible shifts **1.89%** above flexible (Mandal et al., 2025, **Conclusions**, computational campaign—not reported in the abstract) |
| Limitations | Parcel last-mile, not meal; no ML forecast pipeline; no manual-plan benchmark |

### 2.5.3. Ozyavas et al. [2025]

| Aspect | Content |
|--------|---------|
| Question | Integrate **shift planning** with pickup-and-delivery under **limited courier availability** |
| Data | Case-based / optimization study |
| Evaluation | Routing + shift cost |
| Main result | Integrated model vs sequential planning |
| Limitations | General courier PD, not meal weekly planning vs manual slots |

### 2.5.4. Cheng et al. [2025]

| Aspect | Content |
|--------|---------|
| Question | **Robust workforce management** with contracted and ad-hoc couriers |
| Data | Solomon VRPTW-based numerical tests + simulations |
| Evaluation | Out-of-sample cost vs empirical optimization |
| Main result | Robust satisficing beats historical-scenario cost minimization |
| Limitations | Platform crowd model; not restaurant network manual plan |

### 2.5.5. Earlier related workforce papers

- **Ulmer and Savelsbergh [2020]:** scheduled vs crowdsourced capacity; continuous approximation for shifts.  
- **Behrendt et al. [2023]:** prescriptive ML for **crowd platform** shift scheduling.  
- **Dai and Liu [2020]:** O2O capacity mix (in-house, full-time/part-time crowd).  
- **Dai et al. [2021]:** in-house vs crowdsourcing staffing trade-offs.

These establish **methods to compare against** but do not evaluate **manual weekly slot plans** with **Yandex/guarantee-style proxies**.

## 2.6. Service operations: flexible demand, periodic delivery, and forecast-to-shifts

**Villarreal et al. [2015]** model **workforce management and scheduling under flexible demand** in a **healthcare back-office** setting: “flexible demand” means a job may be completed within a **time window**, not at a single due date. They develop a **model and decision-support tool** (simulation-optimization) for that environment—not a general result on **staffing under non-stationary arrival rates** in delivery.

**Smilowitz et al. [2013]** address **periodic delivery** workforce management—linking recurring demand patterns to **route/workforce** decisions.

**Bertsimas and Kallus [2020]** formalize **from predictive to prescriptive analytics**: using forecasts inside decision models. This is the conceptual bridge **forecast → capacity/shifts**, though not meal-specific.

**Implication for this thesis:** Our pipeline explicitly implements **prescriptive use of forecasts** (slots), but evaluates against **manual planning** and **operational proxies**, not only forecast MAE—aligning with the prescriptive analytics view but with a **comparative empirical** design. A dedicated survey of **staffing/scheduling under non-stationary demand** in service operations will be expanded for the final thesis text.

## 2.7. Metrics, simulation validation, and multi-objective comparison (literature basis)

**Forecast accuracy:** WAPE/MAE/RMSE are standard [Hyndman and Athanasopoulos, 2021; Makridakis et al., 2022].

**Capacity/service proxies:** Unfulfilled orders, outsourcing, and fleet size appear in Mandal et al. [2025], Zhang et al. [2026], Cheng et al. [2025]—analogous to our **external coverage** proxy.

**Simulation validation:** Discrete-event and optimization papers often validate models on historical scenarios [Auad et al., 2024; Behrendt et al., 2023]; we will **calibrate** a lightweight simulator on manual plans (details in Methodology chapter)—conceptually similar to comparing simulated and realized cost in workforce OR.

**Pareto / multi-objective trade-offs:** Undercoverage vs overcoverage is a **two-objective** trade-off; **Pareto dominance** is standard terminology in multi-objective optimization when improving one objective worsens another [Ehrgott, 2005]. Detailed simulation-validation methodology will be grounded in the simulation literature in **Chapter 3 (Methodology)**.

**This chapter does not fix** implementation thresholds (e.g. 10% calibration error, bootstrap by location)—those belong to **Chapter 3 (Methodology)**.

## 2.8. Synthesis and research gap

| Stream | Representative works | Gap relative to this thesis |
|--------|----------------------|-----------------------------|
| Hourly ML demand | Yu et al. [2023] | Weekly slots, manual plan, business proxies |
| Meal OMD scheduling | Zhao et al. [2025] | Tactical weekly slots vs manual planner |
| Meal workforce OR | **Zhang et al. [2026]** | Empirical manual baseline; Yandex/guarantee simulation |
| Last-mile tactical shifts | Mandal et al. [2025] | Meal restaurant network; ML forecast pipeline |
| Crowd workforce robustness | Cheng et al. [2025] | Owned network; comparative empirical study |
| Forecasting technology | M5 [2022], CatBoost [2018], Chronos/TimesFM [2024] | Prescriptive slot use + operational KPIs |
| Flexible-demand service OR | Villarreal et al. [2015] | Healthcare time windows; not delivery staffing |

**Research gap (to the best of our knowledge, given the search in Section 2.2):** an **integrated empirical comparison** of **weekly demand forecast → courier slots** against the **actual manual plan** on real restaurant-network data, with **calibrated simulation** of under/overcoverage (external coverage and guarantee) and a **pre-specified Pareto criterion**.

A negative result remains valid.

## 2.9. Implications for methods of this thesis

From the review we will:

- Compare **seasonal naive**, **four-week average**, **linear regression**, and **CatBoost** [Hyndman and Athanasopoulos, 2021; Prokhorenkova et al., 2018; Makridakis et al., 2022].
- Treat **actual manual slots** as operational baseline (not in Zhang et al. [2026] or Mandal et al. [2025]).
- Evaluate with **forecast metrics** and **simulated operational proxies**, informed by workforce OR [Zhang et al., 2026; Auad et al., 2024].
- Optionally explore **quantile-based buffers** as robustness extension [Cheng et al., 2025].
- Document **Chronos/TimesFM** as optional/zero-shot baselines if resources allow (Section 2.3.4).

---

## References

1. Ansari A.F., Stella L., Turkmen C., et al. Chronos: Learning the Language of Time Series. *Transactions on Machine Learning Research*, 2024. arXiv:2403.07815.

2. Auad R., Erera A.L., Savelsbergh M.W.P. Capacity requirements and demand management strategies in meal delivery. *EURO Journal on Transportation and Logistics*, 2024, 13, 100135. DOI: 10.1016/j.ejtl.2024.100135.

3. Behrendt A., Savelsbergh M.W.P., Wang H. A prescriptive machine learning method for courier scheduling on crowdsourced delivery platforms. *Transportation Science*, 2023, 57(4), 889–907.

4. Bertsimas D., Kallus N. From predictive to prescriptive analytics. *Management Science*, 2020, 66(3), 1025–1044.

5. Cheng C., Sim M., Zhao Y. Robust workforce management with crowdsourced delivery. *Operations Research*, 2025, 73(2), 595–612.

6. Dai H., Liu P. Workforce planning for O2O delivery systems with crowdsourced drivers. *Annals of Operations Research*, 2020, 291, 219–245.

7. Dai H., Liu Y., Yan N., Zhou W. Optimal staffing for online-to-offline on-demand delivery systems. *Asia-Pacific Journal of Operational Research*, 2021, 38(1), 2050037.

8. Das A., Kong W., Sen R., Zhou Y. A decoder-only foundation model for time-series forecasting. *ICML*, 2024. arXiv:2310.10688.

9. Hyndman R.J., Athanasopoulos G. *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021. https://otexts.com/fpp3/

10. Makridakis S., Spiliotis E., Assimakopoulos V. The M5 accuracy competition: Results, findings and conclusions. *International Journal of Forecasting*, 2022, 38(4), 1346–1364.

11. Mandal M.P., Santini A., Archetti C. Tactical workforce sizing and scheduling decisions for last-mile delivery. *European Journal of Operational Research*, 2025, 323(1), 153–169.

12. Mao W., Ming L., Rong Y., Tang C.S., Zheng H. On-demand meal delivery platforms: Operational level data and research opportunities. *Manufacturing & Service Operations Management*, 2022, 24(5), 2535–2542.

13. Ozyavas P., Ursavas E., Buijs P., Teunter R. Integrating shift planning and pick-up and delivery problems under limited courier availability. *European Journal of Operational Research*, 2025, 326(2), 343–356.

14. Prokhorenkova L., Gusev G., Vorobev A., Dorogush A.V., Gulin A. CatBoost: unbiased boosting with categorical features. *NeurIPS*, 2018.

15. Smilowitz K., Nowak M., Jiang T. Workforce management in periodic delivery operations. *Transportation Science*, 2013, 47(2), 214–230.

16. Steever J., Karwan M.H., Murray C.C. Dynamic courier routing for a food delivery service. *Computers & Operations Research*, 2019, 107, 173–188.

17. Ulmer M.W., Savelsbergh M.W.P. Workforce scheduling in the era of crowdsourced delivery. *Transportation Science*, 2020, 54(4), 1113–1133.

18. Ulmer M.W., Thomas B.W., Campbell A.M., Woyak N. The restaurant meal delivery problem: Dynamic pickup and delivery with deadlines and random ready times. *Transportation Science*, 2021, 55(1), 75–100.

19. Villarreal M.C., Goldsman D., Keskinocak P. Workforce management and scheduling under flexible demand. *Service Science*, 2015, 7(4), 331–351.

20. Yildiz B., Savelsbergh M.W.P. Provably high-quality solutions for the meal delivery routing problem. *Transportation Science*, 2019, 53(5), 1372–1388.

21. Yu X., Lan A., Mao H. Short-term demand prediction for on-demand food delivery with attention-based convolutional LSTM. *Systems*, 2023, 11(10), 485.

22. Zhang J., Zhang Y., Baldacci R., Tang J. Workforce planning for meal deliveries with ad-hoc drivers: A distributionally robust contextual optimization approach. *European Journal of Operational Research*, 2026, 330(2), 427–443.

23. Zhao Y., Alfandari L., Archetti C. Stochastic scheduling and routing decisions in online meal delivery platforms with mixed force. *European Journal of Operational Research*, 2025, 323(1), 139–152.

24. Ehrgott M. *Multicriteria Optimization*. 2nd ed. Springer, 2005.

---

*End of draft Chapters 1–2.*
