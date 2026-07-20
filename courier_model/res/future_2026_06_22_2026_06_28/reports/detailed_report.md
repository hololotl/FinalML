# Подробный отчёт по прогнозу слотов

## Segment + time_segment

| segment | time_segment | locations | predicted_orders | auto_slots | bike_slots | total_slots | mean_orders_per_window | mean_slots_per_window | orders_per_slot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medium | block_12_18 | 51 | 1,604.1 | 154 | 193 | 347 | 4.49 | 0.97 | 4.62 |
| high | lunch | 15 | 1,991.1 | 167 | 173 | 340 | 18.96 | 3.24 | 5.86 |
| high | morning | 15 | 891.9 | 147 | 164 | 311 | 8.49 | 2.96 | 2.87 |
| high | afternoon | 15 | 1,009.9 | 148 | 161 | 309 | 9.62 | 2.94 | 3.27 |
| mega | lunch | 8 | 1,743.1 | 136 | 158 | 294 | 31.13 | 5.25 | 5.93 |
| medium | block_18_24 | 52 | 972.5 | 136 | 154 | 290 | 2.67 | 0.80 | 3.35 |
| mega | morning | 8 | 809.4 | 126 | 154 | 280 | 14.45 | 5.00 | 2.89 |
| high | dinner | 15 | 925.9 | 127 | 143 | 270 | 8.82 | 2.57 | 3.43 |
| high | early_morning | 15 | 336.3 | 87 | 162 | 249 | 3.20 | 2.37 | 1.35 |
| mega | afternoon | 8 | 864.0 | 104 | 141 | 245 | 15.43 | 4.38 | 3.53 |
| mega | early_morning | 8 | 365.7 | 110 | 135 | 245 | 6.53 | 4.38 | 1.49 |
| mega | dinner | 8 | 834.8 | 96 | 105 | 201 | 14.91 | 3.59 | 4.15 |
| medium | block_06_12 | 45 | 1,099.0 | 44 | 146 | 190 | 3.49 | 0.60 | 5.78 |
| high | opening | 15 | 349.5 | 29 | 132 | 161 | 3.33 | 1.53 | 2.17 |
| high | late_evening | 15 | 413.3 | 52 | 69 | 121 | 3.94 | 1.15 | 3.42 |
| mega | late_evening | 8 | 400.1 | 42 | 67 | 109 | 7.15 | 1.95 | 3.67 |
| mega | opening | 8 | 293.2 | 44 | 63 | 107 | 5.24 | 1.91 | 2.74 |
| high | deep_night | 15 | 299.0 | 0 | 98 | 98 | 2.85 | 0.93 | 3.05 |
| medium | block_00_06 | 6 | 218.2 | 59 | 0 | 59 | 5.90 | 1.59 | 3.70 |
| mega | deep_night | 8 | 239.1 | 56 | 0 | 56 | 4.27 | 1.00 | 4.27 |
| low | block_00_06 | 6 | 148.8 | 14 | 19 | 33 | 4.65 | 1.03 | 4.51 |
| low | block_06_12 | 48 | 96.2 | 21 | 7 | 28 | 0.29 | 0.08 | 3.43 |
| low | block_12_18 | 65 | 103.5 | 8 | 0 | 8 | 0.23 | 0.02 | 12.93 |
| low | block_18_24 | 63 | 97.1 | 2 | 5 | 7 | 0.22 | 0.02 | 13.87 |

## Top 30 peak windows

| location_id | segment_datetime | segment | time_segment | orders_prediction | auto_order_prediction | bike_order_prediction | auto_slots_needed | bike_slots_needed | total_slots_needed | auto_slot_capacity_source | bike_slot_capacity_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 2026-06-24 12:00:00+03:00 | mega | lunch | 40.06 | 26.35 | 13.71 | 7 | 2 | 9 | location_shift_history | location_shift_history |
| 29 | 2026-06-26 12:00:00+03:00 | mega | lunch | 39.74 | 26.14 | 13.60 | 7 | 2 | 9 | location_shift_history | location_shift_history |
| 29 | 2026-06-25 12:00:00+03:00 | mega | lunch | 39.48 | 25.97 | 13.51 | 7 | 2 | 9 | location_shift_history | location_shift_history |
| 29 | 2026-06-22 12:00:00+03:00 | mega | lunch | 39.44 | 25.94 | 13.50 | 7 | 2 | 9 | location_shift_history | location_shift_history |
| 29 | 2026-06-23 12:00:00+03:00 | mega | lunch | 39.35 | 25.88 | 13.47 | 7 | 2 | 9 | location_shift_history | location_shift_history |
| 29 | 2026-06-27 12:00:00+03:00 | mega | lunch | 38.11 | 25.06 | 13.04 | 7 | 2 | 9 | location_shift_history | location_shift_history |
| 29 | 2026-06-27 09:00:00+03:00 | mega | morning | 19.53 | 13.50 | 6.03 | 7 | 2 | 9 | location_shift_history | location_shift_history |
| 29 | 2026-06-28 12:00:00+03:00 | mega | lunch | 34.97 | 23.00 | 11.97 | 6 | 2 | 8 | location_shift_history | location_shift_history |
| 29 | 2026-06-28 09:00:00+03:00 | mega | morning | 18.21 | 12.59 | 5.62 | 6 | 2 | 8 | location_shift_history | location_shift_history |
| 29 | 2026-06-22 09:00:00+03:00 | mega | morning | 17.43 | 12.05 | 5.38 | 6 | 2 | 8 | location_shift_history | location_shift_history |
| 29 | 2026-06-26 09:00:00+03:00 | mega | morning | 17.02 | 11.77 | 5.25 | 6 | 2 | 8 | location_shift_history | location_shift_history |
| 29 | 2026-06-25 09:00:00+03:00 | mega | morning | 16.92 | 11.70 | 5.22 | 6 | 2 | 8 | location_shift_history | location_shift_history |
| 29 | 2026-06-24 09:00:00+03:00 | mega | morning | 16.71 | 11.55 | 5.16 | 6 | 2 | 8 | location_shift_history | location_shift_history |
| 29 | 2026-06-23 09:00:00+03:00 | mega | morning | 16.70 | 11.54 | 5.15 | 6 | 2 | 8 | location_shift_history | location_shift_history |
| 69 | 2026-06-27 12:00:00+03:00 | mega | lunch | 50.67 | 21.03 | 29.64 | 3 | 4 | 7 | location_shift_history | location_shift_history |
| 69 | 2026-06-28 12:00:00+03:00 | mega | lunch | 46.15 | 19.16 | 26.99 | 3 | 4 | 7 | location_shift_history | location_shift_history |
| 69 | 2026-06-28 09:00:00+03:00 | mega | morning | 23.79 | 10.22 | 13.57 | 3 | 4 | 7 | location_shift_history | location_shift_history |
| 69 | 2026-06-27 09:00:00+03:00 | mega | morning | 23.58 | 10.13 | 13.45 | 3 | 4 | 7 | location_shift_history | location_shift_history |
| 29 | 2026-06-25 15:00:00+03:00 | mega | afternoon | 19.64 | 12.41 | 7.23 | 5 | 2 | 7 | location_shift_history | location_shift_history |
| 29 | 2026-06-22 15:00:00+03:00 | mega | afternoon | 19.54 | 12.34 | 7.19 | 5 | 2 | 7 | location_shift_history | location_shift_history |
| 29 | 2026-06-26 06:00:00+03:00 | mega | early_morning | 8.48 | 6.28 | 2.21 | 5 | 2 | 7 | location_shift_history | location_shift_history |
| 29 | 2026-06-25 06:00:00+03:00 | mega | early_morning | 8.45 | 6.26 | 2.20 | 5 | 2 | 7 | location_shift_history | location_shift_history |
| 29 | 2026-06-27 06:00:00+03:00 | mega | early_morning | 8.42 | 6.23 | 2.19 | 5 | 2 | 7 | location_shift_history | location_shift_history |
| 29 | 2026-06-24 06:00:00+03:00 | mega | early_morning | 8.09 | 5.98 | 2.10 | 5 | 2 | 7 | location_shift_history | location_shift_history |
| 29 | 2026-06-23 06:00:00+03:00 | mega | early_morning | 8.06 | 5.96 | 2.10 | 5 | 2 | 7 | location_shift_history | location_shift_history |
| 29 | 2026-06-22 06:00:00+03:00 | mega | early_morning | 7.85 | 5.81 | 2.04 | 5 | 2 | 7 | location_shift_history | location_shift_history |
| 69 | 2026-06-26 12:00:00+03:00 | mega | lunch | 44.43 | 18.44 | 25.99 | 3 | 3 | 6 | location_shift_history | location_shift_history |
| 69 | 2026-06-23 12:00:00+03:00 | mega | lunch | 44.16 | 18.33 | 25.83 | 3 | 3 | 6 | location_shift_history | location_shift_history |
| 69 | 2026-06-25 12:00:00+03:00 | mega | lunch | 43.64 | 18.12 | 25.52 | 3 | 3 | 6 | location_shift_history | location_shift_history |
| 69 | 2026-06-22 12:00:00+03:00 | mega | lunch | 41.91 | 17.40 | 24.51 | 3 | 3 | 6 | location_shift_history | location_shift_history |

## Top 30 locations by total slots

| location_id | segment | is_grouped_location | predicted_orders | auto_slots | bike_slots | total_slots | max_window_orders | max_window_slots | orders_per_slot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | mega | False | 818.8 | 207 | 77 | 284 | 40.06 | 9 | 2.88 |
| 69 | mega | False | 996.0 | 110 | 121 | 231 | 50.67 | 7 | 4.31 |
| 7 | mega | False | 599.2 | 118 | 86 | 204 | 26.14 | 5 | 2.94 |
| 99 | mega | False | 809.0 | 7 | 184 | 191 | 39.30 | 6 | 4.24 |
| 6 | mega | False | 531.9 | 79 | 97 | 176 | 28.53 | 6 | 3.02 |
| 30 | mega | False | 557.0 | 94 | 70 | 164 | 26.90 | 5 | 3.40 |
| 39 | mega | False | 596.2 | 92 | 71 | 163 | 30.70 | 5 | 3.66 |
| 4 | high | False | 520.7 | 50 | 110 | 160 | 27.73 | 5 | 3.25 |
| 33 | high | False | 458.7 | 68 | 80 | 148 | 23.47 | 4 | 3.10 |
| 115 | high | False | 420.0 | 95 | 52 | 147 | 22.79 | 5 | 2.86 |
| 21 | high | False | 459.6 | 83 | 61 | 144 | 20.54 | 4 | 3.19 |
| 35 | high | False | 555.0 | 73 | 67 | 140 | 27.50 | 4 | 3.96 |
| 14 | high | False | 339.9 | 88 | 47 | 135 | 16.24 | 4 | 2.52 |
| 12 | high | False | 439.3 | 47 | 83 | 130 | 23.20 | 4 | 3.38 |
| 17 | high | False | 329.8 | 72 | 54 | 126 | 14.27 | 4 | 2.62 |
| 101 | high | False | 326.6 | 93 | 33 | 126 | 14.51 | 4 | 2.59 |
| 8 | mega | False | 641.5 | 7 | 117 | 124 | 33.79 | 4 | 5.17 |
| 9 | high | False | 319.7 | 59 | 56 | 115 | 9.84 | 3 | 2.78 |
| 126 | high | False | 439.6 | 0 | 105 | 105 | 26.38 | 3 | 4.19 |
| 2 | high | False | 437.4 | 0 | 103 | 103 | 22.88 | 3 | 4.25 |
| 123 | high | False | 365.5 | 21 | 81 | 102 | 25.39 | 3 | 3.58 |
| 128 | high | False | 492.0 | 0 | 92 | 92 | 23.66 | 3 | 5.35 |
| 25 | high | False | 313.2 | 8 | 78 | 86 | 15.61 | 3 | 3.64 |
| 11 | medium | False | 263.5 | 54 | 0 | 54 | 22.90 | 5 | 4.88 |
| 335 | low | False | 187.8 | 17 | 24 | 41 | 19.64 | 4 | 4.58 |
| 18 | medium | False | 285.7 | 18 | 21 | 39 | 22.79 | 3 | 7.33 |
| grp_19_54 | medium | True | 161.6 | 36 | 0 | 36 | 14.18 | 3 | 4.49 |
| 434 | medium | False | 271.5 | 14 | 21 | 35 | 20.45 | 2 | 7.76 |
| 20 | medium | False | 201.2 | 7 | 21 | 28 | 16.42 | 2 | 7.19 |
| 15 | medium | False | 151.2 | 7 | 21 | 28 | 8.65 | 1 | 5.40 |

## Capacity source counts

| source | auto_rows | bike_rows |
| --- | --- | --- |
| default_slot_capacity | 0 | 230 |
| location_shift_history | 952 | 861 |
| segment_shift_history | 1,304 | 1,302 |
| time_segment_shift_history | 1,356 | 1,219 |