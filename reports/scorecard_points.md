# Scorecard points table

Logistic regression on WoE bins of the application features, fitted on the whole development set (246,008 applicants). Scale: 600 points at good:bad odds of 50:1, 20 points to double the odds. Higher scores mean lower risk.

An applicant's score is the base points plus the points for the bin they fall into on each of the 17 features. **Base points: 557.**

| Feature | Bin | Share of applicants | Default rate | WoE | Points |
|---|---|---|---|---|---|
| `EXT_SOURCE_MEAN` | < 0.3495 | 15.0% | 20.5% | -1.079 | -20 |
| `EXT_SOURCE_MEAN` | 0.3495 to 0.4136 | 10.0% | 12.3% | -0.473 | -9 |
| `EXT_SOURCE_MEAN` | 0.4136 to 0.4837 | 15.0% | 8.5% | -0.061 | -1 |
| `EXT_SOURCE_MEAN` | 0.4837 to 0.5242 | 10.0% | 6.6% | +0.216 | +4 |
| `EXT_SOURCE_MEAN` | 0.5242 to 0.5635 | 10.0% | 5.5% | +0.417 | +8 |
| `EXT_SOURCE_MEAN` | 0.5635 to 0.6024 | 10.0% | 4.1% | +0.726 | +14 |
| `EXT_SOURCE_MEAN` | 0.6024 to 0.6667 | 15.0% | 3.5% | +0.876 | +16 |
| `EXT_SOURCE_MEAN` | >= 0.6667 | 15.0% | 2.2% | +1.355 | +25 |
| `EXT_SOURCE_MEAN` | Missing | 0.1% | 8.6% | -0.110 | -2 |
| `EXT_SOURCE_3` | < 0.2837 | 12.0% | 17.9% | -0.908 | -11 |
| `EXT_SOURCE_3` | 0.2837 to 0.4084 | 12.0% | 10.5% | -0.287 | -3 |
| `EXT_SOURCE_3` | 0.4084 to 0.4758 | 8.1% | 8.1% | +0.002 | +0 |
| `EXT_SOURCE_3` | 0.4758 to 0.5638 | 12.0% | 6.1% | +0.301 | +4 |
| `EXT_SOURCE_3` | 0.5638 to 0.643 | 12.0% | 4.8% | +0.552 | +7 |
| `EXT_SOURCE_3` | 0.643 to 0.6941 | 8.0% | 4.2% | +0.702 | +9 |
| `EXT_SOURCE_3` | >= 0.6941 | 16.1% | 3.4% | +0.927 | +11 |
| `EXT_SOURCE_3` | Missing | 19.8% | 9.3% | -0.152 | -2 |
| `EXT_SOURCE_1` | < 0.2558 | 6.5% | 15.8% | -0.760 | -3 |
| `EXT_SOURCE_1` | 0.2558 to 0.4027 | 8.7% | 9.6% | -0.189 | -1 |
| `EXT_SOURCE_1` | 0.4027 to 0.5054 | 6.5% | 7.0% | +0.149 | +1 |
| `EXT_SOURCE_1` | 0.5054 to 0.7102 | 13.1% | 5.2% | +0.480 | +2 |
| `EXT_SOURCE_1` | >= 0.7102 | 8.7% | 3.0% | +1.044 | +4 |
| `EXT_SOURCE_1` | Missing | 56.3% | 8.5% | -0.059 | -0 |
| `EMPLOYED_YEARS` | < 1.687 | 16.4% | 11.3% | -0.374 | -3 |
| `EMPLOYED_YEARS` | 1.687 to 2.513 | 8.2% | 10.9% | -0.334 | -3 |
| `EMPLOYED_YEARS` | 2.513 to 3.951 | 12.3% | 10.2% | -0.261 | -2 |
| `EMPLOYED_YEARS` | 3.951 to 5.12 | 8.2% | 9.0% | -0.115 | -1 |
| `EMPLOYED_YEARS` | 5.12 to 6.727 | 8.2% | 8.0% | +0.012 | +0 |
| `EMPLOYED_YEARS` | 6.727 to 8.695 | 8.2% | 7.2% | +0.126 | +1 |
| `EMPLOYED_YEARS` | 8.695 to 11.99 | 8.2% | 6.3% | +0.261 | +2 |
| `EMPLOYED_YEARS` | >= 11.99 | 12.3% | 4.8% | +0.547 | +5 |
| `EMPLOYED_YEARS` | Missing | 17.9% | 5.4% | +0.437 | +4 |
| `OCCUPATION_TYPE` | Accountants | 3.2% | 4.6% | +0.592 | +5 |
| `OCCUPATION_TYPE` | Cleaning staff | 1.5% | 9.5% | -0.184 | -2 |
| `OCCUPATION_TYPE` | Cooking staff | 1.9% | 10.5% | -0.291 | -2 |
| `OCCUPATION_TYPE` | Core staff | 9.0% | 6.4% | +0.253 | +2 |
| `OCCUPATION_TYPE` | Drivers | 6.1% | 11.3% | -0.368 | -3 |
| `OCCUPATION_TYPE` | High skill tech staff | 3.7% | 6.3% | +0.270 | +2 |
| `OCCUPATION_TYPE` | Laborers | 18.0% | 10.5% | -0.292 | -2 |
| `OCCUPATION_TYPE` | Managers | 6.9% | 6.2% | +0.280 | +2 |
| `OCCUPATION_TYPE` | Medicine staff | 2.8% | 7.0% | +0.156 | +1 |
| `OCCUPATION_TYPE` | Missing | 31.3% | 6.5% | +0.235 | +2 |
| `OCCUPATION_TYPE` | Other | 3.0% | 10.1% | -0.251 | -2 |
| `OCCUPATION_TYPE` | Sales staff | 10.4% | 9.6% | -0.190 | -2 |
| `OCCUPATION_TYPE` | Security staff | 2.2% | 10.7% | -0.310 | -3 |
| `CREDIT_GOODS_RATIO` | < 1.106 | 44.5% | 6.5% | +0.240 | +5 |
| `CREDIT_GOODS_RATIO` | 1.106 to 1.132 | 9.5% | 7.1% | +0.139 | +3 |
| `CREDIT_GOODS_RATIO` | 1.132 to 1.158 | 10.2% | 7.2% | +0.122 | +2 |
| `CREDIT_GOODS_RATIO` | 1.158 to 1.211 | 15.4% | 9.1% | -0.136 | -3 |
| `CREDIT_GOODS_RATIO` | 1.211 to 1.396 | 14.4% | 11.0% | -0.345 | -7 |
| `CREDIT_GOODS_RATIO` | >= 1.396 | 5.9% | 13.2% | -0.552 | -11 |
| `CREDIT_GOODS_RATIO` | Missing | 0.1% | 7.7% | +0.026 | +1 |
| `ORGANIZATION_TYPE` | Business Entity Type 1 | 2.0% | 8.0% | +0.002 | +0 |
| `ORGANIZATION_TYPE` | Business Entity Type 2 | 3.4% | 8.6% | -0.074 | -1 |
| `ORGANIZATION_TYPE` | Business Entity Type 3 | 22.1% | 9.3% | -0.151 | -1 |
| `ORGANIZATION_TYPE` | Construction | 2.2% | 11.9% | -0.435 | -3 |
| `ORGANIZATION_TYPE` | Government | 3.4% | 6.9% | +0.174 | +1 |
| `ORGANIZATION_TYPE` | Industry: type 3 | 1.1% | 10.8% | -0.322 | -2 |
| `ORGANIZATION_TYPE` | Industry: type 9 | 1.1% | 6.7% | +0.206 | +2 |
| `ORGANIZATION_TYPE` | Kindergarten | 2.3% | 7.1% | +0.133 | +1 |
| `ORGANIZATION_TYPE` | Medicine | 3.7% | 6.7% | +0.194 | +1 |
| `ORGANIZATION_TYPE` | Other | 19.0% | 7.6% | +0.066 | +1 |
| `ORGANIZATION_TYPE` | School | 2.9% | 5.8% | +0.348 | +3 |
| `ORGANIZATION_TYPE` | Security | 1.1% | 10.5% | -0.292 | -2 |
| `ORGANIZATION_TYPE` | Self-employed | 12.4% | 10.2% | -0.258 | -2 |
| `ORGANIZATION_TYPE` | Trade: type 3 | 1.1% | 10.2% | -0.260 | -2 |
| `ORGANIZATION_TYPE` | Trade: type 7 | 2.5% | 9.4% | -0.171 | -1 |
| `ORGANIZATION_TYPE` | Transport: type 4 | 1.8% | 9.1% | -0.131 | -1 |
| `ORGANIZATION_TYPE` | XNA | 17.9% | 5.4% | +0.437 | +3 |
| `REGION_RATING_CLIENT_W_CITY` | < 2 | 11.1% | 4.9% | +0.540 | +6 |
| `REGION_RATING_CLIENT_W_CITY` | 2 to 3 | 74.6% | 7.9% | +0.023 | +0 |
| `REGION_RATING_CLIENT_W_CITY` | >= 3 | 14.3% | 11.5% | -0.388 | -4 |
| `NAME_EDUCATION_TYPE` | Higher education | 24.3% | 5.4% | +0.432 | +7 |
| `NAME_EDUCATION_TYPE` | Incomplete higher | 3.4% | 8.6% | -0.073 | -1 |
| `NAME_EDUCATION_TYPE` | Lower secondary | 1.2% | 11.1% | -0.351 | -6 |
| `NAME_EDUCATION_TYPE` | Other | 0.1% | 1.6% | +1.459 | +24 |
| `NAME_EDUCATION_TYPE` | Secondary / secondary special | 71.0% | 8.9% | -0.108 | -2 |
| `DAYS_LAST_PHONE_CHANGE` | < -2,159 | 10.0% | 5.3% | +0.441 | +3 |
| `DAYS_LAST_PHONE_CHANGE` | -2,159 to -1,886 | 5.0% | 6.2% | +0.287 | +2 |
| `DAYS_LAST_PHONE_CHANGE` | -1,886 to -1,569 | 10.0% | 6.3% | +0.268 | +2 |
| `DAYS_LAST_PHONE_CHANGE` | -1,569 to -1,228 | 10.0% | 6.9% | +0.166 | +1 |
| `DAYS_LAST_PHONE_CHANGE` | -1,228 to -1,058 | 5.0% | 7.5% | +0.082 | +1 |
| `DAYS_LAST_PHONE_CHANGE` | -1,058 to -757 | 10.0% | 8.3% | -0.033 | -0 |
| `DAYS_LAST_PHONE_CHANGE` | -757 to -363 | 20.0% | 9.2% | -0.148 | -1 |
| `DAYS_LAST_PHONE_CHANGE` | >= -363 | 30.0% | 9.5% | -0.179 | -1 |
| `DAYS_LAST_PHONE_CHANGE` | Missing | 0.0% | 0.0% | -1.334 | -10 |
| `ANNUITY_CREDIT_RATIO` | < 0.03226 | 15.0% | 5.8% | +0.356 | +5 |
| `ANNUITY_CREDIT_RATIO` | 0.03226 to 0.04538 | 25.0% | 6.7% | +0.209 | +3 |
| `ANNUITY_CREDIT_RATIO` | >= 0.04538 | 60.0% | 9.2% | -0.147 | -2 |
| `ANNUITY_CREDIT_RATIO` | Missing | 0.0% | 0.0% | +0.612 | +9 |
| `FLOORSMAX_AVG` | < 0.1667 | 10.7% | 8.8% | -0.093 | -1 |
| `FLOORSMAX_AVG` | 0.1667 to 0.2917 | 22.0% | 7.1% | +0.140 | +1 |
| `FLOORSMAX_AVG` | 0.2917 to 0.375 | 11.1% | 6.1% | +0.297 | +3 |
| `FLOORSMAX_AVG` | >= 0.375 | 6.5% | 5.1% | +0.499 | +5 |
| `FLOORSMAX_AVG` | Missing | 49.7% | 9.2% | -0.141 | -1 |
| `DAYS_ID_PUBLISH` | < -4,429 | 20.0% | 6.1% | +0.301 | +1 |
| `DAYS_ID_PUBLISH` | -4,429 to -4,047 | 15.0% | 6.8% | +0.182 | +1 |
| `DAYS_ID_PUBLISH` | -4,047 to -3,255 | 15.0% | 7.9% | +0.022 | +0 |
| `DAYS_ID_PUBLISH` | -3,255 to -2,953 | 5.0% | 8.2% | -0.015 | -0 |
| `DAYS_ID_PUBLISH` | -2,953 to -1,720 | 20.0% | 8.9% | -0.105 | -1 |
| `DAYS_ID_PUBLISH` | -1,720 to -1,055 | 10.0% | 9.5% | -0.175 | -1 |
| `DAYS_ID_PUBLISH` | -1,055 to -376 | 10.0% | 9.9% | -0.223 | -1 |
| `DAYS_ID_PUBLISH` | >= -376 | 5.0% | 10.4% | -0.276 | -1 |
| `FLAG_DOCUMENT_3` | < 1 | 29.0% | 6.2% | +0.287 | +4 |
| `FLAG_DOCUMENT_3` | >= 1 | 71.0% | 8.8% | -0.100 | -1 |
| `DAYS_REGISTRATION` | < -9,936 | 10.0% | 5.6% | +0.387 | +2 |
| `DAYS_REGISTRATION` | -9,936 to -9,023 | 5.0% | 6.2% | +0.289 | +2 |
| `DAYS_REGISTRATION` | -9,023 to -7,480 | 10.0% | 7.0% | +0.147 | +1 |
| `DAYS_REGISTRATION` | -7,480 to -6,771 | 5.0% | 7.4% | +0.099 | +1 |
| `DAYS_REGISTRATION` | -6,771 to -5,457 | 10.0% | 7.9% | +0.024 | +0 |
| `DAYS_REGISTRATION` | -5,457 to -1,065 | 45.0% | 8.7% | -0.078 | -0 |
| `DAYS_REGISTRATION` | -1,065 to -692 | 5.0% | 9.2% | -0.139 | -1 |
| `DAYS_REGISTRATION` | >= -692 | 10.0% | 9.8% | -0.213 | -1 |
| `REG_CITY_NOT_LIVE_CITY` | < 1 | 92.2% | 7.7% | +0.050 | +0 |
| `REG_CITY_NOT_LIVE_CITY` | >= 1 | 7.8% | 12.4% | -0.473 | -3 |
| `OWN_CAR_AGE` | < 9 | 15.7% | 5.8% | +0.362 | +6 |
| `OWN_CAR_AGE` | 9 to 17 | 11.2% | 8.0% | +0.005 | +0 |
| `OWN_CAR_AGE` | >= 17 | 7.1% | 9.4% | -0.172 | -3 |
| `OWN_CAR_AGE` | Missing | 66.0% | 8.5% | -0.054 | -1 |
