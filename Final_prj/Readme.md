# Retail Recommender Systems Project Solution

## Overview

Hybrid recommender (als + own + popular)

### Restrictions:

- @5 recommendations
- only > 1$ cost items @ rec
- 2 new items @ rec
- at least 1 expensive item > 7 $ @ rec
- different commodities (unique sub_commodity_desc) @ rec


### Filters:

N_POPULAR_ITEMS = 4000


## Results

Scores (MoneyPrecision@5):

Train-test split baseline 0.2

Two-level baseline metric 0.14

Public test baseline under restrictions: 0.07

Bottom public score: 0.034

Final Public score: 0.277897


