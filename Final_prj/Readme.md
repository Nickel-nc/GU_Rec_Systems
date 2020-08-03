# Retail Recommender Systems Project Solution

## Overview

Unpacked recommender sandbox:

[FP_Test.ipynb](https://github.com/Nickel-nc/GU_Rec_Systems/blob/master/Final_prj/FP_Test.ipynb)

Main module:

[Run_1lvl.py](https://github.com/Nickel-nc/GU_Rec_Systems/blob/master/Final_prj/Run_1lvl.py) + src folder

### Run project locally


```
cd {ROOT}

# Pull the image
$ docker pull nickelnc/item-recommender-app

# Copy initial data
$ mkdir -p ./item-recommender-app/data
$ cp /path/to/unpacked/data/*.csv ./data

# Run app locally
$ docker run item-recommender-app

```

### Features

Hybrid recommender (als + own + popular)

#### Restrictions:

- @5 recommendations
- only > 1$ cost items @ rec
- 2 new items @ rec
- at least 1 expensive item > 7 $ @ rec
- different commodities (unique sub_commodity_desc) @ rec


### Results

Scores (MoneyPrecision@5):

Train-test split baseline 0.2

Two-level baseline metric 0.14

Public test baseline under restrictions: 0.07

Bottom public score: 0.034

Final Public score: 0.277897


