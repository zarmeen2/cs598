# Bitcoin Price Prediction Report

By: Zarmeen Hasan

## Introduction

This report goes over my process for determining the best model for predicting the next day's Bitcoin price. I use the Bitcoin dataset provided in the assignment's instructions. The data contains variables such as Bitcoin market price, market cap, trade volume, cost per transaction, number of transactions, etc. The goal of the project is to predict hte Bitcoin market price (`btc_market_price`). This dataset is a time-series dataset so I use lag features (i.e. previous days' data) in order to predict the next day's price.

In the following sections I will walk through the data cleaning, data analysis, and model selection steps I took and finally end with my results and conclusion. All methodologies and practices use the concepts taught in the class thus far. Using these concepts, I conclude that the best model type (out of those taught in class) to predict Bitcoin market price is Ridge regression.

## Data Cleaning Process

### Missing and Zero Values

The first thing I checked for was missing values in the data. Luckily, there was only one variable with 21 missing values: `btc_trade_volume`. Because this is a time-series dataset, it's risky to simply remove these rows. Removing rows could result in gaps in the timeline. Thus, I performed a forward fill on the missing `btc_trade_volume`values. This means, days when `btc_trade_volume` is missing, they'll now be populated by the `btc_trade_volume` value from the day before.

The second thing I checked for was values less than or equal ot zero. It's important to check for negative values because in the context of the data, it would not make sense. There can never be a negative price of Bitcoin or a negative number of transactions. I did not find any negative values in the data. However, I did find zeros in the data: 

```
Columns with 0.0 values:  Index(['btc_market_price', 'btc_market_cap', 'btc_trade_volume',
       'btc_blocks_size', 'btc_n_orphaned_blocks',
       'btc_median_confirmation_time', 'btc_miners_revenue',
       'btc_transaction_fees', 'btc_cost_per_transaction',
       'btc_estimated_transaction_volume_usd'],
      dtype='object')
```

Looking through these columns, it would only make sense for `btc_n_orphaned_blocks` to have zero values on days there were no blocks orphaned. The rest cannot be zero because Bitcoin prices never reach zero. Bitcoin is so volatile and it will always have some value associated with it. This made me wonder if these values were only zero in early years when the Bitcoin did not exist or there were no transactions that day. These are the date ranges when each variable has zeros: 

```
Column 'btc_market_price' has 0 values from 2010-02-23 00:00:00 to 2010-08-16 00:00:00
Column 'btc_market_cap' has 0 values from 2010-02-23 00:00:00 to 2010-08-16 00:00:00
Column 'btc_trade_volume' has 0 values from 2010-02-23 00:00:00 to 2010-07-16 00:00:00
Column 'btc_blocks_size' has 0 values from 2010-02-23 00:00:00 to 2010-07-13 00:00:00
Column 'btc_n_orphaned_blocks' has 0 values from 2010-02-23 00:00:00 to 2018-02-20 00:00:00
Column 'btc_median_confirmation_time' has 0 values from 2010-02-23 00:00:00 to 2011-12-01 00:00:00
Column 'btc_miners_revenue' has 0 values from 2010-02-23 00:00:00 to 2010-08-16 00:00:00
Column 'btc_transaction_fees' has 0 values from 2010-02-23 00:00:00 to 2010-11-25 00:00:00
Column 'btc_cost_per_transaction' has 0 values from 2010-02-23 00:00:00 to 2010-08-16 00:00:00
Column 'btc_estimated_transaction_volume_usd' has 0 values from 2010-02-23 00:00:00 to 2010-08-16 00:00:00
```

Majority of the variables have only have zeros in 2010. This was very early on in the data's history, meaning the Bitcoin might not have existed then, so the zeros are justified. I already determined `btc_n_orphaned_blocks` can be zero, but `btc_median_confirmation_time` can only be zero if there were no transactions that day. We already know `btc_n_transactions` has no zero nor missing values, so rows where `btc_median_confirmation_time` are zero should be considered "bad" rows. I filtered out rows where the following variables were zero:

- `btc_market_price`
- `btc_market_cap`
- `btc_trade_volume`
- `btc_blocks_size`
- `btc_median_confirmation_time`
- `btc_miners_revenue`
- `btc_transaction_fees`
- `btc_cost_per_transaction`
- `btc_estimated_transaction_volume_usd`

I finally checked for duplicate rows as these would introduce noise and bias when creating lag features, but there were none. 

## Data Analysis & Model Selection



## Results

## Conclusion