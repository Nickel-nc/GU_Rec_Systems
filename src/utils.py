import pandas as pd
import numpy as np


value_string_template = '\033[91m[[value]]\033[0m'

def prefilter_items(data, take_n_popular=5000, margin_slice_rate=0.9):
    
    """Предфильтрация товаров"""
    
    n_before = value_string_template.replace('[[value]]', str(data['item_id'].nunique()))
    
    # расчет цены единицы товара
    data['price'] = data['sales_value'] / data['quantity']
    
    # 1. Удаление товаров, со средней ценой < 1$
    data = data[data['price'] > 1]
    
    # 2. Удаление товаров со средней ценой > 30$
    data = data[data['price'] < 30]
    
    # 3. Удаление 10% товаров c наименьшей выручкой (сдвигает минимум выручки с 1.1$ до 94.8$ для unsplitted data)
    marginality = data.groupby('item_id')['sales_value'].sum().reset_index()
    ten_percent_slice_idx = int(marginality.shape[0] * margin_slice_rate)

    top_margin = marginality.sort_values('sales_value', ascending=False)[:ten_percent_slice_idx].item_id.tolist()
    data = data[data['item_id'].isin(top_margin)]
    
    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    top_popular = popularity.sort_values('quantity', ascending=False)[:take_n_popular].item_id.tolist()
    data = data[data['item_id'].isin(top_popular)]
    
    n_after = value_string_template.replace('[[value]]', str(data['item_id'].nunique()))
    print(f"Items variety reduced from: {n_before} to: {n_after} samples...", end='')
    print('\033[94mDone\033[0m')
    
    return data


def get_raw_data_splits(data_path, n_weeks_split):
    
    """
    Returns data splits:
    
    data_train: base train split
    data_test: used for lvl 1 validation & lvl 2 train
    data_val: used for lvl 2 validation
    
    for lvl_size_weeks in [6, 3] returns:   
    train_lvl1: week_no (1-85), val_lvl1 & train_lvl2: week_no (86-91), val_lvl2: week_no (92-95)
    """
    print("Preparing raw data...", end='')
    data = pd.read_csv(data_path)
    
    # drop 0 purchases
    data = data.drop(data[data['quantity']==0].index)
    

    data_train = data[data['week_no'] < data['week_no'].max() - (n_weeks_split[0] + n_weeks_split[1])]
    data_test = data[(data['week_no'] >= data['week_no'].max() /
                           - (n_weeks_split[0] + n_weeks_split[1])) &
                          (data['week_no'] < data['week_no'].max() - (n_weeks_split[1]))]

    data_val_1 = data_test.copy()
    data_val_2 = data[data['week_no'] >= data['week_no'].max() - n_weeks_split[1]] 
    print('\033[94mDone\033[0m') 
    
    return data, data_train, data_test, data_val_1, data_val_2


def get_price_list(data):
    return data.groupby('item_id')['price'].mean().reset_index()
    

def get_raw_features(item_features_path, user_features_path):
    
    """Loads raw item and user features:"""
    
    print("Preparing raw features...", end='')
    item_features = pd.read_csv(item_features_path)
    user_features = pd.read_csv(user_features_path)

    # column processing
    item_features.columns = [col.lower() for col in item_features.columns]
    user_features.columns = [col.lower() for col in user_features.columns]
    item_features.rename(columns={'product_id': 'item_id'}, inplace=True)
    user_features.rename(columns={'household_key': 'user_id'}, inplace=True)    
    print('\033[94mDone\033[0m')
    
    return item_features, user_features