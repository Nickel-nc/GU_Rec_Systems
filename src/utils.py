import pandas as pd
import numpy as np


def prefilter_items_(data, take_n_popular=5000, margin_slice_rate=0.9):
    
    """Предфильтрация товаров"""
    
    # рачсет цены единицы товара
    data['price'] = data['sales_value'] / data['quantity']
    
    # 1. Удаление товаров, со средней ценой < 1$
    data = data[data['price'] > 1]
    
    # 2. Удаление товаров со средней ценой > 30$
    data = data[data['price'] < 30]
    
    # 3. Удаление 10% товаров c наименьшей выручкой (сдвигает минимум выручки с 1.1$ до 94.8$)
    marginality = data.groupby('item_id')['sales_value'].sum().reset_index()
    ten_percent_slice_idx = int(marginality.shape[0] * margin_slice_rate)

    top_margin = marginality.sort_values('sales_value', ascending=False)[:ten_percent_slice_idx].item_id.tolist()
    data = data[data['item_id'].isin(top_margin)]
    
    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    top_popular = popularity.sort_values('quantity', ascending=False)[:take_n_popular].item_id.tolist()
    data = data[data['item_id'].isin(top_popular)]
    
    return data