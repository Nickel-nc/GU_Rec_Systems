import pandas as pd
import numpy as np

# print func formatter
text_string_template = '\033[94m[[text]]\033[0m'
value_string_template = '\033[91m[[value]]\033[0m'

def prefilter_items(data, take_n_popular=5000, margin_slice_rate=0.9):
    
    """Предфильтрация товаров
    
    INPUTS:
    
    data: pd.DataFrame датасет
    n_popular: отсечка наименее продаваемых товаров по кол-ву суммарных продаж (=output size)
    margin_slice_rate: отсечка доли товарных категорий с минимальной выручкой 
    """
    
    print("Prefilter items...", end='')
    n_before = value_string_template.replace('[[value]]', str(data['item_id'].nunique()))
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
    n_after = value_string_template.replace('[[value]]', str(data['item_id'].nunique()))
    
    print(f"Data reduced from: {n_before} to: {n_after} samples...", end='')
    print('\033[94mDone\033[0m')
    
    return data