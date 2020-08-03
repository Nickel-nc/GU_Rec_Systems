import numpy as np

############################
# Key usable metrics for RC
############################


def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


# Recall - доля рекомендованных товаров среди релевантных = Какой % купленных товаров был среди рекомендованных

# Recall= (# of recommended items that are relevant) / (# of relevant items)  
# Recall@k = (# of recommended items @k that are relevant) / (# of relevant items)
# Money Recall@k = (revenue of recommended items @k that are relevant) / (revenue of relevant items)

def recall_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    
    flags = np.isin(bought_list, recommended_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:k])
    
    flags = np.isin(bought_list, recommended_list)
    
    recall = np.dot(flags, prices_bought) / np.dot(np.ones(len(bought_list)), prices_bought)
    
    return recall


# Precision - доля релевантных товаров среди рекомендованных = Какой % рекомендованных товаров  юзер купил
# Money Precision@k = (revenue of recommended items @k that are relevant) / (revenue of recommended items @k) 

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:k])
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = np.dot(flags, prices_bought) / np.dot(np.ones(k), prices_recommended)
    
    
    
    return precision


