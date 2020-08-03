
from src.settings import *
from src.utils import (prefilter_items, get_raw_data_splits, get_price_list, get_bought_ever_list,
                        get_item_commodities_list, preprare_features, check_valid_items, postfilter_items)
from src.metrics import money_precision_at_k
from src.recommenders import MainRecommender

def eval_routine(DATA_PATH,
                 TEST_PATH,
                 ITEM_FEATURES_PATH,
                 USER_FEATURES_PATH,
                 TEST_SIZE_WEEKS,
                 N_POPULAR_ITEMS,
                 INIT_NUM_RECS,
                 N_FIN_RECS):
    data_train_lvl_1 = get_raw_data_splits(DATA_PATH, mode=0)
    data_val_lvl_1 = get_raw_data_splits(TEST_PATH, mode=0)
    item_features, user_features = preprare_features(ITEM_FEATURES_PATH, USER_FEATURES_PATH)
    data_train_lvl_1 = prefilter_items(data_train_lvl_1, N_POPULAR_ITEMS)  # Prefilter routine
    data_val_lvl_1 = prefilter_items(data_val_lvl_1, N_POPULAR_ITEMS)  # Prefilter routine
    itemid_to_price = get_price_list(data_train_lvl_1, data_val_lvl_1)
    user_bought_history = get_bought_ever_list(data_train_lvl_1)
    item_to_commodity = get_item_commodities_list(item_features)
    recommender = MainRecommender(data_train_lvl_1, itemid_to_price)
    result_lvl_1 = data_val_lvl_1.groupby('user_id')['item_id'].unique().reset_index()
    result_lvl_1.columns = ['user_id', 'actual']
    result_lvl_1['base_rec'] = result_lvl_1['user_id'].apply(
        lambda x: recommender.get_own_recommendations(x, N=INIT_NUM_RECS))
    #     result_lvl_1['als_rec'] = result_lvl_1['user_id'].apply(lambda x: recommender.get_als_recommendations(x, N=INIT_NUM_RECS))
    result_lvl_1 = postfilter_items(result_lvl_1,
                                    recommender.overall_top_purchases,
                                    item_to_commodity,
                                    itemid_to_price,
                                    user_bought_history,
                                    n=N_FIN_RECS)

    res = result_lvl_1.apply(lambda row: money_precision_at_k(row['result'],
                                                              row['actual'],
                                                              itemid_to_price,
                                                              k=5), axis=1).mean()
    print(f"Result money precision @ 5 metric: {res}")
    if SAVE_RESULTS:
        result_lvl_1[['user_id', 'result']].to_csv('YN_recs.csv', index=False)
    return res


if __name__ == "__main__":
    eval_routine(DATA_PATH,
                 TEST_PATH,
                 ITEM_FEATURES_PATH,
                 USER_FEATURES_PATH,
                 TEST_SIZE_WEEKS,
                 N_POPULAR_ITEMS,
                 INIT_NUM_RECS,
                 N_FIN_RECS)