import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    
    """ALS Rec system
    
    Input
    -----
    user_item_matrix: pd.DataFrame

    """
    
    def __init__(self, data, features, weighting=True):
                
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
#         self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]


        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
#         self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
                        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        # List item_id == CTM
        self.ctm = self.get_ctm(features)
        self.ctm_itemid_to_id = {k: v for k, v in self.itemid_to_id.items() if k in self.ctm}
        
        
        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T
            
#         # формат saprse matrix
        self.sparse_user_item = csr_matrix(self.user_item_matrix).T.tocsr()
        
        self.model = self.fit(self.user_item_matrix)
        
     
    @staticmethod
    def prepare_matrix(data, value_pivot='quantity', agg='count'): 
        
        
        """Output:
        pivot table over target field
        formated for implicit func
        """
            
        print("Preparing ui matrix...", end='')
        ui_matrix = pd.pivot_table(data, 
                                      index='user_id', columns='item_id', 
                                      values=value_pivot,
                                      aggfunc=agg, 
                                      fill_value=0
                                     )

        ui_matrix = ui_matrix.astype(float) # необходимый тип матрицы для implicit
        print('\033[94mDone\033[0m')
        
        return ui_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
    
    @staticmethod
    def get_ctm(item_features):
        
        # Dict {item_id: 0/1}
        # Deprecated
        # item_features['ctm'] = item_features['brand']=='Private'
        # item_features['ctm'] = item_features['ctm'].astype('uint8')
        # is_ctm = item_features[['item_id', 'ctm']].groupby(['item_id']).mean().to_dict()['ctm']
        
        # Dict {item_id: 1}
        ctm_ids = item_features[item_features['brand']=='Private']['item_id'].unique()

        return ctm_ids
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
   

    @staticmethod
    def fit(user_item_matrix, n_factors=32, regularization=0.001, iterations=15, num_threads=8):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model
    
    
    def _extend_with_top_popular(self, recommendations, N=5):
        
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""
        
        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]
        return recommendations
    
    
    def get_similar_ctm_item(self, item_id):
        
        """Находит товар, похожий на item_id"""
        
        # Товар похож на себя -> рекомендуем 2 товара
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        top_rec = recs[1][0]
#         print(id_to_itemid)
        res = self.id_to_itemid[top_rec]
        return res
        
        
        
        return self.id_to_itemid[top_rec]


    def get_recommendations(self, user, N=5, filter_=None):
        
        
        res = [self.id_to_itemid[rec[0]] for rec in 
                    self.model.recommend(userid=self.userid_to_id[user], 
                                    user_items=self.sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=filter_, 
                                    recalculate_user=True)]
        return res
    

    def get_own_recommendations(self, user, N=5):
        
        """Рекомендуем товары среди тех, которые юзер уже купил"""
        
        return self.get_recommendations(user)

    
    
    def get_similar_items_recommendation(self, user, N=5):
        
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        # 1. топ-N покупок юзера по не-СТМ товарам с сортировкой по кол-ву
        top_user_nonctm_purchases = self.top_purchases[self.top_purchases['user_id'] == user]\
                                               [~self.top_purchases['item_id'].isin(self.ctm)][:N]
                
        # 2. Для каждого товара по эмбеддингам находим ближайший СТМ
        res = top_user_nonctm_purchases['item_id'].apply(lambda x: self.get_similar_ctm_item(x)).tolist()

        return res
    

    def get_similar_users_recommendation(self, user, N=5):
        
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)

        return res