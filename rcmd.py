import tensorflow as tf
import numpy as np
from multiprocessing import Process, Queue

class MF_Mean_Model(tf.keras.Model):
    def __init__(self, user_ids, item_ids, rating_mean, rank):
        super().__init__()
        # user id들을 사전으로 구성하고 user id를 입력을 받아 이를 1부터 시작하는 인덱스로 변환해주는 Layer를 생성합니다. OOV: 0
        self.lu_user = tf.keras.layers.IntegerLookup(
            vocabulary = tf.constant(user_ids)
        )
        # item id들을 사전으로 구성하고 user id를 입력을 받아 이를 1부터 시작하는 인덱스로 변환해주는 Layer를 생성합니다. OOV: 0
        self.lu_item = tf.keras.layers.IntegerLookup(
            vocabulary = tf.constant(item_ids)
        )
        
        # user id에 대한 상대적 평균을 나타내는 임베딩입니다.
        self.emb_user_mean = tf.keras.layers.Embedding(len(user_ids) + 1, 1)
        # item id에 대한 상대적 평균을 나타내는 임베딩입니다. 
        self.emb_item_mean = tf.keras.layers.Embedding(len(item_ids) + 1, 1)
        
        # 사용자 행렬을 나타내는 사용자 임베딩입니다. 사이즈는 rank 입니다
        self.emb_user = tf.keras.layers.Embedding(len(user_ids) + 1, rank)
        # 영화 행렬을 나타내는 사용자 임베딩입니다.
        self.emb_item = tf.keras.layers.Embedding(len(item_ids) + 1, rank)

        # 점수의 평균입니다. 상수 텐서로 상수인 float32 텐서로 저장합니다.
        self.rating_mean = rating_mean
        
        # 맵핑된 사용자 임베딩과 맵핑된 아이템 임베딩을 Row-wise inner product 연산을 해주는 layer입니다. 
        self.dot = tf.keras.layers.Dot(axes=-1)

    def build(self):   
        # 1. 각 Lookup 레이어 build
        #   user id, item id는 모두 정수형이므로 동일한 shape로 처리
        self.lu_user.build((None, 1))
        self.lu_item.build((None, 1))
    
        # 2. Lookup 후 Embedding 레이어 shape 계산
        user_lu_shape = self.lu_user.compute_output_shape((None, 1))
        item_lu_shape = self.lu_item.compute_output_shape((None, 1))
    
        # 3. Embedding 레이어 build
        self.emb_user_mean.build(user_lu_shape)
        self.emb_item_mean.build(item_lu_shape)
        self.emb_user.build(user_lu_shape)
        self.emb_item.build(item_lu_shape)
    
        # 4. Dot layer build
        # Dot([user_vec, item_vec])의 입력 shape = (batch, rank)
        user_vec_shape = self.emb_user.compute_output_shape(user_lu_shape)
        item_vec_shape = self.emb_item.compute_output_shape(item_lu_shape)
        self.dot.build([user_vec_shape, item_vec_shape])

    def call(self, X, training=False):
        x_user = self.lu_user(X['user id'])# 사용자 ID에서 임베딩의 위치 인덱스로 변환합니다. X['user id'] N×1 정수 / x_user: N×1 정수
        user_vec = self.emb_user(x_user) #사용자 임베딩을 가져옵니다. user_vec: N×rank 실수
        user_mean = self.emb_user_mean(x_user) # 사용자의 상대적 평균을 가져옵니다. user_mean: N×1 실수
        
        x_item = self.lu_item(X['item id']) # 아이템 ID에서 임베딩 인덱스로 변환합니다. X['item id'] N×1 정수 / x_movie: N×1 정수
        item_vec =  self.emb_item(x_item) # 아이템 임베딩을 가져옵니다. user_vec: N×rank 실수
        item_mean = self.emb_item_mean(x_item) # 아이템의 상대적 평균을 가져옵니다.
        
        return user_mean + item_mean + self.dot([user_vec, item_vec]) + self.rating_mean
    
    def predict_by_userid(self, user_id, item_ids):
        """
            call에서의 연산은 (사용자1, 아이템1), (사용자2, 아이템2), ... 의 예측 연산입니다.
            실제 모델은 사용자 단위로 다수의 아이템에 대한 예측을 하게 됩니다.
            이러한 상황을 고려한 더욱 최적화한 루틴으로,
            한명의 사용자에게 여러 개의 아이템의 평점을 예측합니다.
        """
        
        x_user = self.lu_user(tf.constant([user_id])) # 사용자 ID에서 임베딩 인덱스로 변환합니다. X['user id'] (1) / x_user (1)
        user_vec = self.emb_user(x_user) # 사용자 임베딩을 가져옵니다. emb_user: 1×rank개의 실수
        # item id 별로 user id의 임베딩을 가져오는 작업과, item id 별로 반복되는 user_emb를 만들 필요가 없고 단일 벡터와 
        # item id 행렬의 곱의 연산이 되므로 계산량이 줄어 들게 됩니다.
        x_item = self.lu_item(item_ids) # 아이템 ID에서 임베딩 인덱스로 변환합니다. X['item id'] N개의 정수 / x_item: N개의 정수
        item_vec = self.emb_item(x_item)
        return tf.squeeze(
            tf.matmul(item_vec, user_vec, transpose_b=True) + 
            self.emb_item_mean(x_item) +
            self.emb_user_mean(x_user) + 
            self.rating_mean
        )

    def get_user_vec(self, user_id):
        x_user = self.lu_user(
            tf.constant([user_id])
        ) # 사용자 ID에서 임베딩 인덱스로 변환합니다. X['user id'] (1) / x_user (1)
        return tf.concat([
            tf.squeeze(self.emb_user(x_user)), tf.constant([1.0])
        ], axis=-1) # 사용자 임베딩을 가져옵니다. emb_user: 1×rank개의 실수

    def get_item_vecs(self, item_ids):
        x_item = self.lu_item(
            tf.constant(item_ids)
        )
        return tf.concat([
            self.emb_item(x_item), self.emb_item_mean(x_item)
        ], axis=-1)

class EmbModel(tf.keras.Model):
    def __init__(self, ids, size, l2=0):
        """
        정수형 ID로 된 임베딩을 받아서 임베딩 벡터를 반환시켜 주는 모델
        Parameters: 
            ids: list
                카테고리의 수준들의 id 리스트
            size: int
                임베딩 벡터의 사이즈
            l2: float
                l2 규제 계수
        """
        super().__init__()
        # 정수형 ID를 1-based Index로 반환해주는 Layer를 만듭니다. - 0은 OOV(out-of-vocabulary)입니다.
        self.lu_ids = tf.keras.layers.IntegerLookup(
            vocabulary=tf.constant(ids)
        )
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        # Index에 대한 Embedding을 반환하는 Layer를 생성합니다.
        self.emb = tf.keras.layers.Embedding(len(ids) + 1, size, embeddings_regularizer=reg)
        self.output_size = size

    def build(self):
        input_shape = (None, 1)
        # 1. 각 Lookup 레이어 build
        self.lu_ids.build(input_shape)
    
        # 2. Lookup 후 Embedding 레이어 shape 계산
        lu_shape = self.lu_ids.compute_output_shape(input_shape)
    
        # 3. Embedding 레이어 build
        self.emb.build(lu_shape)
    
    def call(self, x, training=False):
        x = self.lu_ids(x) # 정수형 ID(Nx1) → Embedding Index (N×1)
        return self.emb(x, training=training) # Embedding Index 
    
# 평균, 사용자 평균 모델, 영화 평균 모델을(사이즈가 1인 임베딩 모델) 받아 평균 모델을 구성합니다.
class MeanModel(tf.keras.Model):
    def __init__(self, mean, user_ids, movie_ids):
        """
        Parameters:
            mean: float
                평균
            user_mean_model: EmbModel
                사용자 평균 모델
            movie_mean_mode: EmbModel
                영화 평균 모델
        """
        super().__init__()
        
        self.mean = mean
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        # 사용자의 평균을 담고 있는 모델
        self.mean_ = tf.constant([mean], dtype=tf.float32)
        self.user_mean_model = EmbModel(user_ids, 1)
        self.movie_mean_model = EmbModel(movie_ids, 1)

    def build(self):
        self.user_mean_model.build()
        self.movie_mean_model.build()

    def call(self, x, training=False):
        return self.mean_ + \
            self.user_mean_model(x['userId'], training=training) + \
            self.movie_mean_model(x['movieId'], training=training)

    def get_model_data(self):
        return {
            'mean': self.mean,
            'user_ids': self.user_ids,
            'movie_ids': self.movie_ids,
            'user_mean_model': self.user_mean_model.get_weights(),
            'movie_mean_model': self.movie_mean_model.get_weights()
        }

    def from_model_data(model_data):
        model = MeanModel(
            model_data['mean'], model_data['user_ids'], model_data['movie_ids']
        )
        model.build()
        model.user_mean_model.set_weights(model_data['user_mean_model'])
        model.movie_mean_model.set_weights(model_data['movie_mean_model'])
        return model
    
    def predict_by_userid(self, x):
        """
            call에서의 연산은 (사용자1, 아이템1), (사용자2, 아이템2), ... 의 예측 연산입니다.
            실제 모델은 사용자 단위로 다수의 아이템에 대한 예측을 하게 됩니다.
            이러한 상황을 고려한 더욱 최적화한 루틴으로,
            한명의 사용자에게 여러 개의 아이템의 평점을 예측합니다.
        """
        
        x_user = self.user_mean_model(
            x['userId'], # N
            training=False
        )
        # item id 별로 user id의 임베딩을 가져오는 작업과, item id 별로 반복되는 user_emb를 만들 필요가 없고 단일 벡터와 
        # item id 행렬의 곱의 연산이 되므로 계산량이 줄어 들게 됩니다.
        x_movie = self.movie_mean_model(
            x['movieIds'], # N
            training=False
        )
        return tf.squeeze(x_user + x_movie + self.mean)

    def get_user_vec(self, user_id, **argv):
        return tf.constant([1.0])

    def get_item_vecs(self, item_ids, **argv):
        return self.movie_mean_model(
            tf.constant(item_ids)
        )
    
# 제공한 모델 각각의 예측 결과를 더하는 모델을 만듭니다.
class AdditiveModel(tf.keras.Model):
    def __init__(self, models):
        """
        Parameters:
            models: list
                tf.keras.Model 객체로 이루진 리스트입니다.
        """
        super().__init__()
        self.models = models

    def build(self):
        for i in self.models:
            i.build()

    def call(self, x, training=False):
        # 각각의 모델에서 나온 출력을 모으기 위한 리스트 입니다.
        y_hat = []
        for i in self.models:
            y_hat.append(i(x, training=training))
        # tf.reduce_sum: 주어진 텐서의 합을 구해줍니다. axis = 0: 첫번째 차원(모델의 위치를 나타내는 차원)에 대한 합을 구합니다.
        return tf.reduce_sum(y_hat, axis=0)

    def predict_by_userid(self, x):
        """
            call에서의 연산은 (사용자1, 아이템1), (사용자2, 아이템2), ... 의 예측 연산입니다.
            실제 모델은 사용자 단위로 다수의 아이템에 대한 예측을 하게 됩니다.
            이러한 상황을 고려한 더욱 최적화한 루틴으로,
            한명의 사용자에게 여러 개의 아이템의 평점을 예측합니다.
        """
        y_hat = []
        for i in self.models:
            y_hat.append(i.predict_by_userid(x))
        return tf.reduce_sum(y_hat, axis=0)
        
    def get_user_vec(self, user_id, **argv):
        return tf.concat(
            [i.get_user_vec(user_id, **argv) for i in self.models], axis=-1
        )

    def get_item_vecs(self, item_ids, **argv):
        return tf.concat(
            [i.get_item_vecs(item_ids, **argv) for i in self.models], axis=-1
        )
    
class UserHistModel(tf.keras.Model):
    """
    사용자가 이전에 평가한 영화와 평점을 입력 받는 모델입니다.
    """
    def __init__(self, user_ids, user_emb_size, movie_model, output_size, hidden_units = [], l2=0):
        """
        Parameters
            user_ids: np.array
                사용자 id
            movie_model: tf.keras.Model
                영화 모델
            output_size: int
                출력 벡터의 수
            l2: float
                L2 규제, 0일 때는 규제를 사용하지 않습니다.
        """
        super().__init__()
        self.user_model = EmbModel(user_ids, user_emb_size)
        self.movie_model = movie_model
        
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        
        # 밀도 은닉층
        self.hidden_layers = [
            tf.keras.layers.Dense(
                i, activation='relu', 
                kernel_initializer=tf.keras.initializers.HeNormal(), 
                kernel_regularizer=reg
            ) for i in hidden_units
        ]      
        
        # 출력층(Output Layer)
        if output_size > 0:
            self.o = tf.keras.layers.Dense(
                output_size, kernel_initializer = tf.keras.initializers.GlorotNormal(), kernel_regularizer=reg
            )
        else:
            self.o = None

        self.output_size = output_size
        # 사용자 벡터, 이전 시청 영화 벡터, 평점을 결합하기 위한 결합층(Concatenate Layer)을 생성합니다.
        self.cc = tf.keras.layers.Concatenate(axis = -1)

    def build(self):
        movie_model_shape = (None, self.movie_model.output_size)
        self.user_model.build()
        user_model_shape = (None, self.user_model.output_size)
        shape_ = self.cc.compute_output_shape([user_model_shape, movie_model_shape, (None, 1)])
        for i in self.hidden_layers:
            i.build(shape_)
            shape_ = i.compute_output_shape(shape_)
        if self.o is not None:
            self.o.build(shape_)
        else:
            self.output_size = shape_[-1]
        
    def call(self, x, prev_movieId, prev_rating, training=False):
        x = self.cc([
             self.user_model(x), # 사용자 벡터를 가져옵니다. N×rank
             self.movie_model(prev_movieId), # 이전 시청 영화 벡터를 가져옵니다. N×rank
             tf.expand_dims(prev_rating, axis=-1) # 이전 평점. N×1
        ]) # N×(2×rank + 1)
        
        for i in self.hidden_layers:
            x = i(x)
        if self.o is not None:
            return self.o(x) # 출력층. N×rank
        else:
            return x
    
class MovieInfoModel(tf.keras.Model):
    def __init__(self, movieinfo, movie_ids, emb_config, output_size, hidden_units = [], l2=0):
        super().__init__()
        self.ids = movieinfo['ids']
        self.lu_movie = tf.keras.layers.IntegerLookup(
            vocabulary=movieinfo['ids'][1:]
        )
        # 영화 별 장르 정보를 담고 저장공간생성
        self.genres = tf.constant(movieinfo['genre'])
        # 영화의 컬렉션 정보를 지니고 있는 저장공간 생성
        self.collection = tf.constant(movieinfo['collection'])
        # 영화의 제목 + 줄거리의 OpenAI에서 구한 Embedding 정보 저장공간 생성
        self.ov_emb = tf.constant(movieinfo['ov_emb'])
        self.release_date = tf.constant(movieinfo['release_date'], dtype=tf.int64)
        self.movie_model = EmbModel(movie_ids, emb_config['movie'])
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        self.emb_collection = tf.keras.layers.Embedding(
            np.max(movieinfo['collection']) + 1, 
            emb_config['collection'], 
            embeddings_regularizer = reg
        )

        # 결합 레이어
        self.cc = tf.keras.layers.Concatenate(axis = -1)
        
        # 밀도 은닉층
        self.hidden_layers = [
            tf.keras.layers.Dense(
                i, activation='relu', 
                kernel_initializer=tf.keras.initializers.HeNormal(), 
                kernel_regularizer=reg
            ) for i in hidden_units
        ]
        
        # 출력 레이어: Dense(rank)
        if output_size > 0:
            self.o = tf.keras.layers.Dense(
                output_size, kernel_initializer = tf.keras.initializers.GlorotNormal(), kernel_regularizer=reg
            )
        else:
            self.o = None
        self.output_size = output_size

    def get_movieinfo(self):
        return {
            'ids': self.ids,
            'genre': self.genres.numpy(),
            'collection': self.collection.numpy(),
            'ov_emb': self.ov_emb.numpy(),
            'release_date': self.release_date.numpy(),
        }
    def build(self):
        movie_model_shape = (None, self.movie_model.output_size)
        self.movie_model.build()
        self.emb_collection.build((None, ))
        shape_ = self.cc.compute_output_shape(
            [
                movie_model_shape, (None, self.genres.shape[-1]), 
                self.emb_collection.compute_output_shape((None, )), 
                (None, self.ov_emb.shape[-1]),(None, 1)
            ]
        )
        for i in self.hidden_layers:
            i.build(shape_)
            shape_ = i.compute_output_shape(shape_)
        if self.o is not None:
            self.o.build(shape_)
        else:
            self.output_size = shape_[-1]

    def call(self, x, ts, training=False):
        x_movie = self.movie_model(x, training=training)
        
        x = self.lu_movie(x)
        x_genre = tf.gather(self.genres, x) # 장르 여부를 가져옵니다.
        x_collection = tf.gather(self.collection, x) # 컬렉션 번호 가져옵니다.
        x_collection = self.emb_collection(x_collection) # 컬렉션 임베딩을 가져옵니다.
        x_ov_emb =  tf.gather(self.ov_emb, x) # OpenAI API를 얻은 영화 줄거리의 Embedding 정보를 가져 옵니다.
        x_ts = tf.gather(self.release_date, x)
        x_ts = tf.cast(
            (tf.minimum(
                tf.math.log(
                    tf.maximum(
                        tf.cast(ts  - x_ts, tf.float64) // 1e9 // 3600 // 24, 1
                    )
                ), 10
            ) - 5) / 5, tf.float32
        ) # 개봉 경과일의 Log를 -1 ~ 1로 스케일로 만듭니다.
        x =  self.cc([
            x_movie, x_genre, x_collection, x_ov_emb, tf.expand_dims(x_ts, axis = -1)
        ]) # x_movie, x_genre, x_collection, x_ov_emb를 결합하여 하나의 텐서로 만듭니다.
        for i in self.hidden_layers:
            x = i(x)
        if self.o is not None:
            return self.o(x)
        else:
            return x

class UserHistModel_ts(UserHistModel):
    def call(self, x, prev_movieId, prev_rating, prev_ts, training=False):
        x = self.cc([
             self.user_model(x), # 사용자 벡터를 가져옵니다. N×rank
             self.movie_model(prev_movieId, prev_ts), # 이전 시청 영화 벡터를 가져옵니다. N×rank
             tf.expand_dims(prev_rating, axis=-1) # 이전 평점. N×1
        ]) # N×(2×rank + 1)
        
        for i in self.hidden_layers:
            x = i(x)
        if self.o is not None:
            return  self.o(x) # 출력층. N×output_size
        else:
            return x

class UserHistModel2(UserHistModel):
    def __init__(self, user_ids, user_emb_size, movie_model, output_size, hidden_units = [], l2=0, rnn = {'type': 'lstm', 'unit': 32}, hist_cnt=8):
        super().__init__(user_ids, user_emb_size, movie_model, output_size, hidden_units, l2)
        self.rnn_ = rnn
        self.hist_cnt = hist_cnt
        if rnn is None:
            self.rnn = None
        elif rnn['type'] == "lstm":
            self.rnn = tf.keras.layers.LSTM(rnn['unit'])
        elif rnn['type'] == "gru":
            self.rnn = tf.keras.layers.GRU(rnn['unit'])
        else:
            self.rnn = None
        # 평가 이력 벡터와 사용자 벡터를 결합시키기 위한 레이어
        self.cc2 = tf.keras.layers.Concatenate(axis=-1)

    def build(self):
        self.user_model.build()
        user_model_shape = (None, self.user_model.output_size)
        if self.rnn is None:
            hist_model_shape = (None, self.movie_model.output_size + 1)
        else:
            hist_model_shape = self.rnn.compute_output_shape((None, self.hist_cnt, self.movie_model.output_size))
            self.rnn.build((None, self.hist_cnt, self.movie_model.output_size + 1))
        shape_ = self.cc.compute_output_shape([user_model_shape, hist_model_shape])
        for i in self.hidden_layers:
            i.build(shape_)
            shape_ = i.compute_output_shape(shape_)
        self.o.build(shape_)
    
    def call(self, x, prev_movieIds, prev_ratings, prev_ts, training=False):
        hist_vec = self.cc2([
            self.movie_model(prev_movieIds, prev_ts, training=training),
            prev_ratings
        ])
        if self.rnn != None:
            hist_vec = self.rnn(hist_vec) # N×32
        else:
            # rnn을 사용하지 않는 다면 평균을 사용합니다. N×(rank + 1)
            hist_vec =  tf.reduce_mean(hist_vec, axis = -2) 
        x = self.cc([
            self.user_model(x, training=training), 
            hist_vec
        ])
        for i in self.hidden_layers:
            x = i(x)
        return self.o(x)

class MFModel3(tf.keras.Model):
    def __init__(self, 
                 user_ids, user_emb_size, user_hidden_units, rnn, hist_cnt,
                 movieinfo, movie_ids, movie_emb_config, movie_hidden_units,
                 rank
        ):
        super().__init__()
        self.user_ids = user_ids
        self.user_emb_size = user_emb_size
        self.user_hidden_units = user_hidden_units
        self.rnn = rnn
        self.hist_cnt = hist_cnt
        self.movie_ids = movie_ids
        self.movie_emb_config = movie_emb_config
        self.movie_hidden_units = movie_hidden_units
        self.rank = rank
        self.is_built = False
        
        self.movie_model = MovieInfoModel(movieinfo, movie_ids, movie_emb_config, rank, movie_hidden_units)
        self.user_model = UserHistModel2(user_ids, user_emb_size, self.movie_model, rank, user_hidden_units, rnn=rnn, hist_cnt=hist_cnt)
        # Row-wise dot Product를 하도록 설정합니다.
        self.dot = tf.keras.layers.Dot(axes=-1)

    def build(self):
        if self.is_built:
            return
        self.user_model.build()
        self.movie_model.build()
        self.is_built = True
        
    def call(self, x, training=False):
        x_movie = self.movie_model(
            x['movieId'], # N
            x['ts'], # N
            training=training
        ) # N×32
        x_user = self.user_model(
            x['userId'], 
            x['prev_movieIds'], 
            tf.expand_dims(x['prev_ratings'], axis=-1), 
            x['prev_ts'],
            training=training
        )
        x_movie = self.movie_model(x['movieId'], x['ts'], training=training)
        return self.dot([x_user, x_movie])

    def get_model_data(self):
        return {
            'user_ids': self.user_ids,
            'user_emb_size': self.user_emb_size, 
            'user_hidden_units': list(self.user_hidden_units),
            'rnn': dict(self.rnn) if self.rnn is not None else None,
            'hist_cnt': self.hist_cnt,
            'movieinfo': self.movie_model.get_movieinfo(), 
            'movie_ids': self.movie_ids,
            'movie_emb_config': dict(self.movie_emb_config),
            'movie_hidden_units': list(self.movie_hidden_units),
            'user_model': self.user_model.get_weights(),
            'movie_model': self.movie_model.get_weights(),
            'rank': self.rank
        }

    def from_model_data(model_data):
        model = MFModel3(
            model_data['user_ids'],  model_data['user_emb_size'], model_data['user_hidden_units'], model_data.get('rnn', None),
            model_data['movieinfo'], model_data['movie_ids'], model_data['movie_emb_config'], model_data['movie_hidden_units'],
            model_data['rank']
        )
        model.build()
        model.user_model.set_weights(model_data['user_model'])
        model.movie_model.set_weights(model_data['movie_model'])
        return model

    def predict_by_userid(self, x):
        """
            call에서의 연산은 (사용자1, 아이템1), (사용자2, 아이템2), ... 의 예측 연산입니다.
            실제 모델은 사용자 단위로 다수의 아이템에 대한 예측을 하게 됩니다.
            이러한 상황을 고려한 더욱 최적화한 루틴으로,
            한명의 사용자에게 여러 개의 아이템의 평점을 예측합니다.
        """
        
        x_user = self.user_model(
            x['userId'],
            x['prev_movieIds'], 
            tf.expand_dims(x['prev_ratings'], axis=-1), 
            x['prev_ts'],
            training=False
        )
        # item id 별로 user id의 임베딩을 가져오는 작업과, item id 별로 반복되는 user_emb를 만들 필요가 없고 단일 벡터와 
        # item id 행렬의 곱의 연산이 되므로 계산량이 줄어 들게 됩니다.
        x_movie = self.movie_model(
            x['movieIds'], # N
            x['ts'], # N
            training=False
        )
        return tf.squeeze(
            tf.matmul(x_movie, x_user, transpose_b=True)
        )

    def get_user_vec(self, user_id, **argv):
        return tf.squeeze(
            self.user_model(
                tf.constant([user_id]), tf.constant([argv['prev_movieIds']]), 
                tf.expand_dims(tf.constant([argv['prev_ratings']]), axis=-1), tf.constant([argv['prev_ts']])
            )
        )

    def get_item_vecs(self, item_ids, **argv):
        return self.movie_model(tf.constant(item_ids), tf.constant(argv['ts']))

class BatchWoker(Process):
    def __init__(self, q, df, df_user_hist, pos, hist_cnt, batch_size, rating_mean, ts_mean):
        self.q = q
        self.hist_cnt = hist_cnt
        self.batch_size = batch_size
        self.df = df
        self.df_user_hist = df_user_hist
        self.pos = pos
        self.rating_mean = rating_mean
        self.ts_mean = ts_mean
        super().__init__()
    def run(self):
        for i in self.pos:
            piece = self.df.iloc[i:i + self.batch_size]
            seq = piece['seq'].values
            from_idx = (seq - self.hist_cnt).clip(0)
            null_hist_idx = (self.hist_cnt - seq).clip(0)
            df_hist = self.df_user_hist.iloc[piece['offset']]
            movie_id, ts, prev_movieIds, prev_ratings, prev_ts, ratings  = list(), list(), list(), list(), list(), list()
            for seq, from_idx, null_hist_idx, prev_movieIds_, prev_ts_, prev_ratings_ in zip(
                seq, from_idx, null_hist_idx, 
                df_hist['movieId'].values, df_hist['ts'].values, df_hist['rating'].values
            ):
                movie_id.append(prev_movieIds_[seq])
                ts.append(prev_ts_[seq])
                ratings.append(prev_ratings_[seq])
                prev_movieIds_ = prev_movieIds_[from_idx:seq]
                prev_ts_ =  prev_ts_[from_idx:seq]
                prev_ratings_ =  prev_ratings_[from_idx:seq]
                if null_hist_idx > 0:
                    if seq == 0:
                        pad_ts, pad_rating = self.ts_mean, self.rating_mean
                    else:
                        pad_ts, pad_rating = np.mean(prev_ts_), np.mean(prev_ratings_)
                    prev_movieIds_ = np.pad(prev_movieIds_, (0, null_hist_idx), constant_values = 0)
                    prev_ts_ = np.pad(prev_ts_, (0, null_hist_idx), constant_values = pad_ts)
                    prev_ratings_ = np.pad(prev_ratings_, (0, null_hist_idx), constant_values = pad_rating)
                prev_movieIds.append(prev_movieIds_)
                prev_ratings.append(prev_ratings_)
                prev_ts.append(prev_ts_)
            batch = (df_hist.index, movie_id, ts, prev_movieIds, prev_ratings, prev_ts, ratings)
            self.q.put(batch)
        self.q.put(None) # 할당량을 채웠을 경우에 None을 반환합니다.
        
def hist_set_iter(df, df_user_hist, hist_cnt, batch_size, pbar=None, shuffle=True, n_job=4, queue_size=16, rating_mean=0.0, ts_mean = 0.0, equal_batch_size=True):
    """
        Multiprocessing을 통해 Batch 데이터를 만들어 내는 Iterator

        df: pd.DataFrame
            rating 데이터프레임
        df_user_hist: pd.DataFrame
            이력 데이터
        pos: list
            위치 인덱스 리스트
        hist_cnt: int
            이전 평가 이력 수
        batch_size: int
            배치(Batch) 사이즈
        pbar: Tqdm
            Progress 표시기
        shuffle: bool
            데이터를 섞음 여부
        n_job: int
            Process 수
        queue_size: int
            Queue 사이즈
        default_rating: float
            이력 평점이 결측일 경우의 대체할 평점
        equal_batch_size: bool
            마지막 부분의 배치사이즈를 동일하게 맞출지 여부
    """
    if len(df) % batch_size != 0 and equal_batch_size:
        # 전체셋 크기가 batch_size에 나누어 떨어지지 않으면
        # 동일한 Batch 사이즈를 위해 마지막은 끝에서 batch_size만큼 뺀 위치 부터 가져옵니다.
        pos = np.hstack([np.arange(0, len(df) - batch_size, batch_size), [len(df) - batch_size]])
    else:
        pos = np.arange(0, len(df), batch_size)
    chunk = (len(pos) + n_job - 1) // n_job
    if shuffle:
        np.random.shuffle(pos)
    # Work Process 에서 작업한 결과를 저장할 Queue입니다.
    q = Queue(queue_size)
    # n_jobs 만큼 Worker를 생성합니다.
    workers = list()
    for i in range(0, len(pos), chunk):
        workers.append(BatchWoker(q, df, df_user_hist, pos[i: i + chunk], hist_cnt, batch_size, rating_mean, ts_mean))
        workers[-1].start()
    job_finished = 0
    if pbar is not None:
        pbar.total = len(pos)
    while(True):
        val = q.get()
        if val == None: # Worker가 작업을 마치면 None을 반환합니다.
            job_finished +=1 # 작업이 끝난 Worker 카운트
            if job_finished == len(workers): # 모든 Worker가 작업을 끝냈다면 중지
                break
            continue
        yield(val)
        if pbar is not None:
            pbar.update(1)