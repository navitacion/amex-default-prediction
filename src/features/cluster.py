import gc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KmeansCluster:
    def __init__(self, feats=None, n_clusters=8, seed=42):
        self.feats = feats
        self.n_clusters = n_clusters
        self.seed = seed

    def _prep(self, df):
        tmp = df.copy()
        # StandardScaling
        self.scaler = StandardScaler()
        tmp = self.scaler.fit_transform(tmp[self.feats])

        # fill na from mean value from Train data
        # 平均が欠損の場合はすべて0にする
        means = np.where(np.isnan(self.scaler.mean_), 0, self.scaler.mean_)
        self.means_from_train = {k: v for k, v in zip(self.feats, means)}
        tmp = pd.DataFrame(tmp, columns=self.feats)
        tmp = tmp.fillna(self.means_from_train)

        # KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed)
        self.kmeans.fit(tmp)

        del tmp
        gc.collect()

    def transform(self, df, phase):

        if self.feats is None:
            self.feats = [c for c in df.select_dtypes(exclude=[object, 'category']).columns if c.startswith('fe')]

        if phase == 'train':
            self._prep(df)

        # Apply Prep
        customer_id = df['customer_ID'].values
        tmp = df.copy()
        tmp = self.scaler.transform(tmp[self.feats])
        tmp = pd.DataFrame(tmp, columns=self.feats)
        tmp = tmp.fillna(self.means_from_train)

        # Cluster ID
        cluster_id = self.kmeans.predict(tmp)

        res = pd.DataFrame({
            'customer_ID': customer_id,
            'fe_kmeans_cluster_id': cluster_id
        })

        res['fe_kmeans_cluster_id'] = res['fe_kmeans_cluster_id'].astype('category')

        # Distance from cluster center
        tmp = self.kmeans.transform(tmp)
        tmp = pd.DataFrame(tmp, columns=[f'fe_kmeans_distance_from_cluster_{k}' for k in range(self.n_clusters)])
        res = pd.concat([res, tmp], axis=1)

        del tmp
        gc.collect()

        return res

    def __call__(self, df, phase):
        return self.transform(df, phase)
