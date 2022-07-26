import gc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler


class KmeansCluster:
    def __init__(self, feats=None, n_clusters=8, seed=42, suffix='all'):
        self.feats = feats
        self.n_clusters = n_clusters
        self.seed = seed
        self.suffix = suffix

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
        else:
            pass

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
            f'fe_kmeans_cluster_id_{self.suffix}': cluster_id
        })

        res[f'fe_kmeans_cluster_id_{self.suffix}'] = res[f'fe_kmeans_cluster_id_{self.suffix}'].astype('category')

        # Distance from cluster center
        tmp = self.kmeans.transform(tmp)
        column_names = [f'fe_kmeans_distance_from_cluster_{k}_{self.suffix}' for k in range(self.n_clusters)]
        tmp = pd.DataFrame(tmp, columns=column_names)
        tmp = tmp.astype(float)
        res = pd.concat([res, tmp], axis=1)

        del tmp
        gc.collect()

        return res

    def __call__(self, df, phase):
        return self.transform(df, phase)


class PCAExecuter:
    def __init__(self, feats=None, n_components=8, seed=42, suffix='all'):
        self.feats = feats
        self.n_components = n_components
        self.seed = seed
        self.suffix = suffix

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

        # PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.seed)
        self.pca.fit(tmp)

        del tmp
        gc.collect()

    def transform(self, df, phase):
        if self.feats is None:
            self.feats = [c for c in df.select_dtypes(exclude=[object, 'category']).columns if c.startswith('fe')]
        else:
            pass

        if phase == 'train':
            self._prep(df)

        # Apply Prep
        customer_id = df['customer_ID'].values
        tmp = df.copy()
        tmp = self.scaler.transform(tmp[self.feats])
        tmp = pd.DataFrame(tmp, columns=self.feats)
        tmp = tmp.fillna(self.means_from_train)

        res = pd.DataFrame({
            'customer_ID': customer_id,
        })

        #
        tmp = self.pca.transform(tmp)
        column_names = [f'fe_pca_{k}_{self.suffix}' for k in range(self.n_components)]
        tmp = pd.DataFrame(tmp, columns=column_names)
        tmp = tmp.astype(float)
        res = pd.concat([res, tmp], axis=1)

        del tmp
        gc.collect()

        return res

    def __call__(self, df, phase):
        return self.transform(df, phase)


class SVDExecuter:
    def __init__(self, feats=None, n_components=8, seed=42, suffix='all'):
        self.feats = feats
        self.n_components = n_components
        self.seed = seed
        self.suffix = suffix

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

        # PCA
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
        self.svd.fit(tmp)

        del tmp
        gc.collect()

    def transform(self, df, phase):
        if self.feats is None:
            self.feats = [c for c in df.select_dtypes(exclude=[object, 'category']).columns if c.startswith('fe')]
        else:
            pass

        if phase == 'train':
            self._prep(df)

        # Apply Prep
        customer_id = df['customer_ID'].values
        tmp = df.copy()
        tmp = self.scaler.transform(tmp[self.feats])
        tmp = pd.DataFrame(tmp, columns=self.feats)
        tmp = tmp.fillna(self.means_from_train)

        res = pd.DataFrame({
            'customer_ID': customer_id,
        })

        #
        tmp = self.svd.transform(tmp)
        column_names = [f'fe_svd_{k}_{self.suffix}' for k in range(self.n_components)]
        tmp = pd.DataFrame(tmp, columns=column_names)
        tmp = tmp.astype(float)
        res = pd.concat([res, tmp], axis=1)

        del tmp
        gc.collect()

        return res

    def __call__(self, df, phase):
        return self.transform(df, phase)
