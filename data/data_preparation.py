import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import featuretools as ft

def details_correction():
    data = pd.read_csv("CreditTraining.csv")
    
    nai = data["Net_Annual_Income"] #net annual income = nai
    aux = []
    for nbr in nai: #the numbers with "," cannot be transformed to float so we replace the ","
        if isinstance(nbr, str):
            nbr = nbr.replace(",", "")
        aux.append(nbr)
            
    data.drop("Net_Annual_Income", axis = 1, inplace = True)
    data["Net_Annual_Income_transform"] = aux
    data.to_csv("CreditTraining.csv", encoding = 'utf-8',  index = False)
    
    pass

def DeepFeatureSynthesis(): #only numerical attributes
    data = pd.read_csv("CreditTraining.csv")
    
    customers_df = data[["Id_Customer", "BirthDate", "Customer_Open_Date", 
                         "Number_Of_Dependant", "Years_At_Residence", 
                         "Net_Annual_Income", "Years_At_Business", 
                         "Prod_Decision_Date", "Nb_Of_Products"]] 
    
    entity = {"customers": (customers_df, "Id_Customer")}
    feature_matrix_customers, _ = ft.dfs(entities=entity, target_entity="customers")

    
    feature_matrix_customers.to_csv("dfs.csv", encoding = 'utf-8',  index = False)
    pass

def feature_creation(pca = True, cluster = True):
    if pca:
        data_pca = pd.read_csv("data2_no_label_cluster.csv")
        data_pca.fillna(0, inplace = True)
    
        scaled_features = StandardScaler().fit_transform(data_pca.values)
        scaled_features_df = pd.DataFrame(scaled_features, index = data_pca.index, columns = data_pca.columns)
    
        mu = np.mean(scaled_features_df, axis = 0)
        n_comp = 3
    
        pca = PCA(n_components = n_comp)
        pca.fit(scaled_features_df)
        
        Xhat = np.dot(pca.transform(scaled_features_df)[:,:n_comp], pca.components_[:n_comp,:])
        Xhat += mu
        Xhat = Xhat.T
        
        dict = {}    
        name = ''
        for i in range(len(Xhat)):
            name = "pca " + str(i)
            dict[name] = list(Xhat[i,])
            name = ''
        df = pd.DataFrame.from_dict(dict)
        print(pca.explained_variance_ratio_)
        df.to_csv("test_no_label.csv")  
          
    if cluster:
        data_cluster = pd.read_csv("data2_no_label.csv")
        data_cluster.fillna(0, inplace = True)
        scaled_features = StandardScaler().fit_transform(data_cluster.values)
        scaled_features_df = pd.DataFrame(scaled_features, index = data_cluster.index, columns = data_cluster.columns)
        
        #K Means
        kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_features_df)
        group_kmeans = kmeans.labels_ + 1
        data_cluster["KMeans"] = group_kmeans
        print(group_kmeans)
        
        #Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average').fit(scaled_features_df)
        group_agg = agg_clustering.labels_ + 1
        data_cluster["Agglomerative Clustering"] = group_agg
        print(group_agg)
          
        #DBSCAN
        dbscan = DBSCAN(eps = 3, min_samples = 2).fit(scaled_features_df)
        group_dbscan = dbscan.labels_ + 1
        data_cluster["DBSCAN"] = group_dbscan
        print(group_dbscan)
        
        data_cluster.to_csv("data2_no_label_cluster.csv") 
        
        
    pass

if __name__ == "__main__":
    #details_correction()
    #DeepFeatureSynthesis()
    feature_creation(pca=True, cluster = False)
    pass