import pandas as pd
import featuretools as ft

def DeepFeatureSynthesis():

    data = pd.read_csv("CreditTraining.csv")
    
    customers_df = data[["Id_Customer", "BirthDate", "Customer_Open_Date", 
                         "Number_Of_Dependant", "Years_At_Residence", "Net_Annual_Income",
                         "Years_At_Business", "Prod_Decision_Date",
                          "Nb_Of_Products"]]
    
    entity = {"customers": (customers_df, "Id_Customer")}
    #relationships = [("customers", "Id_Customer")]
    feature_matrix_customers, features_defs = ft.dfs(entities=entity,
                                                      
                                                    target_entity="customers")

    nai = feature_matrix_customers["Net_Annual_Income"]
    aux = []

    for nbr in nai:
        if isinstance(nbr, str):
            nbr = nbr.replace(",", "")
        aux.append(nbr)
            
    feature_matrix_customers.drop("Net_Annual_Income", axis = 1, inplace = True)
    
    feature_matrix_customers["Net_Annual_Income_transform"] = aux

    feature_matrix_customers.to_csv("dfs.csv", encoding = 'utf-8',  index = False)
    
    
    pass

if __name__ == "__main__":
    DeepFeatureSynthesis()
    pass