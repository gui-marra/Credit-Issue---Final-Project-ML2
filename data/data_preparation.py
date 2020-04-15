import pandas as pd
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

def DeepFeatureSynthesis():
    data = pd.read_csv("CreditTraining.csv")
    
    customers_df = data[["Id_Customer", "BirthDate", "Customer_Open_Date", 
                         "Number_Of_Dependant", "Years_At_Residence", 
                         "Net_Annual_Income", "Years_At_Business", 
                         "Prod_Decision_Date", "Nb_Of_Products"]] #only numerical attributes
    
    entity = {"customers": (customers_df, "Id_Customer")}
    feature_matrix_customers, _ = ft.dfs(entities=entity, target_entity="customers")

    
    feature_matrix_customers.to_csv("dfs.csv", encoding = 'utf-8',  index = False)
    pass


if __name__ == "__main__":
    details_correction()
    #DeepFeatureSynthesis()
    
    pass