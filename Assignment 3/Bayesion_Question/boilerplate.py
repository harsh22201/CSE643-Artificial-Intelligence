#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files, and encode categorical features."""
    # Load datasets
    train_df = pd.read_csv("train_data.csv")
    val_df = pd.read_csv("validation_data.csv")
    
    # Define label mappings
    Distance_label = {"short": 1, "medium": 2, "long": 3}
    Fare_Category_label = {"Low": 1, "Medium": 2, "High": 3}
    
    # Apply label encoding to the categorical columns
    train_df['Distance'] = train_df['Distance'].map(Distance_label)
    train_df['Fare_Category'] = train_df['Fare_Category'].map(Fare_Category_label)
    
    val_df['Distance'] = val_df['Distance'].map(Distance_label)
    val_df['Fare_Category'] = val_df['Fare_Category'].map(Fare_Category_label)
    
    return train_df, val_df

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Define the structure of the DAG as a list of edges
    dag = [
        # ('Start_Stop_ID', 'Fare_Category'),
        ('End_Stop_ID', 'Fare_Category'),
        ('Distance', 'Fare_Category'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Route_Type', 'Fare_Category'),
        ('Start_Stop_ID', 'Distance'),
        ('End_Stop_ID', 'Distance'),
        ('Zones_Crossed', 'Distance'),
        ('Route_Type', 'Distance'),
        ('Start_Stop_ID','Zones_Crossed'),
        ('End_Stop_ID', 'Zones_Crossed'),
        ('Route_Type', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('End_Stop_ID', 'Route_Type'), 
        ('Start_Stop_ID', 'End_Stop_ID'),
    ]
    # Create Bayesian Network
    # model = bn.make_DAG(dag)
    model = bn.structure_learning.fit(df)
    # Fit the Bayesian Network (define the CPDs based on the data)
    model = bn.parameter_learning.fit(model, df)
    bn.plot(model)
    # Return the fitted model
    return model

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Define the structure of the DAG as a list of edges
    dag = [
        ('Start_Stop_ID', 'Fare_Category'),
        ('End_Stop_ID', 'Fare_Category'),
        ('Distance', 'Fare_Category'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Route_Type', 'Fare_Category'),
        ('Start_Stop_ID', 'Distance'),
        ('End_Stop_ID', 'Distance'),
        ('Zones_Crossed', 'Distance'),
        ('Route_Type', 'Distance'),
        ('Start_Stop_ID','Zones_Crossed'),
        ('End_Stop_ID', 'Zones_Crossed'),
        ('Route_Type', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('End_Stop_ID', 'Route_Type'), 
        ('Start_Stop_ID', 'End_Stop_ID'),
    ]

    pruned_dag = []
    egdes_pruned = []
    # Edge pruning based on correlation less than correlation_threshold
    correlation_threshold = 0.4
    # Compute correlation matrix
    corr_matrix = df.corr()
    for edge in dag:
        node1, node2 = edge
        # Check if nodes exist in the correlation matrix
        if node1 in corr_matrix.columns and node2 in corr_matrix.columns:
            corr_value = abs(corr_matrix.loc[node1, node2])  # Get the absolute correlation value
            if corr_value >= correlation_threshold:
                pruned_dag.append(edge)  # Keep edge if correlation is >= 0.4
            else:
                egdes_pruned.append(edge)
    
    print(f"Edges pruned: {egdes_pruned}")
    
    # Create Bayesian Network with pruned DAG
    model = bn.make_DAG(pruned_dag)
    # Fit the Bayesian Network (define the CPDs based on the data)
    model = bn.parameter_learning.fit(model, df)
    bn.plot(model)
    # Return the pruned and fitted model
    return model

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Define the structure of the DAG as a list of edges
    dag = [
        ('Start_Stop_ID', 'Fare_Category'),
        ('End_Stop_ID', 'Fare_Category'),
        ('Distance', 'Fare_Category'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Route_Type', 'Fare_Category'),
        ('Start_Stop_ID', 'Distance'),
        ('End_Stop_ID', 'Distance'),
        ('Zones_Crossed', 'Distance'),
        ('Route_Type', 'Distance'),
        ('Start_Stop_ID','Zones_Crossed'),
        ('End_Stop_ID', 'Zones_Crossed'),
        ('Route_Type', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('End_Stop_ID', 'Route_Type'), 
        ('Start_Stop_ID', 'End_Stop_ID'),
    ]
    # Apply structure learning (Hill Climbing) to refine the network
    model = bn.structure_learning.fit(df, methodtype='hc',bw_list_method='edges',white_list=dag)
    # Learn the parameters (CPDs) for the refined network
    model = bn.parameter_learning.fit(model, df)
    bn.plot(model)
    # Return the refined model
    return model

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(f"{fname}", 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

