from argparse import Namespace
import pandas as pd
import numpy as np
import json
from helper_functions import seed_everything, combine_features, train_validate, create_stratified_sample, apply_indices_to_features
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    # Read settings and config files
    with open("./SETTINGS.json") as file:
        settings = json.load(file)
    with open("./config/train_config.json") as file:
        train_config = json.load(file)
        
    print("\nRead data and build features...")
    de_train = pd.read_parquet(settings["TRAIN_RAW_DATA_PATH"])
    xlist  = ['cell_type','sm_name']
    ylist = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
    one_hot_train = pd.DataFrame(np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}one_hot_train.npy'))
    # y = de_train.drop(columns=ylist)
    mean_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_cell_type.csv')
    std_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_cell_type.csv')
    mean_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_sm_name.csv')
    std_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_sm_name.csv')
    quantiles_df = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}quantiles_cell_type.csv')
    train_chem_feat = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_train.npy')
    train_chem_feat_mean = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_train_mean.npy')
    X_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                    [train_chem_feat,train_chem_feat_mean], de_train, one_hot_train, quantiles_df)
    
    cell_type_boost_factors={
        'B cells': 6,
        'Myeloid cells': 3, 
        'NK cells': 2,
        'T cells CD4+': 2,
        'T regulatory cells': 1,
        'T cells CD8+': 1
    }   
    
    boosted_de_train, boost_indices = create_stratified_sample(
        de_train, 
        cell_type_boost_factors
    )

    # Apply the same indices to feature vectors
    X_vec_boosted = apply_indices_to_features(X_vec, boost_indices)
    X_vec_light_boosted = apply_indices_to_features(X_vec_light, boost_indices)
    X_vec_heavy_boosted = apply_indices_to_features(X_vec_heavy, boost_indices)

    # Use boosted data for training
    boosted_y = boosted_de_train.drop(columns=ylist)
    boosted_cell_types_sm_names = boosted_de_train[['cell_type', 'sm_name']]

    ## Start training
    # cell_types_sm_names = de_train[['cell_type', 'sm_name']]
    print("\nTraining starting...")
    train_validate(X_vec_boosted, X_vec_light_boosted, X_vec_heavy_boosted, 
                  boosted_y, boosted_cell_types_sm_names, train_config) # Boosted cell type representation
    print("\nDone.")
    
    
    
