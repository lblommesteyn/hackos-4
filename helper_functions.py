import os
import json
import time
from argparse import Namespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import StratifiedKFold 
from models import Conv, LSTM, GRU
from helper_classes import Dataset

with open("./SETTINGS.json") as file:
    settings = json.load(file)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything():
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('-----Seed Set!-----')
    
    
#### Data preprocessing utilities
def one_hot_encode(data_train, data_test, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    encoder = OneHotEncoder()
    combined_data = pd.concat([data_train, 
                          data_test])
    # Fit on the combined data
    encoder.fit(combined_data)
    train_features = encoder.transform(data_train)
    test_features = encoder.transform(data_test)
    np.save(f"{out_dir}/one_hot_train.npy", train_features.toarray().astype(float))
    np.save(f"{out_dir}/one_hot_test.npy", test_features.toarray().astype(float))        
        
def build_ChemBERTa_features(smiles_list):
    chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    chemberta.eval()
    embeddings = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)
    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            encoded_input = tokenizer(smiles, return_tensors="pt", padding=False, truncation=True)
            model_output = chemberta(**encoded_input)
            embedding = model_output[0][::,0,::]
            embeddings[i] = embedding
            embedding = torch.mean(model_output[0], 1)
            embeddings_mean[i] = embedding
    return embeddings.numpy(), embeddings_mean.numpy()


def save_ChemBERTa_features(smiles_list, out_dir, on_train_data=False):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    emb, emb_mean = build_ChemBERTa_features(smiles_list)
    if on_train_data:
        np.save(f"{out_dir}/chemberta_train.npy", emb)
        np.save(f"{out_dir}/chemberta_train_mean.npy", emb_mean)
    else:
        np.save(f"{out_dir}/chemberta_test.npy", emb)
        np.save(f"{out_dir}/chemberta_test_mean.npy", emb_mean)                
                
def combine_features(data_aug_dfs, chem_feats, main_df, one_hot_dfs=None, quantiles_df=None):
    """
    This function concatenates the provided vectors, matrices and data frames (i.e., one hot, std, mean, etc) into a single long vector. This is done for each input pair (cell_type, sm_name)
    """
    new_vecs = []
    chem_feat_dim = 600
    if len(data_aug_dfs) > 0:
        add_len = sum(aug_df.shape[1]-1 for aug_df in data_aug_dfs)+chem_feat_dim*len(chem_feats)+one_hot_dfs.shape[1] if\
        one_hot_dfs is not None else sum(aug_df.shape[1]-1 for aug_df in data_aug_dfs)+chem_feat_dim*len(chem_feats)
    else:
        add_len = chem_feat_dim*len(chem_feats)+one_hot_dfs.shape[1] if\
        one_hot_dfs is not None else chem_feat_dim*len(chem_feats)
    if quantiles_df is not None:
        add_len += (quantiles_df.shape[1]-1)//3
    for i in range(len(main_df)):
        if one_hot_dfs is not None:
            vec_ = (one_hot_dfs.iloc[i,:].values).copy()
        else:
            vec_ = np.array([])
        for df in data_aug_dfs:
            if 'cell_type' in df.columns:
                values = df[df['cell_type']==main_df.iloc[i]['cell_type']].values.squeeze()[1:].astype(float)
                vec_ = np.concatenate([vec_, values])
            else:
                assert 'sm_name' in df.columns
                values = df[df['sm_name']==main_df.iloc[i]['sm_name']].values.squeeze()[1:].astype(float)
                vec_ = np.concatenate([vec_, values])
        for chem_feat in chem_feats:
            vec_ = np.concatenate([vec_, chem_feat[i]])
        final_vec = np.concatenate([vec_,np.zeros(add_len-vec_.shape[0],)])
        new_vecs.append(final_vec)
    return np.stack(new_vecs, axis=0).astype(float).reshape(len(main_df), 1, add_len)

def augment_data(x_, y_):
    copy_x = x_.copy()
    new_x = []
    new_y = y_.copy()
    dim = x_.shape[2]
    k = int(0.3*dim)
    for i in range(x_.shape[0]):
        idx = random.sample(range(dim), k=k)
        copy_x[i,:,idx] = 0
        new_x.append(copy_x[i])
    return np.stack(new_x, axis=0), new_y

#### Metrics
def mrrmse_np(y_pred, y_true):
    return np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()


#### Training utilities
def train_step(dataloader, model, opt, clip_norm):
    model.train()
    train_losses = []
    train_mrrmse = []
    for x, target in dataloader:
        model.to(device)
        x = x.to(device)
        target = target.to(device)
        loss = model(x, target)
        train_losses.append(loss.item())
        pred = model(x).detach().cpu().numpy()
        train_mrrmse.append(mrrmse_np(pred, target.cpu().numpy()))
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
    return np.mean(train_losses), np.mean(train_mrrmse)

def validation_step(dataloader, model):
    model.eval()
    val_losses = []
    val_mrrmse = []
    for x, target in dataloader:
        model.to(device)
        x = x.to(device)
        target = target.to(device)
        loss = model(x,target)
        pred = model(x).detach().cpu().numpy()
        val_mrrmse.append(mrrmse_np(pred, target.cpu().numpy()))
        val_losses.append(loss.item())
    return np.mean(val_losses), np.mean(val_mrrmse)


def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def train_function(model, x_train, y_train, x_val, y_val, info_data, config, clip_norm=1.0):

    model.apply(weight_init)
    if model.name in ['GRU']:
        print('lr', config["LEARNING_RATES"][2])
        opt = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATES"][2], weight_decay=5e-3) # Higher weight decay to prevent overfitting
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATES"][0], weight_decay=5e-3)

    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.9999, patience=10, verbose=True)
    
    model.to(device)
    results = {'train_loss': [], 'val_loss': [], 'train_mrrmse': [], 'val_mrrmse': [],\
               'train_cell_type': info_data['train_cell_type'], 'val_cell_type': info_data['val_cell_type'], 'train_sm_name': info_data['train_sm_name'], 'val_sm_name': info_data['val_sm_name'], 'runtime': None}
    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    x_train_aug = np.concatenate([x_train, x_train_aug], axis=0)
    y_train_aug = np.concatenate([y_train, y_train_aug], axis=0)
    data_x_train = torch.FloatTensor(x_train_aug)
    data_y_train = torch.FloatTensor(y_train_aug)
    data_x_val = torch.FloatTensor(x_val)
    data_y_val = torch.FloatTensor(y_val)
    train_dataloader = DataLoader(Dataset(data_x_train, data_y_train), num_workers=4, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(Dataset(data_x_val, data_y_val), num_workers=4, batch_size=32, shuffle=False)
    best_loss = np.inf
    best_weights = None
    t0 = time.time()

    # Early stopping parameters
    best_loss = np.inf
    best_weights = None
    patience = 50  # Number of epochs to wait for improvement
    patience_counter = 0

    for e in range(config["EPOCHS"]):
        loss, mrrmse = train_step(train_dataloader, model, opt, clip_norm)
        val_loss, val_mrrmse = validation_step(val_dataloader, model)

        scheduler.step(val_loss)

        results['train_loss'].append(float(loss))
        results['val_loss'].append(float(val_loss))
        results['train_mrrmse'].append(float(mrrmse))
        results['val_mrrmse'].append(float(val_mrrmse))


        if val_mrrmse < best_loss:
            best_loss = val_mrrmse
            best_weights = model.state_dict()
            patience_counter = 0
            print('BEST ----> ')
        else:   
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break
        print(f"{model.name} Epoch {e}, train_loss {round(loss,3)}, val_loss {round(val_loss, 3)}, val_mrrmse {val_mrrmse}")

    t1 = time.time()
    results['runtime'] = float(t1-t0)
    model.load_state_dict(best_weights)
    return model, results


def cross_validate_models(X, y, cell_types_sm_names, config=None, scheme='initial', clip_norm=1.0):
    trained_models = []
    kf_cv = StratifiedKFold(n_splits=config["KF_N_SPLITS"], shuffle=True, random_state=42)
    cell_types = cell_types_sm_names['cell_type'].values

    for i,(train_idx,val_idx) in enumerate(kf_cv.split(X, cell_types)):
        print(f"\nSplit {i+1}/{config['KF_N_SPLITS']}...")
        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = y.values[train_idx], y.values[val_idx]
        
        # Print distribution of cell types in this fold
        train_cell_types = cell_types_sm_names.iloc[train_idx]['cell_type']
        val_cell_types = cell_types_sm_names.iloc[val_idx]['cell_type']
        print("Training fold cell type distribution:")
        print(pd.Series(train_cell_types).value_counts(normalize=True))
        print("Validation fold cell type distribution:")
        print(pd.Series(val_cell_types).value_counts(normalize=True))

        info_data = {'train_cell_type': cell_types_sm_names.iloc[train_idx]['cell_type'].tolist(),
                    'val_cell_type': cell_types_sm_names.iloc[val_idx]['cell_type'].tolist(),
                    'train_sm_name': cell_types_sm_names.iloc[train_idx]['sm_name'].tolist(),
                    'val_sm_name': cell_types_sm_names.iloc[val_idx]['sm_name'].tolist()}
        for Model in [LSTM, Conv, GRU]:
            model = Model(scheme)
            model, results = train_function(model, x_train, y_train, x_val, y_val, info_data, config=config, clip_norm=clip_norm)
            model.to('cpu')
            trained_models.append(model)
            torch.save(model.state_dict(), f'{settings["MODEL_DIR"]}pytorch_{model.name}_{scheme}_fold{i}.pt')
            with open(f'{settings["LOGS_DIR"]}{model.name}_{scheme}_fold{i}.json', 'w') as file:
                json.dump(results, file)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return trained_models

def train_validate(X_vec, X_vec_light, X_vec_heavy, y, cell_types_sm_names, config):

    trained_models = {'initial': [], 'light': [], 'heavy': []}
    if not os.path.exists(settings["MODEL_DIR"]):
        os.mkdir(settings["MODEL_DIR"])
    if not os.path.exists(settings["LOGS_DIR"]):
        os.mkdir(settings["LOGS_DIR"])
    for scheme, clip_norm, input_features in zip(['initial', 'light', 'heavy'], config["CLIP_VALUES"], [X_vec, X_vec_light, X_vec_heavy]):
        seed_everything()
        models = cross_validate_models(input_features, y, cell_types_sm_names, config=config, scheme=scheme, clip_norm=clip_norm)
        trained_models[scheme].extend(models)
    return trained_models

#### Inference utilities
def inference_pytorch(model, dataloader):
    model.eval()
    preds = []
    for x in dataloader:
        model.to(device)
        x = x.to(device)
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(preds, axis=0)

def average_prediction(X_test, trained_models):
    all_preds = []
    test_dataloader = DataLoader(Dataset(torch.FloatTensor(X_test)), num_workers=4, batch_size=64, shuffle=False)
    for i,model in enumerate(trained_models):
        current_pred = inference_pytorch(model, test_dataloader)
        all_preds.append(current_pred)
    return np.stack(all_preds, axis=1).mean(axis=1)


def weighted_average_prediction(X_test, trained_models, model_wise=[0.25, 0.35, 0.40], fold_wise=None):
    all_preds = []
    test_dataloader = DataLoader(Dataset(torch.FloatTensor(X_test)), num_workers=4, batch_size=64, shuffle=False)
    for i,model in enumerate(trained_models):
        current_pred = inference_pytorch(model, test_dataloader)
        current_pred = model_wise[i%3]*current_pred
        if fold_wise:
            current_pred = fold_wise[i//3]*current_pred
        all_preds.append(current_pred)
    return np.stack(all_preds, axis=1).sum(axis=1)

def load_trained_models(path=settings["MODEL_DIR"], kf_n_splits=5):
    trained_models = {'initial': [], 'light': [], 'heavy': []}
    for scheme in ['initial', 'light', 'heavy']:
        for fold in range(kf_n_splits):
            for Model in [LSTM, Conv, GRU]:
                model = Model(scheme)
                for weights_path in os.listdir(path):
                    if model.name in weights_path and scheme in weights_path and f'fold{fold}' in weights_path:
                        model.load_state_dict(torch.load(f'{path}{weights_path}', map_location='cpu'))
                        trained_models[scheme].append(model)
    return trained_models

def create_stratified_sample(de_train, cell_type_boost_factors):
    
    boosted_indices = []
    original_indices = np.arange(len(de_train))
    
    # Get indices for each cell type to boost
    for cell_type, boost_factor in cell_type_boost_factors.items():
        if boost_factor > 1:  # Only boost if factor > 1
            cell_indices = de_train[de_train['cell_type'] == cell_type].index.tolist()
            # Add these indices multiple times based on boost factor
            for _ in range(boost_factor - 1):  # -1 because we already have them once
                boosted_indices.extend(cell_indices)
    
    # Combine original and boosted indices
    all_indices = np.concatenate([original_indices, np.array(boosted_indices)])
    
    # Create the new boosted dataframe
    boosted_df = de_train.iloc[all_indices].reset_index(drop=True)

    print("Cell type distribution before boosting:")
    print(de_train['cell_type'].value_counts()/len(de_train))
    print("\nCell type distribution after boosting:")
    print(boosted_df['cell_type'].value_counts()/len(boosted_df))
    
    return boosted_df, all_indices

def apply_indices_to_features(feature_array, indices):
    return feature_array[indices]