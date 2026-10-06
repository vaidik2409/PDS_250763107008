"""
Housing Prices Prediction: Ames, Iowa
======================================
Complete End-to-End Competitive Machine Learning Pipeline:
- Data Ingestion & Preprocessing
- Domain-Aware Missing Value Imputation
- Ordinal Encoding & Out-of-Fold Leakage-Free Target Encoding
- Feature Engineering (Interactions, Composites, Polynomials, Ratios)
- Skewness Correction (Box-Cox) & One-Hot Encoding
- 10-Fold Out-of-Fold Validation Framework
- Diverse Base Model Ensemble (Linear, Kernel, Tree-based GBDTs)
- Tri-Strategy Meta-Ensemble (Bayesian Ridge Stacking + Ridge Stacking + SLSQP Constrained Blend)
- Final Inverse-CV Weighted Blend & Submission Guardrails
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew
from scipy.special import boxcox1p as sp_boxcox1p

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 10
np.random.seed(SEED)


def load_data(data_dir="."):
    """Load train.csv and test.csv from local or Kaggle directory."""
    candidate_paths = [
        (os.path.join(data_dir, "train.csv"), os.path.join(data_dir, "test.csv")),
        ('/kaggle/input/competitions/home-data-for-ml-course/train.csv',
         '/kaggle/input/competitions/home-data-for-ml-course/test.csv'),
        ('/kaggle/input/home-data-for-ml-course/train.csv',
         '/kaggle/input/home-data-for-ml-course/test.csv'),
        ('train.csv', 'test.csv'),
    ]

    for tr, te in candidate_paths:
        if os.path.exists(tr) and os.path.exists(te):
            print(f"[Data] Loading datasets from: {tr}")
            train_df = pd.read_csv(tr)
            test_df = pd.read_csv(te)
            return train_df, test_df

    raise FileNotFoundError("Could not locate train.csv and test.csv.")


def preprocess_data(train_df, test_df):
    """
    Cleans data, removes documented outliers, performs domain-aware imputation,
    encodes ordinals, applies out-of-fold target encoding, creates domain features,
    and applies Box-Cox transformation.
    """
    print("[Preprocessing] Removing documented outliers (GrLivArea > 4000 & SalePrice < 200k)...")
    outliers = train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 200000)].index
    train_df = train_df.drop(outliers).reset_index(drop=True)

    y_train = np.log1p(train_df['SalePrice'])
    test_ids = test_df['Id']
    n_train = len(train_df)

    all_data = pd.concat([
        train_df.drop(['Id', 'SalePrice'], axis=1),
        test_df.drop(['Id'], axis=1)
    ], axis=0).reset_index(drop=True)

    print(f"[Preprocessing] Combined dataset shape: {all_data.shape}")

    # 1. Structural Missing Value Imputation
    none_cols = [
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
    ]
    for c in none_cols:
        if c in all_data.columns:
            all_data[c] = all_data[c].fillna('None')

    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for c in zero_cols:
        if c in all_data.columns:
            all_data[c] = all_data[c].fillna(0)

    # Neighborhood median for LotFrontage
    if 'LotFrontage' in all_data.columns:
        all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    # Functional default Typ
    if 'Functional' in all_data.columns:
        all_data['Functional'] = all_data['Functional'].fillna('Typ')

    # General remaining imputation
    for c in all_data.columns:
        if all_data[c].isnull().sum() > 0:
            if all_data[c].dtype == 'object':
                all_data[c] = all_data[c].fillna(all_data[c].mode()[0])
            else:
                all_data[c] = all_data[c].fillna(all_data[c].median())

    # 2. Ordinal Encoding
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    ordinal_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
    ]
    for c in ordinal_cols:
        if c in all_data.columns:
            all_data[c] = all_data[c].map(qual_map).fillna(0).astype(int)

    exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
    if 'BsmtExposure' in all_data.columns:
        all_data['BsmtExposure'] = all_data['BsmtExposure'].map(exposure_map).fillna(0).astype(int)

    finish_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
    for c in ['BsmtFinType1', 'BsmtFinType2']:
        if c in all_data.columns:
            all_data[c] = all_data[c].map(finish_map).fillna(0).astype(int)

    garage_fin_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
    if 'GarageFinish' in all_data.columns:
        all_data['GarageFinish'] = all_data['GarageFinish'].map(garage_fin_map).fillna(0).astype(int)

    functional_map = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}
    if 'Functional' in all_data.columns:
        all_data['Functional'] = all_data['Functional'].map(functional_map).fillna(8).astype(int)

    fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0}
    if 'Fence' in all_data.columns:
        all_data['Fence'] = all_data['Fence'].map(fence_map).fillna(0).astype(int)

    all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

    # 3. Leakage-Free OOF Target Encoding
    def target_encode_oof(train_col, y, test_col, n_folds=10, smoothing=20, seed=42):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        global_mean = y.mean()
        train_encoded = pd.Series(np.nan, index=train_col.index, dtype=float)

        for tr_idx, va_idx in kf.split(train_col):
            tr_y = y.iloc[tr_idx]
            tr_col = train_col.iloc[tr_idx]
            stats = tr_y.groupby(tr_col).agg(['mean', 'count'])
            smooth_mean = (
                (stats['count'] * stats['mean'] + smoothing * global_mean) /
                (stats['count'] + smoothing)
            )
            train_encoded.iloc[va_idx] = train_col.iloc[va_idx].map(smooth_mean).fillna(global_mean)

        full_stats = y.groupby(train_col).agg(['mean', 'count'])
        full_smooth = (
            (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) /
            (full_stats['count'] + smoothing)
        )
        test_encoded = test_col.map(full_smooth).fillna(global_mean)
        return pd.concat([train_encoded, test_encoded]).reset_index(drop=True)

    for col in ['Neighborhood', 'MSSubClass']:
        train_col = all_data[col].iloc[:n_train]
        test_col = all_data[col].iloc[n_train:]
        all_data[col + '_TE'] = target_encode_oof(train_col, y_train, test_col, smoothing=20, seed=SEED)

    # 4. Composite & Domain-Engineered Features
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['TotalFloorSF'] = all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['TotalPorchSF'] = (
        all_data['WoodDeckSF'] + all_data['OpenPorchSF'] +
        all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']
    )
    all_data['TotalBathrooms'] = (
        all_data['FullBath'] + 0.5 * all_data['HalfBath'] +
        all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
    )

    all_data['HouseAge'] = (all_data['YrSold'] - all_data['YearBuilt']).clip(lower=0)
    all_data['RemodAge'] = (all_data['YrSold'] - all_data['YearRemodAdd']).clip(lower=0)
    all_data['GarageAge'] = (all_data['YrSold'] - all_data['GarageYrBlt']).clip(lower=0)
    all_data.loc[all_data['GarageYrBlt'] == 0, 'GarageAge'] = 0
    all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
    all_data['IsNewHouse'] = (all_data['YearBuilt'] == all_data['YrSold']).astype(int)

    all_data['Qual_TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']
    all_data['Qual_GrLivArea'] = all_data['OverallQual'] * all_data['GrLivArea']
    all_data['ExterQual_SF'] = all_data['ExterQual'] * all_data['TotalSF']
    all_data['KitchenQual_SF'] = all_data['KitchenQual'] * all_data['TotalSF']
    all_data['BsmtQual_SF'] = all_data['BsmtQual'] * all_data['TotalBsmtSF']
    all_data['GarageScore'] = all_data['GarageQual'] * all_data['GarageArea']

    all_data['QualIndex'] = (
        all_data['OverallQual'] + all_data['ExterQual'] + all_data['KitchenQual'] +
        all_data['BsmtQual'] + all_data['GarageQual'] + all_data['FireplaceQu']
    )

    all_data['OverallQual_Sq'] = all_data['OverallQual'] ** 2
    all_data['OverallQual_Cu'] = all_data['OverallQual'] ** 3
    all_data['TotalSF_Sq'] = all_data['TotalSF'] ** 2
    all_data['GrLivArea_Sq'] = all_data['GrLivArea'] ** 2

    all_data['Neighborhood_Qual'] = all_data['Neighborhood_TE'] * all_data['OverallQual']
    all_data['SF_Per_Room'] = all_data['TotalSF'] / (all_data['TotRmsAbvGrd'] + 1)
    all_data['LivArea_Ratio'] = all_data['GrLivArea'] / (all_data['TotalSF'] + 1)
    all_data['BsmtFinRatio'] = (
        (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2']) / (all_data['TotalBsmtSF'] + 1)
    )
    all_data['GarageToLot'] = all_data['GarageArea'] / (all_data['LotArea'] + 1)

    all_data['Has2ndFloor'] = (all_data['2ndFlrSF'] > 0).astype(int)
    all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
    all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
    all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
    all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
    all_data['HasPorch'] = (all_data['TotalPorchSF'] > 0).astype(int)

    # 5. Near-Zero Variance Dropping
    drop_cols = []
    for c in all_data.columns:
        if all_data[c].dtype == 'object':
            top_freq = all_data[c].value_counts(normalize=True).iloc[0]
        else:
            top_freq = all_data[c].value_counts(normalize=True).iloc[0] if all_data[c].nunique() > 1 else 1.0
        if top_freq > 0.995:
            drop_cols.append(c)

    print(f"[Preprocessing] Dropping {len(drop_cols)} near-zero variance features: {drop_cols}")
    all_data = all_data.drop(columns=drop_cols, errors='ignore')

    # 6. Box-Cox Power Transformation
    num_cols = all_data.select_dtypes(include=[np.number]).columns
    skewed = all_data[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed[abs(skewed) > 0.5]

    lam = 0.15
    binary_feats = [
        'IsRemodeled', 'IsNewHouse', 'Has2ndFloor', 'HasGarage',
        'HasBsmt', 'HasFireplace', 'HasPool', 'HasPorch'
    ]
    for feat in high_skew.index:
        if feat not in binary_feats:
            all_data[feat] = sp_boxcox1p(all_data[feat], lam)

    # 7. One-Hot Encoding
    all_data = pd.get_dummies(all_data, drop_first=True)

    X_train = all_data.iloc[:n_train].copy()
    X_test = all_data.iloc[n_train:].copy()

    print(f"[Preprocessing] Final Feature Matrix: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, y_train, X_test, test_ids


def evaluate_model(model, X, y, X_test, kf, name='Model'):
    """Evaluates a single model using k-fold cross validation and computes OOF predictions."""
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        m = clone(model)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        pred = m.predict(X.iloc[va_idx])
        oof[va_idx] = pred
        scores.append(np.sqrt(mean_squared_error(y.iloc[va_idx], pred)))
        test_pred += m.predict(X_test) / kf.n_splits

    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"[{name:28s}] 10-Fold CV RMSE: {rmse:.5f}  (std: {np.std(scores):.5f})")
    return oof, test_pred, rmse


def build_base_models():
    """Initializes diverse base learners across linear, kernel, and tree models."""
    base_models = {
        'Ridge':       make_pipeline(RobustScaler(), Ridge(alpha=12.0, random_state=SEED)),
        'Lasso':       make_pipeline(RobustScaler(), Lasso(alpha=0.0003, max_iter=20000, random_state=SEED)),
        'ElasticNet':  make_pipeline(RobustScaler(), ElasticNet(alpha=0.0003, l1_ratio=0.6, max_iter=20000, random_state=SEED)),
        'BayesianRidge': make_pipeline(RobustScaler(), BayesianRidge(max_iter=500)),
        'KernelRidge': make_pipeline(RobustScaler(), KernelRidge(alpha=0.4, kernel='polynomial', degree=2, coef0=2.5)),
        'SVR':         make_pipeline(RobustScaler(), SVR(C=30, epsilon=0.006, gamma=0.0003)),
        'GBM':         GradientBoostingRegressor(
                           n_estimators=1500, learning_rate=0.01, max_depth=4,
                           max_features='sqrt', loss='huber',
                           subsample=0.75, random_state=SEED),
        'HGBM':        HistGradientBoostingRegressor(
                           max_iter=1500, learning_rate=0.01, max_depth=4,
                           l2_regularization=1.0, min_samples_leaf=10,
                           random_state=SEED),
    }

    try:
        from xgboost import XGBRegressor
        base_models['XGBoost'] = XGBRegressor(
            n_estimators=1500, learning_rate=0.01, max_depth=4,
            subsample=0.7, colsample_bytree=0.65,
            reg_alpha=0.3, reg_lambda=1.5,
            min_child_weight=3,
            random_state=SEED, n_jobs=-1
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        base_models['LightGBM'] = LGBMRegressor(
            n_estimators=1500, learning_rate=0.01, max_depth=4,
            num_leaves=15, subsample=0.7, colsample_bytree=0.65,
            reg_alpha=0.3, reg_lambda=1.5,
            min_child_samples=10,
            random_state=SEED, verbose=-1
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor
        base_models['CatBoost'] = CatBoostRegressor(
            iterations=1500, learning_rate=0.01, depth=5,
            l2_leaf_reg=5.0, subsample=0.7,
            random_state=SEED, verbose=0
        )
    except ImportError:
        pass

    return base_models


def train_meta_ensemble(oof_dict, test_dict, score_dict, X_train, y_train, X_test, kf):
    """
    Executes Level-2 Meta-Stacking and Blending:
    Strategy A: Bayesian Ridge Meta-Stacker
    Strategy B: Ridge Meta-Stacker
    Strategy C: SLSQP Constrained Blend
    Final: Inverse CV-RMSE Weighted Blend
    """
    names = list(oof_dict.keys())
    OOF = np.column_stack([oof_dict[n] for n in names])
    TEST = np.column_stack([test_dict[n] for n in names])

    print("\n--- Level-2 Meta-Ensembling ---")

    # Strategy A: Bayesian Ridge Stacker
    meta_bayes = BayesianRidge(max_iter=500)
    meta_oof = np.zeros(len(X_train))
    meta_test = np.zeros(len(X_test))

    for tr_idx, va_idx in kf.split(OOF, y_train):
        m = clone(meta_bayes)
        m.fit(OOF[tr_idx], y_train.iloc[tr_idx])
        meta_oof[va_idx] = m.predict(OOF[va_idx])
        meta_test += m.predict(TEST) / kf.n_splits

    stack_rmse = np.sqrt(mean_squared_error(y_train, meta_oof))
    print(f"[Strategy A: Bayesian Ridge Stacker] CV RMSE: {stack_rmse:.5f}")

    # Strategy B: Ridge Stacker
    meta_ridge = make_pipeline(RobustScaler(), Ridge(alpha=6.0, random_state=SEED))
    ridge_meta_oof = np.zeros(len(X_train))
    ridge_meta_test = np.zeros(len(X_test))

    for tr_idx, va_idx in kf.split(OOF, y_train):
        m = clone(meta_ridge)
        m.fit(OOF[tr_idx], y_train.iloc[tr_idx])
        ridge_meta_oof[va_idx] = m.predict(OOF[va_idx])
        ridge_meta_test += m.predict(TEST) / kf.n_splits

    ridge_stack_rmse = np.sqrt(mean_squared_error(y_train, ridge_meta_oof))
    print(f"[Strategy B: Regularized Ridge Stacker] CV RMSE: {ridge_stack_rmse:.5f}")

    # Strategy C: SLSQP Constrained Blend
    def blend_loss(w):
        w = np.array(w) / np.sum(w)
        return np.sqrt(mean_squared_error(y_train, OOF @ w))

    res = minimize(
        blend_loss,
        x0=[1.0 / len(names)] * len(names),
        method='SLSQP',
        bounds=[(0, 1)] * len(names),
        constraints={'type': 'eq', 'fun': lambda w: 1 - sum(w)}
    )
    w_opt = res.x / res.x.sum()

    slsqp_oof = OOF @ w_opt
    slsqp_test = TEST @ w_opt
    slsqp_rmse = np.sqrt(mean_squared_error(y_train, slsqp_oof))
    print(f"[Strategy C: SLSQP Constrained Blend ] CV RMSE: {slsqp_rmse:.5f}")

    # Final Combined Inverse-CV Weighted Blend
    strategies = {
        'BayesRidge': (meta_oof, meta_test, stack_rmse),
        'RidgeStack': (ridge_meta_oof, ridge_meta_test, ridge_stack_rmse),
        'SLSQP':     (slsqp_oof, slsqp_test, slsqp_rmse),
    }

    inv_sum = sum(1.0 / s[2] for s in strategies.values())
    final_oof = sum((1.0 / s[2]) / inv_sum * s[0] for s in strategies.values())
    final_test = sum((1.0 / s[2]) / inv_sum * s[1] for s in strategies.values())
    final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))

    print(f"\n================================================")
    print(f"★ FINAL META-ENSEMBLE 10-FOLD CV RMSE: {final_rmse:.5f} ★")
    print(f"================================================\n")

    return final_test, final_rmse


def create_submission(final_test, y_train, test_ids, output_path="submission.csv"):
    """Inverts log transform, clips outliers, runs sanity checks, and writes submission.csv."""
    predictions = np.expm1(final_test)
    low = np.expm1(y_train).quantile(0.002)
    high = np.expm1(y_train).quantile(0.998)
    predictions = np.clip(predictions, low, high)

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})

    assert submission['SalePrice'].isnull().sum() == 0, "Submission contains nulls!"
    assert (submission['SalePrice'] <= 0).sum() == 0, "Submission contains non-positive values!"
    assert len(submission) == len(test_ids), "Submission row count mismatch!"

    submission.to_csv(output_path, index=False)
    print(f"[Submission] Successfully wrote {len(submission)} rows to {output_path}")
    print("\nSummary of Predictions:")
    print(submission['SalePrice'].describe())


def main():
    print("=" * 60)
    print("Housing Prices Prediction Pipeline - Ames, Iowa")
    print("=" * 60)

    train_df, test_df = load_data()
    X_train, y_train, X_test, test_ids = preprocess_data(train_df, test_df)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    base_models = build_base_models()

    oof_dict, test_dict, score_dict = {}, {}, {}
    print("\n--- Training Base Learners (10-Fold CV) ---")
    for name, model in base_models.items():
        oof_p, test_p, s = evaluate_model(model, X_train, y_train, X_test, kf, name)
        oof_dict[name] = oof_p
        test_dict[name] = test_p
        score_dict[name] = s

    final_test, final_rmse = train_meta_ensemble(
        oof_dict, test_dict, score_dict, X_train, y_train, X_test, kf
    )
    create_submission(final_test, y_train, test_ids)


if __name__ == "__main__":
    main()
