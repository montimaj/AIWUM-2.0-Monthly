# Author: Sayantan Majumdar
# Email: sayantan.majumdar@colostate.edu


import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV as DaskGCV
from dask_ml.model_selection import RandomizedSearchCV as DaskRCV
from dask_jobqueue import SLURMCluster
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay as PDisp
from .sysops import makedirs, make_proper_dir_name


def get_model_param_dict(random_state=0, use_dask=False):
    """
    Get model object dictionaries and parameter dictionary for different models
    :param random_state: PRNG seed
    :param use_dask: Set True if using Dask in a distributed computing environment
    :return: Dictionary of the model objects and Dictionary of models containing dictionary of the corresponding
    hyperparameters as tuples
    """

    n_jobs = -1
    if use_dask:
        n_jobs = 1
    model_dict = {
        'LGBM': LGBMRegressor(
            tree_learner='feature', random_state=random_state,
            deterministic=True, force_row_wise=True,
        ),
        'DRF': LGBMRegressor(
            boosting_type='rf', tree_learner='feature',
            subsample_freq=1, random_state=random_state,
            deterministic=True, force_row_wise=True
        ),
        'RF': RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
        'ETR': ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs, bootstrap=True),
        'DT': DecisionTreeRegressor(random_state=random_state),
        'BT': BaggingRegressor(random_state=random_state, n_jobs=n_jobs),
        'ABR': AdaBoostRegressor(random_state=random_state),
        'KNN': KNeighborsRegressor(n_jobs=n_jobs),
        'SVR': LinearSVR(random_state=random_state),
        'LR': LinearRegression(n_jobs=n_jobs)
    }

    param_dict = {'LGBM': {
        'n_estimators': [500],
        'max_depth': [-1],
        'learning_rate': [0.03],
        'subsample': [1],
        'colsample_bytree': [1],
        'colsample_bynode': [1],
        'path_smooth': [0],
        'num_leaves': [127],
        'min_child_samples': [10]
    }, 'DRF': {
        'n_estimators': [500],
        'max_depth': [12, -1],
        'learning_rate': [1e-4],
        'subsample': [0.99],
        'colsample_bytree': [1.0],
        'reg_lambda': [0],
        'path_smooth': [0],
        'num_leaves': [200],
        'min_child_samples': [30],
    }, 'RF': {
        'n_estimators': [500],
        #'max_features': [10, 12, 8, 9, 3],
        'max_depth': [14],
        'max_leaf_nodes': [None],
        'max_samples': [None],
        'min_samples_leaf': [2]
    }, 'ETR': {
        'n_estimators': [300, 400, 500],
        'max_features': [5, 6, 7],
        'max_depth': [6, 10, None],
        'max_samples': [None, 0.9, 0.8, 0.7],
        'min_samples_leaf': [1, 5e-4, 1e-5]
    }, 'DT': {
        'max_features': [5, 6, 7],
        'max_depth': [6, 10, 20, None],
        'min_samples_leaf': [1, 5e-4, 1e-5]
    }, 'BT': {
        'n_estimators': [300, 400, 500],
        'max_features': [5, 6, 7],
        'max_samples': [1, 0.9, 0.8]
    }, 'ABR': {
        'n_estimators': [300, 400, 500, 600, 700],
        'learning_rate': [0.005, 0.0098, 0.01, 0.05],
        'loss': ['linear', 'square']
    }, 'KNN': {
        'n_neighbors': [5, 8, 10],
        'weights': ['uniform', 'distance'],
        'leaf_size': [30, 50, 20],
        'p': [1, 2, 3, 5],
    }, 'SVR': {
        'C': [1, 1.5, 2],
        'max_iter': [1000, 2000],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
    }, 'LR': {
    }}
    return model_dict, param_dict


def build_crop_ml_models(x_train, x_test, y_train, y_test, year_train, year_test, year_col, crop_col, model_dir,
                         model_name='DRF', random_state=43, load_model=False, fold_count=5, repeats=3, x_scaler=None,
                         y_scaler=None, randomized_search=False, stratified_kfold=False, use_dask=False):
    """
    Build individual ML models for each crop type
    :param x_train: X_train numpy array or pandas dataframe
    :param x_test: X_test numpy array or pandas dataframe
    :param y_train: y_train numpy array
    :param y_test: y_test numpy array
    :param year_train: Year train data frame to append to train data
    :param year_test: Year test data frame to append to test data
    :param year_col: Name of the year column
    :param crop_col: Name of the crop column
    :param model_dir: Model directory to store/load model
    :param model_name: ML model name as per the model_dict keys
    :param random_state: PRNG seed
    :param load_model: Set model name to load existing model
    :param fold_count: Number of folds for KFold
    :param repeats: Number of repeats for KFold
    :param x_scaler: X scaler object
    :param y_scaler: y scaler object
    :param randomized_search: Set True to use the more computationally efficient RandomizedSearchCV
    :param stratified_kfold: Set True to use RepeatedStratifiedKFold based on the crop type
    :param use_dask: Flag for using dask
    :return: List of trained models
    """

    crop_train_arr = x_train[crop_col].to_numpy().ravel()
    crop_test_arr = x_test[crop_col].to_numpy().ravel()
    models = []
    merged_pred_df = pd.DataFrame()
    for crop in x_train[crop_col].unique():
        crop_train_check = x_train[crop_col] == crop
        crop_test_check = x_test[crop_col] == crop
        x_train_data = x_train[crop_train_check]
        x_test_data = x_test[crop_test_check]
        x_train_data = x_train_data.drop(columns=[crop_col])
        x_test_data = x_test_data.drop(columns=[crop_col])
        y_train_data = y_train[crop_train_arr == crop]
        y_test_data = y_test[crop_test_arr == crop]
        year_train_df = year_train[crop_train_check]
        year_test_df = year_test[crop_test_check]
        crop_model_name = model_name + '_' + str(crop)
        model = build_ml_model(x_train_data, y_train_data, model_dir, crop_model_name, random_state, load_model,
                               fold_count, repeats, y_scaler, randomized_search, stratified_kfold, use_dask)
        models.append(model)
        pred_df = get_prediction_results(model, x_train_data, x_test_data, y_train_data, y_test_data, x_scaler,
                                         y_scaler, year_train_df, year_test_df, model_dir, crop_model_name, year_col,
                                         crop_col)
        merged_pred_df = pd.concat([merged_pred_df, pred_df])
    calc_train_test_metrics(merged_pred_df, crop_col, year_col)
    merged_pred_df.to_csv(model_dir + 'Merged_Crop_Predictions_{}.csv'.format(model_name), index=False)
    return models


def build_ml_model(x_train, y_train, model_dir, model_name='DRF', random_state=43, load_model=False,
                   fold_count=5, repeats=3, y_scaler=None, randomized_search=False, stratified_kfold=False,
                   use_dask=False, **kwargs):
    """
    Build an ML model
    :param x_train: X_train numpy array or pandas dataframe
    :param y_train: y_train numpy array
    :param model_dir: Model directory to store/load model
    :param model_name: ML model name as per the model_dict keys
    :param random_state: PRNG seed
    :param load_model: Set model name to load existing model
    :param fold_count: Number of folds for KFold. Performs LOOCV if <=0
    :param repeats: Number of repeats for KFold
    :param y_scaler: y scaler object
    :param randomized_search: Set True to use the more computationally efficient RandomizedSearchCV
    :param stratified_kfold: Set True to use RepeatedStratifiedKFold based on the crop type
    :param use_dask: Flag for using dask
    :param kwargs: Pass the 'crop_train' or 'year_train' Pandas dataframe if stratified_kfold is True
    :return: Trained model object
    """

    model_file = model_dir + model_name
    if not load_model:
        dask_client = None
        cv_lib = 'sklearn'
        if use_dask:
            cluster = SLURMCluster(
                cores=32,
                processes=1,
                memory="10G",
                walltime="00:30:00",
                interface='ib0',
                env_extra=['#SBATCH --out=Foundry-Dask-%j.out']
            )
            cluster.adapt(
                minimum=10, maximum=50,
                minimum_jobs=10, maximum_jobs=50,
                minimum_memory='8G', maximum_memory='10G'
            )
            dask_client = Client(cluster)
            print('Waiting for dask workers...')
            dask_client.wait_for_workers(1)
            cv_lib = 'dask_ml'
        model_dict, param_dict = get_model_param_dict(random_state, use_dask)
        cv = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
        if stratified_kfold:
            stratify_labels = kwargs['stratify_labels'].to_numpy().ravel()
            cv = RepeatedStratifiedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
            cv = cv.split(x_train, stratify_labels)
        makedirs([make_proper_dir_name(model_dir)])
        print('\nSearching best params for {}...'.format(model_name))
        if '_' in model_name:
            model_name = model_name[: model_name.find('_')]
        model = model_dict[model_name]
        scoring_metrics = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']
        cv_func_dict = {
            'dask_ml': {1: DaskRCV, 0: DaskGCV},
            'sklearn': {1: RandomizedSearchCV, 0: GridSearchCV}
        }
        cv_func = cv_func_dict[cv_lib][int(randomized_search)]
        if randomized_search:
            model_grid = cv_func(
                estimator=model, param_distributions=param_dict[model_name],
                scoring=scoring_metrics, n_jobs=-1, cv=cv, refit=scoring_metrics[1],
                return_train_score=True, random_state=random_state
            )
        else:
            model_grid = cv_func(
                estimator=model, param_grid=param_dict[model_name],
                scoring=scoring_metrics, n_jobs=-1, cv=cv, refit=scoring_metrics[1],
                return_train_score=True
            )
        model_grid.fit(x_train, y_train)
        pickle.dump(model, open(model_file, mode='wb+'))
        get_grid_search_stats(model_grid, y_scaler)
        model = model_grid.best_estimator_
        print('Best params: ', model_grid.best_params_)
        if dask_client:
            dask_client.close()
    else:
        model = pickle.load(open(model_file, mode='rb'))
    if model_name in ['RF', 'ETR', 'LGBM', 'DRF']:
        imp_dict = {'Features': list(x_train.columns)}
        f_imp = np.array(model.feature_importances_).astype(np.float)
        if model_name in ['LGBM', 'DRF']:
            f_imp /= np.sum(f_imp)
        imp_dict['F_IMP'] = np.round(f_imp, 5)
        imp_df = pd.DataFrame(data=imp_dict).sort_values(by='F_IMP', ascending=False)
        print(imp_df)
        imp_df.to_csv(model_dir + 'F_IMP.csv', index=False)
    return model


def calc_train_test_metrics(pred_df, crop_col=None, year_col=None):
    """
    Calculate train and test metrics from the prediction data frames
    :param pred_df: Prediction data frame
    :param crop_col: Name of the crop column
    :param year_col: Name of the year column
    :return: None
    """

    train_data = pred_df[pred_df.DATA == 'TRAIN']
    test_data = pred_df[pred_df.DATA == 'TEST']
    train_actual = train_data.Actual_GW.to_numpy().ravel()
    train_pred = train_data.Pred_GW.to_numpy().ravel()
    test_actual = test_data.Actual_GW.to_numpy().ravel()
    test_pred = test_data.Pred_GW.to_numpy().ravel()
    print('***Overall stats***\n')
    print('Train + Validation results...')
    r2, mae, rmse = get_prediction_stats(train_actual, train_pred)
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    print('\nTest results...')
    r2, mae, rmse = get_prediction_stats(test_actual, test_pred)
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    if crop_col and year_col:
        cols = [col for col in pred_df.columns if col.startswith(crop_col)] + [year_col]
        for col in cols:
            print('\n***{} specific stats***\n'.format(col))
            if col != year_col:
                col_val_list = [1]
            else:
                col_val_list = np.unique(np.append(train_data[col].unique(), test_data[col].unique()))
            for val in sorted(col_val_list):
                print('\n{} type: {}'.format(col, val))
                train_actual = train_data[train_data[col] == val]['Actual_GW'].to_numpy().ravel()
                train_pred = train_data[train_data[col] == val]['Pred_GW'].to_numpy().ravel()
                test_actual = test_data[test_data[col] == val]['Actual_GW'].to_numpy().ravel()
                test_pred = test_data[test_data[col] == val]['Pred_GW'].to_numpy().ravel()
                print('Train + Validation results...')
                r2, mae, rmse = get_prediction_stats(train_actual, train_pred)
                print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
                print('\nTest results...')
                r2, mae, rmse = get_prediction_stats(test_actual, test_pred)
                print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)


def get_grid_search_stats(gs_model, y_scaler=None):
    """
    Get GridSearchCV stats
    :param gs_model: Fitted GridSearchCV/RandomizedSearchCV object
    :param y_scaler:y scaler object
    :return:
    """

    scores = gs_model.cv_results_
    print('Train Results...')
    r2 = scores['mean_train_r2'].mean()
    rmse = -scores['mean_train_neg_root_mean_squared_error'].mean()
    mae = -scores['mean_train_neg_mean_absolute_error'].mean()
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    print('Validation Results...')
    r2 = scores['mean_test_r2'].mean()
    rmse = -scores['mean_test_neg_root_mean_squared_error'].mean()
    mae = -scores['mean_test_neg_mean_absolute_error'].mean()
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)


def get_prediction_stats(actual_values, pred_values, precision=3):
    """
    Get prediction statistics R^2, MAE, RMSE
    :param actual_values: Numpy array of actual values
    :param pred_values: Numpy array of predicted values
    :param precision: Floating point precision to use
    :return: R^2, MAE, RMSE as tuples
    """

    r2, mae, rmse = (np.nan,) * 3
    if actual_values.size and pred_values.size:
        r2 = np.round(r2_score(actual_values, pred_values), precision)
        mae = np.round(mean_absolute_error(actual_values, pred_values), precision)
        rmse = np.round(mean_squared_error(actual_values, pred_values, squared=False), precision)
    return r2, mae, rmse


def get_prediction_results(model, x_train, x_test, y_train, y_test, x_scaler, y_scaler, year_train, year_test,
                           model_dir, model_name='DRF', year_col='ReportYear',  crop_col='Crop_CDL',
                           crop_train=None, crop_test=None):
    """
    Get model prediction results
    :param model: Trained model object
    :param x_train: X_train numpy array or pandas dataframe
    :param x_test: X_test numpy array or pandas dataframe
    :param y_train: y_train numpy array
    :param y_test: y_test numpy array
    :param x_scaler: X scaler object
    :param y_scaler: y scaler object
    :param year_train: Year train data frame to append to train data
    :param year_test: Year test data frame to append to test data
    :param model_dir: Model directory to store/load results
    :param model_name: Model name
    :param year_col: Name of the year column
    :param crop_col: Name of the crop column
    :param crop_train: Crop train data frame to append to train data
    :param crop_test: Crop test data frame to append to test data
    :return: Prediction data frame
    """

    y_pred_train = np.abs(model.predict(x_train))
    y_pred_test = np.abs(model.predict(x_test))
    if x_scaler and y_scaler:
        x_train = pd.DataFrame(x_scaler.inverse_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(x_scaler.inverse_transform(x_test), columns=x_test.columns)
        y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_train = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
        y_pred_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    train_df = x_train.copy()
    train_df[year_col] = year_train[year_col].to_numpy().ravel()
    test_df = x_test.copy()
    test_df[year_col] = year_test[year_col].to_numpy().ravel()
    crop_models = '_' in model_name
    if (not crop_models) and (crop_col not in train_df.columns):
        train_df[crop_col] = crop_train[crop_col].to_numpy().ravel()
        test_df[crop_col] = crop_test[crop_col].to_numpy().ravel()
    train_df['DATA'] = ['TRAIN'] * train_df.shape[0]
    train_df['Pred_GW'] = y_pred_train
    train_df['Actual_GW'] = y_train
    test_df['DATA'] = ['TEST'] * test_df.shape[0]
    test_df['Pred_GW'] = y_pred_test
    test_df['Actual_GW'] = y_test
    pred_df = pd.concat([train_df, test_df])
    pred_df['Error_GW'] = pred_df['Actual_GW'] - pred_df['Pred_GW']
    if crop_models:
        crop = model_name[model_name.find('_') + 1:]
        pred_df[crop_col] = [crop] * pred_df['Error_GW'].size
    pred_df.to_csv(model_dir + 'Predictions_{}.csv'.format(model_name), index=False)
    return pred_df


def create_pdplots(x_train, model, feature_names, outdir, scaling=True, random_state=0):
    """
    Create partial dependence plots
    :param x_train: Training set
    :param model: Random Forest model
    :param feature_names: Feature names for which PDP will be generated. Set 'All' to use all the features used for
    model training.
    :param outdir: Output directory for storing partial dependence plot
    :param scaling: Set False if scaling is not used for the model
    :param random_state: PRNG seed
    :return: None
    """

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print('Plotting PDP...')
    matplotlib.rcParams.update({'font.size': 16})
    feature_dict = {
        'ppt': 'PRISM Precipitation',
        'SSEBop': 'SSEBop ET',
        'Relative_SSEBop': 'Relative SSEBop ET',
        'SM_IDAHO': 'Soil Moisture Change',
        'SWB_IRR': 'Irrigation Demand',
        'HSG_INF': 'Infiltration Rate',
        'tmax': r'PRISM Max Temperature',
        'tmin': r'PRISM Min Temperature',
        'tmean': r'PRISM Mean Temperature',
        'RO':  'Surface Runoff',
        'Latitude': r'Latitude',
        'Longitude': r'Longitude',
        'GW': 'Groundwater Use',
        'Crop(s)_Corn': 'Corn',
        'Crop(s)_Soybeans': 'Soybeans',
        'Crop(s)_Cotton': 'Cotton',
        'Crop(s)_Fish Culture': 'Aquaculture',
        'Crop(s)_Rice': 'Rice'
    }
    pdp_feature_dict = {}
    if 'All' in feature_names:
        feature_names = x_train.columns
    for feature in feature_dict.keys():
        if not scaling:
            if feature in ['tmax', 'tmin', 'tmean']:
                unit = r'$^\circ$C'
            elif feature in ['Latitude', 'Longitude']:
                unit = r'$^\circ$'
            elif feature == 'HSG_INF':
                unit = 'mm/hr'
            elif feature.startswith('Relative') or feature.startswith('Crop(s)'):
                unit = 'Unitless'
            else:
                unit = 'mm'
        else:
            unit = 'Normalized'
        feature_dict[feature] += ' ({})'.format(unit)
        if feature in feature_names:
            pdp_feature_dict[feature] = feature_dict[feature]
    x_train = x_train.rename(columns=pdp_feature_dict)
    feature_names = sorted(list(pdp_feature_dict.values()))
    pdisp = PDisp.from_estimator(
        model, X=x_train, features=feature_names,
        n_jobs=-1, random_state=random_state,
        subsample=0.8
    )
    for row_idx in range(pdisp.axes_.shape[0]):
        pdisp.axes_[row_idx][0].set_ylabel(feature_dict['GW'])
    pdisp.figure_.set_size_inches(30, 15)
    plt.savefig(outdir + 'PDP.png', dpi=600)
