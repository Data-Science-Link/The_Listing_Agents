def pre_processing(path, include_variables = 'num_and_chosen_upgradable'):
    '''
    This function takes in Aimes, Iowa housing data and spits out a scikit multi-linear regression model
    '''
    
    # import relevant libraries
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression 
    from sklearn.model_selection import train_test_split
    
    # read data in from parent directory
    from read_path_module import read_data_relative_path
    try:
        df_test = read_data_relative_path(relative_dataset_path = path, data_type='csv')
    except:
        df_test = path
    
    df_train = df_test.copy() # so the script does not have to be changed
    
    # Modify kitchenQual due to small sample size of Fair 
    df_train.KitchenQual = df_train.KitchenQual.str.replace('Fa', 'low')
    df_train.KitchenQual = df_train.KitchenQual.str.replace('TA', 'low')
    
    # Modify BsmtFinType1 due to small sample size of below average quality basements
    df_train.BsmtFinType1 = df_train.BsmtFinType1.str.replace('BLQ', 'below_average')
    df_train.BsmtFinType1 = df_train.BsmtFinType1.str.replace('Rec', 'below_average')
    df_train.BsmtFinType1 = df_train.BsmtFinType1.str.replace('LwQ', 'below_average')
    df_train.BsmtFinType1 = df_train.BsmtFinType1.str.replace('Unf', 'below_average')
    
    # impute truly "missing" data (i.e. the NA's do not have significance)
    df_train['LotFrontage'] = df_train['LotFrontage'].mask(df_train['LotFrontage'].isnull(), np.random.uniform(df_train['LotFrontage'].min(), df_train['LotFrontage'].max(), size = df_train['LotFrontage'].shape))
    df_train['GarageYrBlt'] = df_train['GarageYrBlt'].mask(df_train['GarageYrBlt'].isnull(), np.random.uniform(df_train['GarageYrBlt'].min(), df_train['GarageYrBlt'].max(), size = df_train['GarageYrBlt'].shape))
    df_train['MasVnrArea'] = df_train['MasVnrArea'].mask(df_train['MasVnrArea'].isnull(), np.random.uniform(df_train['MasVnrArea'].min(), df_train['MasVnrArea'].max(), size = df_train['MasVnrArea'].shape))
    
    # impute missing data from test data specifically  
    df_train['GarageCars'] = df_train['GarageCars'].mask(df_train['GarageCars'].isnull(), np.random.uniform(df_train['GarageCars'].min(), df_train['GarageCars'].max(), size = df_train['GarageCars'].shape))
    
    df_train['GarageArea'] = df_train['GarageArea'].mask(df_train['GarageArea'].isnull(), np.random.uniform(df_train['GarageArea'].min(), df_train['GarageArea'].max(), size = df_train['GarageArea'].shape))
    
    df_train['BsmtUnfSF'] = df_train['BsmtUnfSF'].mask(df_train['BsmtUnfSF'].isnull(), np.random.uniform(df_train['BsmtUnfSF'].min(), df_train['BsmtUnfSF'].max(), size = df_train['BsmtUnfSF'].shape))
    
    df_train['BsmtFinSF1'] = df_train['BsmtFinSF1'].mask(df_train['BsmtFinSF1'].isnull(), np.random.uniform(df_train['BsmtFinSF1'].min(), df_train['BsmtFinSF1'].max(), size = df_train['BsmtFinSF1'].shape))
    
    df_train['TotalBsmtSF'] = df_train['TotalBsmtSF'].mask(df_train['TotalBsmtSF'].isnull(), np.random.uniform(df_train['TotalBsmtSF'].min(), df_train['TotalBsmtSF'].max(), size = df_train['TotalBsmtSF'].shape))
    
    df_train['BsmtFinSF2'] = df_train['BsmtFinSF2'].mask(df_train['BsmtFinSF2'].isnull(), np.random.uniform(df_train['BsmtFinSF2'].min(), df_train['BsmtFinSF2'].max(), size = df_train['BsmtFinSF2'].shape))

    df_train['BsmtHalfBath'] = df_train['BsmtHalfBath'].mask(df_train['BsmtHalfBath'].isnull(), np.random.uniform(df_train['BsmtHalfBath'].min(), df_train['BsmtHalfBath'].max(), size = df_train['BsmtHalfBath'].shape))
    
    df_train['BsmtFullBath'] = df_train['BsmtFullBath'].mask(df_train['BsmtFullBath'].isnull(), np.random.uniform(df_train['BsmtFullBath'].min(), df_train['BsmtFullBath'].max(), size = df_train['BsmtFullBath'].shape))
    

    
    # fill rest of NA's with Nothing to add categorical meaning (i.e. a NA in poolQC means that there is no pool)
    df_train.KitchenQual.fillna('low', inplace = True)
        
    # fill rest of NA's with Nothing to add categorical meaning (i.e. a NA in poolQC means that there is no pool)
    df_train.fillna('Nothing', inplace = True)  
    
    # drop non-needed columns
    df = df_train.drop(['Id'], axis = 1)
    df = df.drop(['Unnamed: 0'], axis = 1) 
    
    # create df copy and isolate the categorical columns for dummification
    categorical = ['Alley', 'BldgType_group', 'BsmtCond_group', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual_group', 'CentralAir', 'Condition1_group', 'Electrical_group', 'ExterCond_group', 'ExterQual', 'Exterior1st_group', 'Exterior2nd_group', 'Fence', 'FireplaceQu', 'Foundation_group', 'GarageCond_group', 'GarageFinish', 'GarageQual', 'GarageType', 'HeatingQC_group', 'HouseStyle_group', 'KitchenQual', 'LandContour_group', 'LandSlope', 'LotConfig_group', 'LotShape_group', 'MS_Zoning_group', 'MasVnrType_group', 'Neighborhood', 'PavedDrive', 'RoofStyle_group', 'SaleCondition_group', 'SaleType_group']
    df_1 = df[categorical]
    df_dum = pd.get_dummies(df_1, drop_first = True, sparse=False)  
    
    # create df copy and isolate the upgradable categorical columns for dummification
    categorical_upgradable = ['RoofStyle_group', 'HeatingQC_group', 'GarageType', 'GarageCond_group',
 'PavedDrive', 'CentralAir', 'FireplaceQu', 'BsmtCond_group', 'KitchenQual', 'Fence', 'GarageQual',
 'BsmtFinType2', 'BsmtFinType1', 'GarageFinish']
    df_2 = df[categorical_upgradable]
    df_dum_upgradable = pd.get_dummies(df_2, drop_first = True, sparse=False) 
    
    # create df copy and isolate the upgradable categorical columns for dummification
    chosen_categorical_upgradable = ['KitchenQual', 'BsmtFinType1'] 
        # These were chosen but we did not want to double dip with df_num'BsmtFullBath', 'FullBath', 
        # took out , 'GarageType' due to strange behavior
    df_3 = df[chosen_categorical_upgradable]
    df_dum_chosen_upgradable = pd.get_dummies(df_3, sparse=False) #drop_first = True,
    df_dum_chosen_upgradable.drop(columns = ['KitchenQual_low', 'BsmtFinType1_below_average'], inplace=True) 
        # drop selected dummified columns for best model interpratibility
    
       
    # create df copy of numerical variables and concatenate this with the dummified df
    df_num = df_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', 
             'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
             'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 
             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'MoSold', 
             'YrSold']] 
    
    if include_variables == 'all':
        df = pd.concat([df_num, df_dum], axis = 1) 
    elif include_variables == 'num':
        df = df_num
    elif include_variables == 'num_and_upgradable':
        df = pd.concat([df_num, df_dum_upgradable], axis = 1) 
    elif include_variables == 'num_and_chosen_upgradable':
        df = pd.concat([df_num, df_dum_chosen_upgradable], axis = 1) 
    else:
        df = pd.concat([df_num, df_dum], axis = 1) 
    
    
    # create X and y      
    X_test = df
        
    return X_test

