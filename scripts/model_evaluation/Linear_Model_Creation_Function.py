def OLS_Model_Creation(path, include_variables = 'num_and_chosen_upgradable'):
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
        df_train = read_data_relative_path(relative_dataset_path = path, data_type='csv')
    except:
        df_train = path
    
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
    chosen_categorical_upgradable = ['KitchenQual', 'BsmtFinType1'] # These were chosen but we did not want to double dip with df_num'BsmtFullBath', 'FullBath', 
    # took out , 'GarageType' due to strange behavior
    df_3 = df[chosen_categorical_upgradable]
    df_dum_chosen_upgradable = pd.get_dummies(df_3, sparse=False) #drop_first = True,
    df_dum_chosen_upgradable.drop(columns = ['KitchenQual_low', 'BsmtFinType1_below_average'], inplace=True)  
    
    
#     # create df copy of numerical variables and concatenate this with the dummified df
#     df_num = df[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']] 
    
    # create df copy of numerical variables and concatenate this with the dummified df
    df_num = df_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', 
             'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
             'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 
             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'MoSold', 
             'YrSold', 'SalePrice']] 
    
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
    
    # Filter out outliers
    mult_upper = 4
    mult_lower = 1.5
    med = df['SalePrice'].median()
    mean = df['SalePrice'].mean()
    std = df['SalePrice'].std()
    df = df.loc[(df['SalePrice'] > med - (mult_lower * std) ) & (df['SalePrice'] < med + (mult_upper * std))]
    
    # create X and y      
    X_train = df.drop(['SalePrice'], axis = 1)
    y_train = df['SalePrice']
    y_log = np.log(df['SalePrice'])
    
   
    # linear regression   
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    
    return lin_reg, X_train, y_train