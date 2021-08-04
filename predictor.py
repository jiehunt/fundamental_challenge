# -*- coding: Shift-JIS -*-

import io
import os
import pickle
import subprocess, os

import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm
# import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
# import json

from scipy.stats import spearmanr


class ScoringService(object):
    # 訓練期間終了日
    TRAIN_END = "2018-12-31"
    # 評価期間開始日
    VAL_START = "2019-02-01"
    # 評価期間終了日
    VAL_END = "2019-12-01"
    # テスト期間開始日
    TEST_START = "2020-01-01"
    # 目的変数
    TARGET_LABELS = ['label_high_20', 'label_low_20']

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    models_fname = None

    # for Stacker model
    class_model = {'label_high_20':{},
                   'label_low_20':{}
                   }


    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_rsi (cls, data, time_window):
        diff = data.diff(1).dropna()        # diff in one field(one day)

        #this preservers dimensions off diff values
        up_chg = 0 * diff
        down_chg = 0 * diff

        # up change is equal to the positive difference, otherwise equal to zero
        up_chg[diff > 0] = diff[ diff>0 ]

        # down change is equal to negative deifference, otherwise equal to zero
        down_chg[diff < 0] = diff[ diff < 0 ]

        # check pandas documentation for ewm
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
        # values are related to exponential decay
        # we set com=time_window-1 so we get decay alpha=1/time_window
        up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
        down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()

        rs = abs(up_chg_avg/down_chg_avg)
        rsi = 100 - 100/(1+rs)
        return rsi

    @classmethod
    def get_macd(cls, close, fast=12, slow=26, signal=9):
        emaslow = close.ewm( span=slow, min_periods=1).mean()
        emafast = close.ewm( span=fast, min_periods=1).mean()

        macd = emafast - emaslow
        macdsignal = macd.ewm( span=9, min_periods=1).mean()
        macdhist = (macd - macdsignal)

        return macd, macdsignal, macdhist

    @classmethod
    def get_features_for_predict(cls, dfs, codes, start_dt="2016-01-01"):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # get all Stock List Featuress
        stock_list_df = dfs['stock_list'].copy()
        stock_list_feats = cls.clean_feature_stock_list(stock_list_df, codes)

        # get All Finance Featuress
        stock_fin = dfs["stock_fin"].copy()
        stock_fin_feats = cls.clean_fin_feature(stock_fin, codes)

        # Get All Stock_list + Finance Features
        stock_fin_index = stock_fin_feats.index
        stock_fin_feats_new = pd.merge(stock_fin_feats ,stock_list_feats, on="Local Code")
        stock_fin_feats_new.index = stock_fin_index

        stock_fin_feats_new = stock_fin_feats_new.select_dtypes(include=["float", 'int', 'int64'])
        stock_fin_feats_new = stock_fin_feats_new.fillna(0)

        # Get All Stock_list + Finance + Price Features
        feats = pd.DataFrame()
        price_df = dfs["stock_price"].copy()
        for code in codes:
            all_feat = pd.DataFrame()
            stock_price_feat = cls.clean_feature_price(price_df, code, start_dt)
            stock_fin_feat = stock_fin_feats_new[stock_fin_feats_new['Local Code'] == code]

            stock_price_feat = stock_price_feat.loc[stock_price_feat.index.isin(stock_fin_feat.index)]
            stock_fin_feat = stock_fin_feat.loc[stock_fin_feat.index.isin(stock_price_feat.index)]

            all_feat = pd.concat([stock_fin_feat, stock_price_feat], axis=1)#.dropna()

            # Feature V2
            all_feat['時価総額'] = all_feat['EndOfDayQuote ExchangeOfficialClose'] * all_feat['IssuedShareEquityQuote_IssuedShare']
            all_feat['EPS'] = all_feat['今期当期純利益'] / all_feat['IssuedShareEquityQuote_IssuedShare']
            all_feat['株ごと純資産'] = all_feat['純資産'] /  all_feat['IssuedShareEquityQuote_IssuedShare']
            all_feat['株ごと総資産'] = all_feat['総資産'] /  all_feat['IssuedShareEquityQuote_IssuedShare']
            all_feat['PBR'] = all_feat['EndOfDayQuote ExchangeOfficialClose']  / all_feat['株ごと純資産']
            all_feat['PBR-1'] = all_feat['株ごと純資産'] / all_feat['EndOfDayQuote ExchangeOfficialClose']
            all_feat['PER'] = all_feat['時価総額']  / all_feat['当期純利益']
            # Feature V2 over

            all_feat = all_feat.loc[pd.Timestamp(start_dt) :]
            feats = pd.concat([feats, all_feat])

        # 欠損値処理を行います。
        # feats = feats.replace([np.inf, -np.inf], 0)
        # 銘柄コードを設定
    #     feats["code"] = code

        # 生成対象日以降の特徴量に絞る
    #     feats = feats.loc[pd.Timestamp(start_dt) :]

        return feats


    @classmethod
    def get_model(cls, model_path="../model/", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if cls.models_fname is None:
            cls.models_fname = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            m = os.path.join(model_path, f"model_{label}.pkl")
            with open(m, "rb") as f:
                # pickle形式で保存されているモデルを読み込み
                model = pickle.load(f)
                # cols_when_model_builds = model.feature_names
                cols_when_model_builds = model.feature_name()

                cls.models[label] = model
                cls.models_fname[label] = cols_when_model_builds

        return True

    @classmethod
    def clean_feature_stock_list(cls, df_org, codes):
        df = df_org[df_org['Local Code'].isin(codes)].copy()
        stock_list_df_use_cols = ['Local Code', 'Section/Products', '33 Sector(Code)', '17 Sector(Code)',
         'Size Code (New Index Series)', 'Size (New Index Series)', 'IssuedShareEquityQuote AccountingStandard',
          'IssuedShareEquityQuote IssuedShare']

        enc = LabelEncoder()
        clean_list = ['Section/Products',
                     'IssuedShareEquityQuote AccountingStandard',
                     'Size Code (New Index Series)',
                     'Size (New Index Series)']

        for cat in clean_list:
            df[cat] = enc.fit_transform(df[cat]).astype('int')

        df['Size Code (New Index Series)'] = df['Size Code (New Index Series)'].replace(['-'], 0)
        res_df = df[stock_list_df_use_cols].copy()

        res_df.rename(columns={'33 Sector(Code)': '33_Sector(Code)',
              '17 Sector(Code)': '17_Sector(Code)',
              'Size Code (New Index Series)':'Size_Code_(New_Index_Series)',
              'Size (New Index Series)':'Size_(New_Index_Series)',
              'IssuedShareEquityQuote AccountingStandard':'IssuedShareEquityQuote_AccountingStandard',
              'IssuedShareEquityQuote IssuedShare':'IssuedShareEquityQuote_IssuedShare',
          }, inplace = True)

        return res_df


    @classmethod
    def clean_fin_feature(cls, df_org, codes):
        fin_columns_trans ={
        #     'base_date':'日付',
        # 'Local Code':'銘柄コード',
        'Result_FinancialStatement AccountingStandard':'会計基準',
        'Result_FinancialStatement FiscalPeriodEnd':'決算期',
        'Result_FinancialStatement ReportType':'決算種別',
        'Result_FinancialStatement FiscalYear':'決算年度',
        'Result_FinancialStatement ModifyDate':'更新日',
        'Result_FinancialStatement CompanyType':'会社区分',
        'Result_FinancialStatement ChangeOfFiscalYearEnd':'決算期変更フラグ',
        'Result_FinancialStatement NetSales':'売上高',
        'Result_FinancialStatement OperatingIncome':'営業利益',
        'Result_FinancialStatement OrdinaryIncome':'経常利益',
        'Result_FinancialStatement NetIncome':'当期純利益',
        'Result_FinancialStatement TotalAssets':'総資産',
        'Result_FinancialStatement NetAssets':'純資産',
        'Result_FinancialStatement CashFlowsFromOperatingActivities':'営業CashFlow',
        'Result_FinancialStatement CashFlowsFromFinancingActivities':'財務CashFlow',
        'Result_FinancialStatement CashFlowsFromInvestingActivities':'投資CashFlow',
        'Forecast_FinancialStatement AccountingStandard':'会計基準予想',
        'Forecast_FinancialStatement FiscalPeriodEnd':'来期予想決算期',
        'Forecast_FinancialStatement ReportType':'来期予想決算種別',
        'Forecast_FinancialStatement FiscalYear':'来期予想決算年度',
        'Forecast_FinancialStatement ModifyDate':'来期予想更新日',
        'Forecast_FinancialStatement CompanyType':'来期予想会社区分',
        'Forecast_FinancialStatement ChangeOfFiscalYearEnd':'来期予想決算期変更Flag',
        'Forecast_FinancialStatement NetSales':'来期予想売上高',
        'Forecast_FinancialStatement OperatingIncome':'来期予想営業利益',
        'Forecast_FinancialStatement OrdinaryIncome':'来期予想経常利益',
        'Forecast_FinancialStatement NetIncome':'来期予想当期純利益',
        'Result_Dividend FiscalPeriodEnd':'配当決算期',
        'Result_Dividend ReportType':'配当決算種別',
        'Result_Dividend FiscalYear':'配当決算年度',
        'Result_Dividend ModifyDate':'配当更新日',
        'Result_Dividend RecordDate':'配当基準日',
        'Result_Dividend DividendPayableDate':'配当支払開始日',
        'Result_Dividend QuarterlyDividendPerShare':'一株四半期配当金',
        'Result_Dividend AnnualDividendPerShare':'一株年間配当金累計',
        'Forecast_Dividend FiscalPeriodEnd':'予想配当決算期',
        'Forecast_Dividend ReportType':'予想配当決算種別',
        'Forecast_Dividend FiscalYear':'予想配当決算年度',
        'Forecast_Dividend ModifyDate':'予想配当更新日',
        'Forecast_Dividend RecordDate':'予想配当基準日',
        'Forecast_Dividend QuarterlyDividendPerShare':'予想一株四半期配当金',
        'Forecast_Dividend AnnualDividendPerShare':'予想一株年間配当金累計',}

        renew_columns = ['決算期', '決算種別','決算年度','売上高', '営業利益', '経常利益' , '当期純利益', '総資産','純資産',
                            '営業CashFlow', '財務CashFlow', '投資CashFlow', '今期売上高', '今期営業利益', '今期経常利益', '今期当期純利益',
               '今期売上高伸び','今期営業利益伸び', '今期経常利益伸び',  '今期当期純利益伸び']


        def get_fin_quarterly(prev_row, row):
            if prev_row['決算年度'] == row['決算年度'] and prev_row['決算種別'] == row['決算種別']:
                n_s = row['決算種別']
                n = int(n_s[-1])
                return row['売上高']/n ,row['営業利益']/n ,row['経常利益']/n ,row['当期純利益']/n

            uri = row['売上高'] - prev_row['売上高']
            eigyo = row['営業利益'] - prev_row['営業利益']
            keijyo = row['経常利益'] - prev_row['経常利益']
            jyun = row['当期純利益'] - prev_row['当期純利益']

            return uri, eigyo, keijyo,jyun

        def get_fin_quarterly_df(df):

            local_index = 0
            uri_list = []
            eigyo_list = []
            keijyo_list = []
            jyun_list = []

            for index, row in df.iterrows():
                uri, eigyo, keijyo,jyun = 0, 0, 0, 0
                base_row = row
                if local_index == 0 or row['決算種別'] == 'Q1':
                    base_row = row
                elif row['決算年度'] == prev_row['決算年度'] and int(row['決算種別'][-1]) == int(prev_row['決算種別'][-1]) + 1:
                    base_row = prev_row
                else:
                    base_row = row

                uri, eigyo, keijyo,jyun = get_fin_quarterly(base_row, row)
                uri_list.append(uri)
                eigyo_list.append(eigyo)
                keijyo_list.append(keijyo)
                jyun_list.append(jyun)

                prev_row = row
                local_index +=1

            quarterly_df = pd.DataFrame([uri_list, eigyo_list, keijyo_list, jyun_list])
            quarterly_df = quarterly_df.T
            quarterly_df.columns = ['今期売上高', '今期営業利益', '今期経常利益', '今期当期純利益']
            quarterly_df.columns = ['今期売上高', '今期営業利益', '今期経常利益', '今期当期純利益']
            quarterly_df["今期売上高伸び"] = quarterly_df['今期売上高'].diff() / quarterly_df['今期売上高'].abs().shift()
            quarterly_df["今期営業利益伸び"] = quarterly_df['今期営業利益'].diff() / quarterly_df['今期営業利益'].abs().shift()
            quarterly_df["今期経常利益伸び"] = quarterly_df['今期経常利益'].diff() / quarterly_df['今期経常利益'].abs().shift()
            quarterly_df["今期当期純利益伸び"] = quarterly_df['今期当期純利益'].diff() / quarterly_df['今期当期純利益'].abs().shift()
            quarterly_df[['今期売上高伸び', '今期営業利益伸び', '今期経常利益伸び', '今期当期純利益伸び']] = quarterly_df[['今期売上高伸び',
            '今期営業利益伸び', '今期経常利益伸び', '今期当期純利益伸び']].fillna('0')
            quarterly_df['今期売上高伸び'] = quarterly_df['今期売上高伸び'].astype(float)
            quarterly_df['今期営業利益伸び'] = quarterly_df['今期営業利益伸び'].astype(float)
            quarterly_df['今期経常利益伸び'] = quarterly_df['今期経常利益伸び'].astype(float)
            quarterly_df['今期当期純利益伸び'] = quarterly_df['今期当期純利益伸び'].astype(float)

            quarterly_df.index = df.index

            return quarterly_df

        def find_near_date(new_time, base_time, delta = 60):
            res = False
            if new_time < base_time + np.timedelta64(delta,'D') or new_time > base_time - np.timedelta64(delta,'D'):
                res = True
            return res



        def find_near_date(new_time, base_time, delta = 60):
            res = False
            if new_time < base_time + np.timedelta64(delta,'D') or new_time > base_time - np.timedelta64(delta,'D'):
                res = True
            return res

        def get_yoso_changerate(stock_fin):
            df = stock_fin.copy()
            uri_list = []
            eigyo_list = []
            keijyo_list = []
            jyun_list = []
            local_index = 0

            for index, row in df.iterrows():
                uri, eigyo, keijyo,jyun = 0, 0, 0, 0
            #     base_row = row
                if local_index == 0:
                    prev_row = row

                if local_index != 0 and row['来期予想決算期'] is not np.nan and prev_row['来期予想決算期'] is not np.nan:


    #                 if prev_row['来期予想売上高'] is not np.nan and row['来期予想売上高'] is not np.nan and :
                    if prev_row['来期予想売上高'] != 0:
                        if row['来期予想決算期'] == prev_row['来期予想決算期'] and int(row['来期予想売上高']) != int(prev_row['来期予想売上高']):
                            uri = (row['来期予想売上高'] - prev_row['来期予想売上高']) / np.abs(prev_row['来期予想売上高'])

    #                 if prev_row['来期予想営業利益'] is not np.nan and row['来期予想営業利益'] is not np.nan:
                    if prev_row['来期予想営業利益'] != 0:
                        if row['来期予想決算期'] == prev_row['来期予想決算期'] and int(row['来期予想営業利益']) != int(prev_row['来期予想営業利益']):
                            eigyo = (row['来期予想営業利益'] - prev_row['来期予想営業利益']) / np.abs(prev_row['来期予想営業利益'])

    #                 if prev_row['来期予想経常利益'] is not np.nan and row['来期予想経常利益'] is not np.nan :
                    if prev_row['来期予想経常利益'] != 0:
                        if row['来期予想決算期'] == prev_row['来期予想決算期'] and int(row['来期予想経常利益']) != int(prev_row['来期予想経常利益']):
                            keijyo = (row['来期予想経常利益'] - prev_row['来期予想経常利益']) / np.abs(prev_row['来期予想経常利益'])

    #                 if prev_row['来期予想当期純利益'] is not np.nan and row['来期予想当期純利益'] is not np.nan:
                    if prev_row['来期予想当期純利益'] != 0:
                        if row['来期予想決算期'] == prev_row['来期予想決算期'] and int(row['来期予想当期純利益']) != int(prev_row['来期予想当期純利益']):
                            jyun = (row['来期予想当期純利益'] - prev_row['来期予想当期純利益']) / np.abs(prev_row['来期予想当期純利益'])

                uri_list.append(uri)
                eigyo_list.append(eigyo)
                keijyo_list.append(keijyo)
                jyun_list.append(jyun)

                prev_row = row
                local_index += 1

            quarterly_df = pd.DataFrame([uri_list, eigyo_list, keijyo_list, jyun_list])
            quarterly_df = quarterly_df.T
            quarterly_df.columns = ['来期予想売上高変更率', '来期予想営業利益変更率', '来期予想経常利益変更率', '来期予想当期純利益変更率']
            quarterly_df.index = df.index
            quarterly_df = pd.concat([df, quarterly_df], axis=1)
            return quarterly_df

        def get_pred_actual_rate(stock_fin):
            df = stock_fin.copy()
            uri_list = []
            eigyo_list = []
            keijyo_list = []
            jyun_list = []
            yoso_list = stock_fin['来期予想決算期'].values.tolist()

            had_list = []  # 二番目は計算しない。
            for index, row in df.iterrows():
                uri, eigyo, keijyo,jyun = 0, 0, 0, 0
                df_back = stock_fin.copy()
                if row['決算期'] in yoso_list and row['決算期'] not in had_list:
                    search_df = df_back.drop_duplicates(subset=['来期予想決算期', ],keep="last")
                    search_df = search_df[search_df['来期予想決算期'] == row['決算期']]

                    values = search_df['来期予想売上高'].values
                    uri =  values[0] if values.size > 0 else 0

                    values = search_df['来期予想営業利益'].values
                    eigyo =  values[0] if values.size > 0 else 0

                    values = search_df['来期予想経常利益'].values
                    keijyo =  values[0] if values.size > 0 else 0

                    values = search_df['来期予想当期純利益'].values
                    jyun =  values[0] if values.size > 0 else 0

                    had_list.append(row['決算期'])

                uri_list.append(uri)
                eigyo_list.append(eigyo)
                keijyo_list.append(keijyo)
                jyun_list.append(jyun)


            quarterly_df = pd.DataFrame([uri_list, eigyo_list, keijyo_list, jyun_list])
            quarterly_df = quarterly_df.T
            quarterly_df.columns = ['前予想売上高', '前予想営業利益', '前予想経常利益', '前予想当期純利益']
            quarterly_df.index = df.index
            quarterly_df = pd.concat([df, quarterly_df], axis=1)


            return quarterly_df

        def get_ratio(pred, actual):
            res = 0
            if pred is not np.nan and actual is not np.nan:
                if pred != 0:
                    res = (actual - pred) / np.abs(pred)

            return res

        df = df_org.copy()
        df.rename(columns=fin_columns_trans, inplace = True)
    #     print (df.columns)

        res_df = pd.DataFrame()

        # Feature V4
        df[['来期予想売上高', '来期予想営業利益', '来期予想経常利益', '来期予想当期純利益',
             '売上高', '営業利益', '経常利益', '当期純利益']] = df[[
            '来期予想売上高', '来期予想営業利益', '来期予想経常利益', '来期予想当期純利益',
            '売上高', '営業利益', '経常利益', '当期純利益']].fillna(0)

        for code in codes:

            stock_fin_tmp = df[df['Local Code'] == code].copy()
            stock_fin_tmp_org = df[df['Local Code'] == code].copy()

            org_len = stock_fin_tmp_org.shape[0]

            #Feature V4
            stock_fin_tmp = get_yoso_changerate(stock_fin_tmp)
            stock_fin_tmp = get_pred_actual_rate(stock_fin_tmp)

            stock_fin_tmp['来期予想売上高変更率']     = stock_fin_tmp['来期予想売上高変更率'].astype(float)
            stock_fin_tmp['来期予想営業利益変更率']   = stock_fin_tmp['来期予想営業利益変更率'].astype(float)
            stock_fin_tmp['来期予想経常利益変更率']   = stock_fin_tmp['来期予想経常利益変更率'].astype(float)
            stock_fin_tmp['来期予想当期純利益変更率'] = stock_fin_tmp['来期予想当期純利益変更率'].astype(float)

            stock_fin_tmp['前予想売上高']     = stock_fin_tmp['前予想売上高'].astype(float)
            stock_fin_tmp['前予想営業利益']   = stock_fin_tmp['前予想営業利益'].astype(float)
            stock_fin_tmp['前予想経常利益']   = stock_fin_tmp['前予想経常利益'].astype(float)
            stock_fin_tmp['前予想当期純利益'] = stock_fin_tmp['前予想当期純利益'].astype(float)

            stock_fin_tmp["当期純利益予実績比"] = stock_fin_tmp.apply(lambda row: get_ratio(row['前予想当期純利益'], row['当期純利益']), axis=1).astype(float)
            stock_fin_tmp["経常利益予実績比"]   = stock_fin_tmp.apply(lambda row: get_ratio(row['前予想経常利益'], row['経常利益']), axis=1).astype(float)
            stock_fin_tmp["営業利益予実績比"]   = stock_fin_tmp.apply(lambda row: get_ratio(row['前予想営業利益'], row['営業利益']), axis=1).astype(float)
            stock_fin_tmp["売上高予実績比"]     = stock_fin_tmp.apply(lambda row: get_ratio(row['前予想売上高'], row['売上高']), axis=1).astype(float)


            # 実績財務情報を処理
            stock_fin_tmp_nan = stock_fin_tmp[stock_fin_tmp['更新日'].isna()]
            stock_fin_tmp = stock_fin_tmp[~stock_fin_tmp['更新日'].isna()]
            stock_fin_tmp['決算種別'] = stock_fin_tmp['決算種別'].replace(['Annual'], 'Q4')

            stock_fin_tmp_back = stock_fin_tmp.copy()
            stock_fin_tmp.drop_duplicates(subset=["決算期",'決算種別',"更新日"],keep="first", inplace=True)

            quarterly_df = get_fin_quarterly_df(stock_fin_tmp)
            stock_fin_tmp_n = pd.concat([stock_fin_tmp,quarterly_df ], axis = 1)

            # Feature V2
            stock_fin_tmp_n["今期総資産伸び"] = stock_fin_tmp_n['総資産'].diff() / stock_fin_tmp_n['総資産'].abs().shift()
            stock_fin_tmp_n["今期純資産伸び"] = stock_fin_tmp_n['純資産'].diff() / stock_fin_tmp_n['純資産'].abs().shift()
            stock_fin_tmp_n["今期営業利益率"] = stock_fin_tmp_n['今期営業利益'] / stock_fin_tmp_n['今期売上高']
            stock_fin_tmp_n["今期純利益率"] = stock_fin_tmp_n['今期当期純利益'] / stock_fin_tmp_n['総資産']

            stock_fin_tmp_n['今期総資産伸び'] = stock_fin_tmp_n['今期総資産伸び'].astype(float)
            stock_fin_tmp_n['今期純資産伸び'] = stock_fin_tmp_n['今期純資産伸び'].astype(float)
            stock_fin_tmp_n["今期営業利益率"] = stock_fin_tmp_n['今期営業利益'].astype(float)
            stock_fin_tmp_n["今期純利益率"] = stock_fin_tmp_n['今期純利益率'].astype(float)

            stock_fin_tmp_back_n = pd.merge(stock_fin_tmp_back, stock_fin_tmp_n[[
                '今期売上高', '今期営業利益', '今期経常利益', '今期当期純利益',
            '今期売上高伸び','今期営業利益伸び', '今期経常利益伸び',  '今期当期純利益伸び',
            '今期総資産伸び','今期純資産伸び',"今期営業利益率", "今期純利益率",
             "決算期",'決算種別',"更新日"]], on=["決算期",'決算種別',  "更新日"], how = 'outer')

            # Feature V2 over
            stock_fin_tmp_back_n.index = stock_fin_tmp_back.index

            tmp_df = pd.concat([stock_fin_tmp_nan, stock_fin_tmp_back_n], sort=True)
            tmp_df.sort_index(inplace = True)

            # Nan の内容を処理

            stock_fin_tmp_nan = tmp_df[tmp_df['更新日'].isna()]
            stock_fin_tmp = tmp_df[~tmp_df['更新日'].isna()]
            tmp_df = pd.DataFrame()
            for i in range(len(stock_fin_tmp_nan)):
                find = 0
                base_time = stock_fin_tmp_nan.index.values[i]

                for index, row in stock_fin_tmp.iterrows():
                    if find_near_date(index,base_time):
                        find = 1
                        stock_fin_tmp_nan.loc[base_time, renew_columns] = row[renew_columns].values
                        break
                if find == 0:
                    print ('fatal error ', code)

            tmp_df = pd.concat([stock_fin_tmp_nan, stock_fin_tmp])
            tmp_df.sort_index(inplace=True)

            res_len = tmp_df.shape[0]
            if res_len != org_len :
                print ('not same len ', code)

            res_df = pd.concat([res_df, tmp_df])

        return res_df


    @classmethod
    def clean_feature_price(cls, df_org, code, start_dt="2016-01-01", n = 90):
        # stock_priceデータを読み込む
    #     price = dfs["stock_price"]
        price = df_org.copy()
        # 特定の銘柄コードのデータに絞る
        price_data = price[price["Local Code"] == code]
        # 終値のみに絞る
        feats = price_data[["EndOfDayQuote ExchangeOfficialClose"]]

        # 特徴量の生成対象期間を指定
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :].copy()

    #     ema_list = talib.EMA(feats['EndOfDayQuote ExchangeOfficialClose'].values, timeperiod=20).tolist()
    #     feats = pd.concat([feats, pd.DataFrame(ema_list, index=feats.index, columns=['ema_20'])], axis=1)
        feats['ema_20'] = feats['EndOfDayQuote ExchangeOfficialClose'].ewm(span=20,min_periods=1).mean()
        feats['ema_10'] = feats['EndOfDayQuote ExchangeOfficialClose'].ewm(span=10,min_periods=1).mean()
        feats['ema_5'] = feats['EndOfDayQuote ExchangeOfficialClose'].ewm(span=5,min_periods=1).mean()

    #     rsi_list = talib.RSI(feats['EndOfDayQuote ExchangeOfficialClose'].values, timeperiod=14).tolist()
    #     feats = pd.concat([feats, pd.DataFrame(rsi_list, index=feats.index, columns=['rsi'])], axis=1)
        feats['rsi'] = cls.get_rsi(feats['EndOfDayQuote ExchangeOfficialClose'], 14)

    #     macd, macdsignal, macdhist = talib.MACD(feats['EndOfDayQuote ExchangeOfficialClose'].values,
    #                                             fastperiod=12, slowperiod=26, signalperiod=9)
    #     feats = pd.concat([feats, pd.DataFrame(macd, index=feats.index, columns=['macd'])], axis=1)
    #     feats = pd.concat([feats, pd.DataFrame(macdsignal, index=feats.index, columns=['macdsignal'])], axis=1)
    #     feats = pd.concat([feats, pd.DataFrame(macdhist, index=feats.index, columns=['macdhist'])], axis=1)
        macd, macdsignal, macdhist = cls.get_macd(feats['EndOfDayQuote ExchangeOfficialClose'])
        macd_df = pd.concat([macd, macdsignal, macdhist],  axis=1 )
        macd_df.columns =['macd', 'macdsignal', 'macdhist']
        feats = pd.concat([feats, macd_df], axis=1)

        feats['diff_close_ema_20'] = feats['EndOfDayQuote ExchangeOfficialClose'] - feats['ema_20']
        feats['diff_close_ema_10'] = feats['EndOfDayQuote ExchangeOfficialClose'] - feats['ema_10']
        feats['diff_close_ema_5'] = feats['EndOfDayQuote ExchangeOfficialClose'] - feats['ema_5']

        feats['ema_20'] = feats['ema_20'].replace([np.nan], 0)
        feats['ema_10'] = feats['ema_10'].replace([np.nan], 0)
        feats['ema_5'] = feats['ema_5'].replace([np.nan], 0)

        feats['macd'] = feats['macd'].replace([np.nan], 0)
        feats['macdsignal'] = feats['macdsignal'].replace([np.nan], 0)
        feats['macdhist'] = feats['macdhist'].replace([np.nan], 0)
        feats["macdhist_diff1"] = feats["macdhist"].diff()

        feats['rsi'] = feats['rsi'].replace([np.nan], -1)

        feats['diff_close_ema_20'] = feats['diff_close_ema_20'].replace([np.nan], 0)
        feats['diff_close_ema_10'] = feats['diff_close_ema_10'].replace([np.nan], 0)
        feats['diff_close_ema_5'] = feats['diff_close_ema_5'].replace([np.nan], 0)
        # 終値の20営業日リターン
        feats["return_1month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(20)
        # 終値の40営業日リターン
        feats["return_2month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(40)
        # 終値の60営業日リターン
        feats["return_3month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(60)
        # 終値の20営業日ボラティリティ
        feats["volatility_1month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(20)
            .std()
        )
        # 終値の40営業日ボラティリティ
        feats["volatility_2month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(40)
            .std()
        )
        # 終値の60営業日ボラティリティ
        feats["volatility_3month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(60)
            .std()
        )
        # 終値と20営業日の単純移動平均線の乖離
        feats["MA_gap_1month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(20).mean()
        )
        # 終値と40営業日の単純移動平均線の乖離
        feats["MA_gap_2month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(40).mean()
        )
        # 終値と60営業日の単純移動平均線の乖離
        feats["MA_gap_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(60).mean()
        )
        # 欠損値処理
        feats = feats.fillna(0)
        # 元データのカラムを削除
    #     feats = feats.drop(["EndOfDayQuote ExchangeOfficialClose"], axis=1)

        return feats

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """

        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # 特徴量を作成
        feats  = cls.get_features_for_predict(cls.dfs, codes, start_dt)

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["Local Code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "Local Code"].astype(
            str
        )

        # 出力対象列を定義
        output_columns = ["code"]

        # 特徴量カラムを指定
        # feature_columns = cls.get_feature_columns(feats)

        # 目的変数毎に予測
        for label in labels:
            # 予測実施
    #         df[label] = cls.models[label].predict(xgb.DMatrix(feats[feature_columns]))
            feats = feats[cls.models_fname[label]]
            # df[label] = cls.models[label].predict(xgb.DMatrix(feats))
            df[label] = cls.models[label].predict(feats)
            # 出力対象列に追加
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)
    #     df.to_csv('res.csv', header=False, index=False, columns=output_columns)

        return out.getvalue()

    @classmethod
    def save_model(cls, model, label, model_path="../model"):
        """
        Args:
            model (RandomForestRegressor): trained model
            label (str): prediction target label
            model_path (str): path to save model
        Returns:
            -
        """
        # tag::save_model_partial[]
        # モデル保存先ディレクトリを作成
    #     os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"model_{label}.pkl"), "wb") as f:
            # モデルをpickle形式で保存
            pickle.dump(model, f)
        # end::save_model_partial[]

    @classmethod
    def get_features_and_label_release(cls, stock_labels_df, codes, feature, label):
        """
        Args:
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
        """
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["Local Code"] == code]
    #         print (feats.index)

            # stock_labelデータを読み込み
            stock_labels = stock_labels_df
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]

            # 特定の目的変数に絞る
            labels = stock_labels[label].copy()
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats
    #             _val_X = feats[VAL_START : VAL_END]
    #             _test_X = feats[TEST_START :]

                _train_y = labels
    #             _val_y = labels[VAL_START : VAL_END]
    #             _test_y = labels[TEST_START :]

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
    #             vals_X.append(_val_X)
    #             tests_X.append(_test_X)

                trains_y.append(_train_y)
    #             vals_y.append(_val_y)
    #             tests_y.append(_test_y)
        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
    #     val_X = pd.concat(vals_X)
    #     test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
    #     val_y = pd.concat(vals_y)
    #     test_y = pd.concat(tests_y)

        return train_X, train_y

    @classmethod
    def train_lgb_release_v2(cls, X_train, Y_train, ishigh = True):
        def lgb_srcc(preds, dtrain):
            # 正解ラベル
            correlation, _ = spearmanr(preds, dtrain.get_label())
            res = np.square(correlation - 1)
            # name, result, is_higher_better
            return 'my_srcc', res, False

        dtrain = lgb.Dataset(X_train, label=Y_train)
    #     dvalid = lgb.Dataset(X_valid, label=Y_valid)

        evaluation_results = {}

        if ishigh :
            lgb_params = {
                'objective': 'regression',
                'boosting' : 'gbdt',

                'learning_rate': 0.0055236664051955976,
                'lambda_l1': 0.1587286765153048,
                'lambda_l2': 8.833730956821778,
                'num_leaves': 189,
                'feature_fraction': 0.5166696703449053,
                'bagging_fraction': 0.5990863491585487,
                'bagging_freq': 3,
                'feature_fraction_bynode': 0.4008196760802559,
                'min_child_samples': 68,
                'extra_trees': False,

                'device': 'gpu',
                'gpu_platform_id':0,
                'gpu_device_id':0,
                'seed':2021,
                'gpu_use_dp':True,

            }
            n_round = 1358
        else:
            lgb_params = {
                'objective': 'regression',
                'boosting' : 'gbdt',

                'learning_rate': 0.007740353750838104,
                'lambda_l1': 0.001278456428331565,
                'lambda_l2': 7.135400428360081,
                'num_leaves': 197,
                'feature_fraction': 0.7547278991463711,
                'bagging_fraction': 0.751573777368312,
                'bagging_freq': 1,
                'feature_fraction_bynode': 0.4404796208698425,
                'min_child_samples': 27,
                'extra_trees': True,

                'device': 'gpu',
                'gpu_platform_id':0,
                'gpu_device_id':0,
                'seed':2021,
                'gpu_use_dp':True,
            }
            n_round = 2875

        model = lgb.train(params=lgb_params,
                          train_set=dtrain,
    #                       valid_sets=[dvalid],
    #                       valid_names=['Valid'],
                          num_boost_round =n_round,
    #                       early_stopping_rounds = 200,
                          verbose_eval = -1,
                          evals_result=evaluation_results,
                          feval =lgb_srcc,
        )


        return model

    @classmethod
    def get_feature_columns( cls, feats):

    #     not_use_list = ['Local Code', '決算年度','EndOfDayQuote ExchangeOfficialClose',]
        not_use_list = ['Local Code', '決算年度','EndOfDayQuote ExchangeOfficialClose',
                       '来期予想決算年度', 'IssuedShareEquityQuote_IssuedShare',
                        '配当決算年度',
                       '一株四半期配当金','一株年間配当金累計', '予想配当決算年度', '予想一株四半期配当金', '予想一株年間配当金累計',
                       ]

        res_list = list(set(feats.columns.tolist()).difference(set(not_use_list)))

        return res_list


    @classmethod
    def train( cls, inputs, start_dt="2016-01-01"):
        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        codes = cls.codes
        # 特徴量を作成
        print ("goto get_features_for_predict")
        feats  = cls.get_features_for_predict(cls.dfs, codes, start_dt)

        f_list = cls.get_feature_columns(feats)
        stock_labels_df = cls.dfs['stock_labels']
        feature = feats

        train_x_high, train_y_high = cls.get_features_and_label_release(stock_labels_df, codes, feature, 'label_high_20')
        train_x_low,  train_y_low  = cls.get_features_and_label_release(stock_labels_df, codes, feature, 'label_low_20')

        print ("goto train")
        model_high = cls.train_lgb_release_v2(train_x_high[f_list], train_y_high, ishigh = True)
        model_low  = cls.train_lgb_release_v2(train_x_low[f_list], train_y_low, ishigh = False)

        cls.save_model(model_high, 'label_high_20')
        cls.save_model(model_low, 'label_low_20')



