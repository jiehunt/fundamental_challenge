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
    # �P�����ԏI����
    TRAIN_END = "2018-12-31"
    # �]�����ԊJ�n��
    VAL_START = "2019-02-01"
    # �]�����ԏI����
    VAL_END = "2019-12-01"
    # �e�X�g���ԊJ�n��
    TEST_START = "2020-01-01"
    # �ړI�ϐ�
    TARGET_LABELS = ['label_high_20', 'label_low_20']

    # �f�[�^�����̕ϐ��ɓǂݍ���
    dfs = None
    # ���f�������̕ϐ��ɓǂݍ���
    models = None
    # �Ώۂ̖����R�[�h�����̕ϐ��ɓǂݍ���
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
            # DataFrame��index��ݒ肵�܂��B
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
        # �\���Ώۂ̖����R�[�h���擾
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
            all_feat['�������z'] = all_feat['EndOfDayQuote ExchangeOfficialClose'] * all_feat['IssuedShareEquityQuote_IssuedShare']
            all_feat['EPS'] = all_feat['�������������v'] / all_feat['IssuedShareEquityQuote_IssuedShare']
            all_feat['�����Ə����Y'] = all_feat['�����Y'] /  all_feat['IssuedShareEquityQuote_IssuedShare']
            all_feat['�����Ƒ����Y'] = all_feat['�����Y'] /  all_feat['IssuedShareEquityQuote_IssuedShare']
            all_feat['PBR'] = all_feat['EndOfDayQuote ExchangeOfficialClose']  / all_feat['�����Ə����Y']
            all_feat['PBR-1'] = all_feat['�����Ə����Y'] / all_feat['EndOfDayQuote ExchangeOfficialClose']
            all_feat['PER'] = all_feat['�������z']  / all_feat['���������v']
            # Feature V2 over

            all_feat = all_feat.loc[pd.Timestamp(start_dt) :]
            feats = pd.concat([feats, all_feat])

        # �����l�������s���܂��B
        # feats = feats.replace([np.inf, -np.inf], 0)
        # �����R�[�h��ݒ�
    #     feats["code"] = code

        # �����Ώۓ��ȍ~�̓����ʂɍi��
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
                # pickle�`���ŕۑ�����Ă��郂�f����ǂݍ���
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
        #     'base_date':'���t',
        # 'Local Code':'�����R�[�h',
        'Result_FinancialStatement AccountingStandard':'��v�',
        'Result_FinancialStatement FiscalPeriodEnd':'���Z��',
        'Result_FinancialStatement ReportType':'���Z���',
        'Result_FinancialStatement FiscalYear':'���Z�N�x',
        'Result_FinancialStatement ModifyDate':'�X�V��',
        'Result_FinancialStatement CompanyType':'��Ћ敪',
        'Result_FinancialStatement ChangeOfFiscalYearEnd':'���Z���ύX�t���O',
        'Result_FinancialStatement NetSales':'���㍂',
        'Result_FinancialStatement OperatingIncome':'�c�Ɨ��v',
        'Result_FinancialStatement OrdinaryIncome':'�o�험�v',
        'Result_FinancialStatement NetIncome':'���������v',
        'Result_FinancialStatement TotalAssets':'�����Y',
        'Result_FinancialStatement NetAssets':'�����Y',
        'Result_FinancialStatement CashFlowsFromOperatingActivities':'�c��CashFlow',
        'Result_FinancialStatement CashFlowsFromFinancingActivities':'����CashFlow',
        'Result_FinancialStatement CashFlowsFromInvestingActivities':'����CashFlow',
        'Forecast_FinancialStatement AccountingStandard':'��v��\�z',
        'Forecast_FinancialStatement FiscalPeriodEnd':'�����\�z���Z��',
        'Forecast_FinancialStatement ReportType':'�����\�z���Z���',
        'Forecast_FinancialStatement FiscalYear':'�����\�z���Z�N�x',
        'Forecast_FinancialStatement ModifyDate':'�����\�z�X�V��',
        'Forecast_FinancialStatement CompanyType':'�����\�z��Ћ敪',
        'Forecast_FinancialStatement ChangeOfFiscalYearEnd':'�����\�z���Z���ύXFlag',
        'Forecast_FinancialStatement NetSales':'�����\�z���㍂',
        'Forecast_FinancialStatement OperatingIncome':'�����\�z�c�Ɨ��v',
        'Forecast_FinancialStatement OrdinaryIncome':'�����\�z�o�험�v',
        'Forecast_FinancialStatement NetIncome':'�����\�z���������v',
        'Result_Dividend FiscalPeriodEnd':'�z�����Z��',
        'Result_Dividend ReportType':'�z�����Z���',
        'Result_Dividend FiscalYear':'�z�����Z�N�x',
        'Result_Dividend ModifyDate':'�z���X�V��',
        'Result_Dividend RecordDate':'�z�����',
        'Result_Dividend DividendPayableDate':'�z���x���J�n��',
        'Result_Dividend QuarterlyDividendPerShare':'�ꊔ�l�����z����',
        'Result_Dividend AnnualDividendPerShare':'�ꊔ�N�Ԕz�����݌v',
        'Forecast_Dividend FiscalPeriodEnd':'�\�z�z�����Z��',
        'Forecast_Dividend ReportType':'�\�z�z�����Z���',
        'Forecast_Dividend FiscalYear':'�\�z�z�����Z�N�x',
        'Forecast_Dividend ModifyDate':'�\�z�z���X�V��',
        'Forecast_Dividend RecordDate':'�\�z�z�����',
        'Forecast_Dividend QuarterlyDividendPerShare':'�\�z�ꊔ�l�����z����',
        'Forecast_Dividend AnnualDividendPerShare':'�\�z�ꊔ�N�Ԕz�����݌v',}

        renew_columns = ['���Z��', '���Z���','���Z�N�x','���㍂', '�c�Ɨ��v', '�o�험�v' , '���������v', '�����Y','�����Y',
                            '�c��CashFlow', '����CashFlow', '����CashFlow', '�������㍂', '�����c�Ɨ��v', '�����o�험�v', '�������������v',
               '�������㍂�L��','�����c�Ɨ��v�L��', '�����o�험�v�L��',  '�������������v�L��']


        def get_fin_quarterly(prev_row, row):
            if prev_row['���Z�N�x'] == row['���Z�N�x'] and prev_row['���Z���'] == row['���Z���']:
                n_s = row['���Z���']
                n = int(n_s[-1])
                return row['���㍂']/n ,row['�c�Ɨ��v']/n ,row['�o�험�v']/n ,row['���������v']/n

            uri = row['���㍂'] - prev_row['���㍂']
            eigyo = row['�c�Ɨ��v'] - prev_row['�c�Ɨ��v']
            keijyo = row['�o�험�v'] - prev_row['�o�험�v']
            jyun = row['���������v'] - prev_row['���������v']

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
                if local_index == 0 or row['���Z���'] == 'Q1':
                    base_row = row
                elif row['���Z�N�x'] == prev_row['���Z�N�x'] and int(row['���Z���'][-1]) == int(prev_row['���Z���'][-1]) + 1:
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
            quarterly_df.columns = ['�������㍂', '�����c�Ɨ��v', '�����o�험�v', '�������������v']
            quarterly_df.columns = ['�������㍂', '�����c�Ɨ��v', '�����o�험�v', '�������������v']
            quarterly_df["�������㍂�L��"] = quarterly_df['�������㍂'].diff() / quarterly_df['�������㍂'].abs().shift()
            quarterly_df["�����c�Ɨ��v�L��"] = quarterly_df['�����c�Ɨ��v'].diff() / quarterly_df['�����c�Ɨ��v'].abs().shift()
            quarterly_df["�����o�험�v�L��"] = quarterly_df['�����o�험�v'].diff() / quarterly_df['�����o�험�v'].abs().shift()
            quarterly_df["�������������v�L��"] = quarterly_df['�������������v'].diff() / quarterly_df['�������������v'].abs().shift()
            quarterly_df[['�������㍂�L��', '�����c�Ɨ��v�L��', '�����o�험�v�L��', '�������������v�L��']] = quarterly_df[['�������㍂�L��',
            '�����c�Ɨ��v�L��', '�����o�험�v�L��', '�������������v�L��']].fillna('0')
            quarterly_df['�������㍂�L��'] = quarterly_df['�������㍂�L��'].astype(float)
            quarterly_df['�����c�Ɨ��v�L��'] = quarterly_df['�����c�Ɨ��v�L��'].astype(float)
            quarterly_df['�����o�험�v�L��'] = quarterly_df['�����o�험�v�L��'].astype(float)
            quarterly_df['�������������v�L��'] = quarterly_df['�������������v�L��'].astype(float)

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

                if local_index != 0 and row['�����\�z���Z��'] is not np.nan and prev_row['�����\�z���Z��'] is not np.nan:


    #                 if prev_row['�����\�z���㍂'] is not np.nan and row['�����\�z���㍂'] is not np.nan and :
                    if prev_row['�����\�z���㍂'] != 0:
                        if row['�����\�z���Z��'] == prev_row['�����\�z���Z��'] and int(row['�����\�z���㍂']) != int(prev_row['�����\�z���㍂']):
                            uri = (row['�����\�z���㍂'] - prev_row['�����\�z���㍂']) / np.abs(prev_row['�����\�z���㍂'])

    #                 if prev_row['�����\�z�c�Ɨ��v'] is not np.nan and row['�����\�z�c�Ɨ��v'] is not np.nan:
                    if prev_row['�����\�z�c�Ɨ��v'] != 0:
                        if row['�����\�z���Z��'] == prev_row['�����\�z���Z��'] and int(row['�����\�z�c�Ɨ��v']) != int(prev_row['�����\�z�c�Ɨ��v']):
                            eigyo = (row['�����\�z�c�Ɨ��v'] - prev_row['�����\�z�c�Ɨ��v']) / np.abs(prev_row['�����\�z�c�Ɨ��v'])

    #                 if prev_row['�����\�z�o�험�v'] is not np.nan and row['�����\�z�o�험�v'] is not np.nan :
                    if prev_row['�����\�z�o�험�v'] != 0:
                        if row['�����\�z���Z��'] == prev_row['�����\�z���Z��'] and int(row['�����\�z�o�험�v']) != int(prev_row['�����\�z�o�험�v']):
                            keijyo = (row['�����\�z�o�험�v'] - prev_row['�����\�z�o�험�v']) / np.abs(prev_row['�����\�z�o�험�v'])

    #                 if prev_row['�����\�z���������v'] is not np.nan and row['�����\�z���������v'] is not np.nan:
                    if prev_row['�����\�z���������v'] != 0:
                        if row['�����\�z���Z��'] == prev_row['�����\�z���Z��'] and int(row['�����\�z���������v']) != int(prev_row['�����\�z���������v']):
                            jyun = (row['�����\�z���������v'] - prev_row['�����\�z���������v']) / np.abs(prev_row['�����\�z���������v'])

                uri_list.append(uri)
                eigyo_list.append(eigyo)
                keijyo_list.append(keijyo)
                jyun_list.append(jyun)

                prev_row = row
                local_index += 1

            quarterly_df = pd.DataFrame([uri_list, eigyo_list, keijyo_list, jyun_list])
            quarterly_df = quarterly_df.T
            quarterly_df.columns = ['�����\�z���㍂�ύX��', '�����\�z�c�Ɨ��v�ύX��', '�����\�z�o�험�v�ύX��', '�����\�z���������v�ύX��']
            quarterly_df.index = df.index
            quarterly_df = pd.concat([df, quarterly_df], axis=1)
            return quarterly_df

        def get_pred_actual_rate(stock_fin):
            df = stock_fin.copy()
            uri_list = []
            eigyo_list = []
            keijyo_list = []
            jyun_list = []
            yoso_list = stock_fin['�����\�z���Z��'].values.tolist()

            had_list = []  # ��Ԗڂ͌v�Z���Ȃ��B
            for index, row in df.iterrows():
                uri, eigyo, keijyo,jyun = 0, 0, 0, 0
                df_back = stock_fin.copy()
                if row['���Z��'] in yoso_list and row['���Z��'] not in had_list:
                    search_df = df_back.drop_duplicates(subset=['�����\�z���Z��', ],keep="last")
                    search_df = search_df[search_df['�����\�z���Z��'] == row['���Z��']]

                    values = search_df['�����\�z���㍂'].values
                    uri =  values[0] if values.size > 0 else 0

                    values = search_df['�����\�z�c�Ɨ��v'].values
                    eigyo =  values[0] if values.size > 0 else 0

                    values = search_df['�����\�z�o�험�v'].values
                    keijyo =  values[0] if values.size > 0 else 0

                    values = search_df['�����\�z���������v'].values
                    jyun =  values[0] if values.size > 0 else 0

                    had_list.append(row['���Z��'])

                uri_list.append(uri)
                eigyo_list.append(eigyo)
                keijyo_list.append(keijyo)
                jyun_list.append(jyun)


            quarterly_df = pd.DataFrame([uri_list, eigyo_list, keijyo_list, jyun_list])
            quarterly_df = quarterly_df.T
            quarterly_df.columns = ['�O�\�z���㍂', '�O�\�z�c�Ɨ��v', '�O�\�z�o�험�v', '�O�\�z���������v']
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
        df[['�����\�z���㍂', '�����\�z�c�Ɨ��v', '�����\�z�o�험�v', '�����\�z���������v',
             '���㍂', '�c�Ɨ��v', '�o�험�v', '���������v']] = df[[
            '�����\�z���㍂', '�����\�z�c�Ɨ��v', '�����\�z�o�험�v', '�����\�z���������v',
            '���㍂', '�c�Ɨ��v', '�o�험�v', '���������v']].fillna(0)

        for code in codes:

            stock_fin_tmp = df[df['Local Code'] == code].copy()
            stock_fin_tmp_org = df[df['Local Code'] == code].copy()

            org_len = stock_fin_tmp_org.shape[0]

            #Feature V4
            stock_fin_tmp = get_yoso_changerate(stock_fin_tmp)
            stock_fin_tmp = get_pred_actual_rate(stock_fin_tmp)

            stock_fin_tmp['�����\�z���㍂�ύX��']     = stock_fin_tmp['�����\�z���㍂�ύX��'].astype(float)
            stock_fin_tmp['�����\�z�c�Ɨ��v�ύX��']   = stock_fin_tmp['�����\�z�c�Ɨ��v�ύX��'].astype(float)
            stock_fin_tmp['�����\�z�o�험�v�ύX��']   = stock_fin_tmp['�����\�z�o�험�v�ύX��'].astype(float)
            stock_fin_tmp['�����\�z���������v�ύX��'] = stock_fin_tmp['�����\�z���������v�ύX��'].astype(float)

            stock_fin_tmp['�O�\�z���㍂']     = stock_fin_tmp['�O�\�z���㍂'].astype(float)
            stock_fin_tmp['�O�\�z�c�Ɨ��v']   = stock_fin_tmp['�O�\�z�c�Ɨ��v'].astype(float)
            stock_fin_tmp['�O�\�z�o�험�v']   = stock_fin_tmp['�O�\�z�o�험�v'].astype(float)
            stock_fin_tmp['�O�\�z���������v'] = stock_fin_tmp['�O�\�z���������v'].astype(float)

            stock_fin_tmp["���������v�\���є�"] = stock_fin_tmp.apply(lambda row: get_ratio(row['�O�\�z���������v'], row['���������v']), axis=1).astype(float)
            stock_fin_tmp["�o�험�v�\���є�"]   = stock_fin_tmp.apply(lambda row: get_ratio(row['�O�\�z�o�험�v'], row['�o�험�v']), axis=1).astype(float)
            stock_fin_tmp["�c�Ɨ��v�\���є�"]   = stock_fin_tmp.apply(lambda row: get_ratio(row['�O�\�z�c�Ɨ��v'], row['�c�Ɨ��v']), axis=1).astype(float)
            stock_fin_tmp["���㍂�\���є�"]     = stock_fin_tmp.apply(lambda row: get_ratio(row['�O�\�z���㍂'], row['���㍂']), axis=1).astype(float)


            # ���э�����������
            stock_fin_tmp_nan = stock_fin_tmp[stock_fin_tmp['�X�V��'].isna()]
            stock_fin_tmp = stock_fin_tmp[~stock_fin_tmp['�X�V��'].isna()]
            stock_fin_tmp['���Z���'] = stock_fin_tmp['���Z���'].replace(['Annual'], 'Q4')

            stock_fin_tmp_back = stock_fin_tmp.copy()
            stock_fin_tmp.drop_duplicates(subset=["���Z��",'���Z���',"�X�V��"],keep="first", inplace=True)

            quarterly_df = get_fin_quarterly_df(stock_fin_tmp)
            stock_fin_tmp_n = pd.concat([stock_fin_tmp,quarterly_df ], axis = 1)

            # Feature V2
            stock_fin_tmp_n["���������Y�L��"] = stock_fin_tmp_n['�����Y'].diff() / stock_fin_tmp_n['�����Y'].abs().shift()
            stock_fin_tmp_n["���������Y�L��"] = stock_fin_tmp_n['�����Y'].diff() / stock_fin_tmp_n['�����Y'].abs().shift()
            stock_fin_tmp_n["�����c�Ɨ��v��"] = stock_fin_tmp_n['�����c�Ɨ��v'] / stock_fin_tmp_n['�������㍂']
            stock_fin_tmp_n["���������v��"] = stock_fin_tmp_n['�������������v'] / stock_fin_tmp_n['�����Y']

            stock_fin_tmp_n['���������Y�L��'] = stock_fin_tmp_n['���������Y�L��'].astype(float)
            stock_fin_tmp_n['���������Y�L��'] = stock_fin_tmp_n['���������Y�L��'].astype(float)
            stock_fin_tmp_n["�����c�Ɨ��v��"] = stock_fin_tmp_n['�����c�Ɨ��v'].astype(float)
            stock_fin_tmp_n["���������v��"] = stock_fin_tmp_n['���������v��'].astype(float)

            stock_fin_tmp_back_n = pd.merge(stock_fin_tmp_back, stock_fin_tmp_n[[
                '�������㍂', '�����c�Ɨ��v', '�����o�험�v', '�������������v',
            '�������㍂�L��','�����c�Ɨ��v�L��', '�����o�험�v�L��',  '�������������v�L��',
            '���������Y�L��','���������Y�L��',"�����c�Ɨ��v��", "���������v��",
             "���Z��",'���Z���',"�X�V��"]], on=["���Z��",'���Z���',  "�X�V��"], how = 'outer')

            # Feature V2 over
            stock_fin_tmp_back_n.index = stock_fin_tmp_back.index

            tmp_df = pd.concat([stock_fin_tmp_nan, stock_fin_tmp_back_n], sort=True)
            tmp_df.sort_index(inplace = True)

            # Nan �̓��e������

            stock_fin_tmp_nan = tmp_df[tmp_df['�X�V��'].isna()]
            stock_fin_tmp = tmp_df[~tmp_df['�X�V��'].isna()]
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
        # stock_price�f�[�^��ǂݍ���
    #     price = dfs["stock_price"]
        price = df_org.copy()
        # ����̖����R�[�h�̃f�[�^�ɍi��
        price_data = price[price["Local Code"] == code]
        # �I�l�݂̂ɍi��
        feats = price_data[["EndOfDayQuote ExchangeOfficialClose"]]

        # �����ʂ̐����Ώۊ��Ԃ��w��
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
        # �I�l��20�c�Ɠ����^�[��
        feats["return_1month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(20)
        # �I�l��40�c�Ɠ����^�[��
        feats["return_2month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(40)
        # �I�l��60�c�Ɠ����^�[��
        feats["return_3month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(60)
        # �I�l��20�c�Ɠ��{���e�B���e�B
        feats["volatility_1month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(20)
            .std()
        )
        # �I�l��40�c�Ɠ��{���e�B���e�B
        feats["volatility_2month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(40)
            .std()
        )
        # �I�l��60�c�Ɠ��{���e�B���e�B
        feats["volatility_3month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(60)
            .std()
        )
        # �I�l��20�c�Ɠ��̒P���ړ����ϐ��̘���
        feats["MA_gap_1month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(20).mean()
        )
        # �I�l��40�c�Ɠ��̒P���ړ����ϐ��̘���
        feats["MA_gap_2month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(40).mean()
        )
        # �I�l��60�c�Ɠ��̒P���ړ����ϐ��̘���
        feats["MA_gap_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(60).mean()
        )
        # �����l����
        feats = feats.fillna(0)
        # ���f�[�^�̃J�������폜
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

        # �f�[�^�ǂݍ���
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # �\���Ώۂ̖����R�[�h�ƖړI�ϐ���ݒ�
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # �����ʂ��쐬
        feats  = cls.get_features_for_predict(cls.dfs, codes, start_dt)

        # ���ʂ��ȉ���csv�`���ŏo�͂���
        # �P���:datetime��code���Ȃ�������(Ex 2016-05-09-1301)
        # �Q���:label_high_20�@�I�l���ō��l�ւ̕ω���
        # �R���:label_low_20�@�I�l���ň��l�ւ̕ω���
        # header�͂Ȃ��AB��C���float64

        # ���t�Ɩ����R�[�h�ɍi�荞��
        df = feats.loc[:, ["Local Code"]].copy()
        # code���o�͌`���̂P��ڂƈ�v������
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "Local Code"].astype(
            str
        )

        # �o�͑Ώۗ���`
        output_columns = ["code"]

        # �����ʃJ�������w��
        # feature_columns = cls.get_feature_columns(feats)

        # �ړI�ϐ����ɗ\��
        for label in labels:
            # �\�����{
    #         df[label] = cls.models[label].predict(xgb.DMatrix(feats[feature_columns]))
            feats = feats[cls.models_fname[label]]
            # df[label] = cls.models[label].predict(xgb.DMatrix(feats))
            df[label] = cls.models[label].predict(feats)
            # �o�͑Ώۗ�ɒǉ�
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
        # ���f���ۑ���f�B���N�g�����쐬
    #     os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"model_{label}.pkl"), "wb") as f:
            # ���f����pickle�`���ŕۑ�
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
        # �����f�[�^�p�̕ϐ����`
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # �����R�[�h���ɓ����ʂ��쐬
        for code in tqdm(codes):
            # �����ʎ擾
            feats = feature[feature["Local Code"] == code]
    #         print (feats.index)

            # stock_label�f�[�^��ǂݍ���
            stock_labels = stock_labels_df
            # ����̖����R�[�h�̃f�[�^�ɍi��
            stock_labels = stock_labels[stock_labels["Local Code"] == code]

            # ����̖ړI�ϐ��ɍi��
            labels = stock_labels[label].copy()
            # nan���폜
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # �����ʂƖړI�ϐ��̃C���f�b�N�X�����킹��
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # �f�[�^�𕪊�
                _train_X = feats
    #             _val_X = feats[VAL_START : VAL_END]
    #             _test_X = feats[TEST_START :]

                _train_y = labels
    #             _val_y = labels[VAL_START : VAL_END]
    #             _test_y = labels[TEST_START :]

                # �f�[�^��z��Ɋi�[ (��قǌ������邽��)
                trains_X.append(_train_X)
    #             vals_X.append(_val_X)
    #             tests_X.append(_test_X)

                trains_y.append(_train_y)
    #             vals_y.append(_val_y)
    #             tests_y.append(_test_y)
        # �������ɍ쐬���������ϐ��f�[�^���������܂��B
        train_X = pd.concat(trains_X)
    #     val_X = pd.concat(vals_X)
    #     test_X = pd.concat(tests_X)
        # �������ɍ쐬�����ړI�ϐ��f�[�^���������܂��B
        train_y = pd.concat(trains_y)
    #     val_y = pd.concat(vals_y)
    #     test_y = pd.concat(tests_y)

        return train_X, train_y

    @classmethod
    def train_lgb_release_v2(cls, X_train, Y_train, ishigh = True):
        def lgb_srcc(preds, dtrain):
            # �������x��
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

    #     not_use_list = ['Local Code', '���Z�N�x','EndOfDayQuote ExchangeOfficialClose',]
        not_use_list = ['Local Code', '���Z�N�x','EndOfDayQuote ExchangeOfficialClose',
                       '�����\�z���Z�N�x', 'IssuedShareEquityQuote_IssuedShare',
                        '�z�����Z�N�x',
                       '�ꊔ�l�����z����','�ꊔ�N�Ԕz�����݌v', '�\�z�z�����Z�N�x', '�\�z�ꊔ�l�����z����', '�\�z�ꊔ�N�Ԕz�����݌v',
                       ]

        res_list = list(set(feats.columns.tolist()).difference(set(not_use_list)))

        return res_list


    @classmethod
    def train( cls, inputs, start_dt="2016-01-01"):
        # �f�[�^�ǂݍ���
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        codes = cls.codes
        # �����ʂ��쐬
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



