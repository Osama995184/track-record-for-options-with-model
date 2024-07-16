#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from IPython.display import display
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# In[47]:


def buy_operation(cash, current_price):
    try:
        n_shares = (cash // current_price)
    except ZeroDivisionError:
        # Handle the case where current_price is zero
        return 0, cash, 0
    n_shares_adjusted = n_shares - (n_shares % 100)  # Adjust n_shares to be divisible by 100
    traded_money = n_shares_adjusted * current_price
    rest_money = cash - traded_money
    return traded_money, rest_money, n_shares_adjusted

def designPortfolio(resulted_df,company_name,types):
    if resulted_df.empty:
        print('The DataFrame is empty')
        return pd.DataFrame()
    df = resulted_df.copy() 
    Current_point = list(df['Option_price'])
    predicted_future_point = list(df[f'predicted_future_point'])
    Actual_future_point = list(df['future_option_one_day'])
    Stock_price = list(df['Stock_price'])
    Strike = list(df['Strike_Price'])
    option_type = list(df['Option_type'])
    
    model_decision = []
    resulted_op_money =[]
    resulted_op_shares = []
    rest_op_money = []
    N_shares = []
    model_corrector = []
    sp_portfolio = []
    commission = []

    teade_yet = False
    starting = True
    starting_cashs = 50000
    
    wrong_decision = 0
    right_decision = 0
    
    for c, p, a, s, o in zip(Current_point, predicted_future_point, Actual_future_point, Strike, option_type):

        
        # Handling --------------- Stock movenemt----------------
        if starting :

            starting_cash, _, starting_shares = buy_operation(starting_cashs, c)
#             sp_portfolio.append(starting_shares*c)
            starting = False
        else:
#             sp_portfolio.append(starting_shares*c)
            starting = False
        # Checked .........True

        # Handling --------------- Model Decision----------------- 
        if teade_yet == True : # you already traded before
            last_decision = model_decision[-1]
            
            if ((p > c)):
                if last_decision == f'BUY_{o}' or last_decision == f'HOLD_{o}': # you already have the S&P

                    model_decision.append(f'HOLD_{o}')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_op_money[-1]

                    commission.append(0)
                    resulted_op_money.append(prev_shares*c+prev_rest)
                    resulted_op_shares.append(prev_shares*c)
                    rest_op_money.append(prev_rest)
                    N_shares.append(prev_shares)
                    sp_portfolio.append(prev_shares*c)


                else: # last_decision, 'SELL', 'SKIP'
                    model_decision.append(f'BUY_{o}')

                    # Buying your previous resulted op money

                    prev_resulted_op_money = resulted_op_money[-1]
                    traded_money, rest_money, n_shares = buy_operation(prev_resulted_op_money, c)

                    resulted_op_money.append(n_shares*c+rest_money)
                    resulted_op_shares.append(n_shares*c)
                    rest_op_money.append(rest_money)
                    N_shares.append(n_shares)
                    sp_portfolio.append(n_shares*c)
                    if ((n_shares*0.01)<3.5):
                        x = 3.5
                    else:
                        x = n_shares*0.01
                    commission.append(x)


            elif p < c: # in case of equality 
                if last_decision == f'SELL_{o}' or last_decision == f'SKIP_{o}': # you already have the S&P
                    model_decision.append(f'SKIP_{o}')

                    # everething Stay as they was
                    commission.append(0)
                    resulted_op_money.append(resulted_op_money[-1])
                    resulted_op_shares.append(resulted_op_shares[-1])
                    rest_op_money.append(rest_op_money[-1])
                    N_shares.append(N_shares[-1])
                    sp_portfolio.append(sp_portfolio[-1])

                else: # last_decision, 'BUY', 'HOLD'
                    model_decision.append(f'SELL_{o}')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_op_money[-1]
                    resulted_op_money.append(prev_shares*c+prev_rest)
                    resulted_op_shares.append(prev_shares*c)
                    rest_op_money.append(prev_rest)
                    N_shares.append(0)
                    sp_portfolio.append(prev_shares*c)
                    if ((n_shares*0.01)<3.5):
                        x = 3.5
                    else:
                        x = n_shares*0.01
                    commission.append(x)
            
            elif p == c:
                if last_decision == f'SELL_{o}' or last_decision == f'SKIP_{o}': # you already have the S&P
                    model_decision.append(f'SKIP_{o}')

                    # everething Stay as they was
                    commission.append(0)
                    resulted_op_money.append(resulted_op_money[-1])
                    resulted_op_shares.append(resulted_op_shares[-1])
                    rest_op_money.append(rest_op_money[-1])
                    N_shares.append(N_shares[-1])
                    sp_portfolio.append(sp_portfolio[-1])
                    
                else: # last_decision, 'BUY', 'HOLD'
                    model_decision.append(f'HOLD_{o}')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_op_money[-1]

                    commission.append(0)
                    resulted_op_money.append(prev_shares*a+prev_rest)
                    resulted_op_shares.append(prev_shares*a)
                    rest_op_money.append(prev_rest)
                    N_shares.append(prev_shares)
                    sp_portfolio.append(prev_shares*c)
                    
                    
                

        else: # 1st time to trade
            if ((p > c)):
                teade_yet = True
                model_decision.append(f'BUY_{o}')
                # executing by operation by starting cash 
                traded_money, rest_money, n_shares = buy_operation(starting_cashs, c)

                resulted_op_money.append(n_shares*c+rest_money)
                resulted_op_shares.append(n_shares*c)
                rest_op_money.append(rest_money)
                N_shares.append(n_shares)
                sp_portfolio.append(n_shares*c)
                if ((n_shares*0.01)<3.5):
                    x = 3.5
                else:
                    x = n_shares*0.01
                commission.append(x)

            else:
                model_decision.append(f'SKIP_{o}')
                resulted_op_money.append(starting_cashs)
                commission.append(0)
                resulted_op_shares.append(0)
                rest_op_money.append(0)
                N_shares.append(0)
                sp_portfolio.append(starting_cashs)

                #Rest_money.append(starting_cash)
                #portfolio_money.append(0)

        # Handle -------------------------------Model Corrector--------------------
        last_decision = model_decision[-1]
        if (p > c and a > c) or (p<c and a<c) or (p == c and a >= c) : 
            model_corrector.append('Right--'+last_decision)
            right_decision += 1
        else:
            model_corrector.append('Wrong--'+last_decision)
            wrong_decision += 1
            



    data = {
            'Date':df['Date'],
            'Day': pd.to_datetime(df['Date']).dt.day_name(),
            'C_point':Current_point, 
            'PN_point': predicted_future_point, 
            'AN_point': Actual_future_point,
            'Strike': Strike,
            'options': option_type,
            'Model-Decision': model_decision,
            'Model-Corrector':model_corrector ,
            'resulted_op_shares':resulted_op_shares,
            'rest_op_money':rest_op_money,
            'resulted_op_money': np.round(resulted_op_money,2),
            'N_shares':N_shares,
            'commission': commission,
            'portfolio':sp_portfolio
            }
    
    porto_df = pd.DataFrame(data)
#     com = round(commission,3)
    commissions = round(sum(commission), 3)
    m_return = ((Stock_price[-1]-Stock_price[0])/Stock_price[0])*100
    print(f'Stock_Return: {round(m_return,3)} %')
    sp_return = (((sp_portfolio[-1]+rest_op_money[-1])-starting_cashs)-commissions)/starting_cashs*100
    print(f'portfolio_Return: {round(sp_return,3)} %')
    rw_ratio = (right_decision)/(right_decision+wrong_decision)*100
    print(f'R/W Ratio: {round(rw_ratio,3)} %')
    mae_test = mean_absolute_error(Actual_future_point, predicted_future_point)
    print(f'MAE : ' , mae_test , 'dollar')
    metrics_df = pd.DataFrame({
        'portfolio_Return': [round(sp_return, 3)],
        'S&P return':[round(m_return,3)],
        'R/W_Ratio': [round(rw_ratio, 3)],
        'commission': commissions,
        'symbol': company_name,
        'type': types
    })
    return porto_df,metrics_df


# In[48]:


def draw_actual_vs_predict(date, actual, predicted, str_, marker):
    # Check the lengths of the input lists
    len_date = len(date)
    len_actual = len(actual)
    len_predicted = len(predicted)

    print(f"Length of date list: {len_date}")
    print(f"Length of actual list: {len_actual}")
    print(f"Length of predicted list: {len_predicted}")

    if len_date != len_actual or len_date != len_predicted:
        print("Error: The lengths of the input lists do not match.")
        return

    # Create the DataFrame
    df = pd.DataFrame({
        'Date': date,
        'actual': actual,
        'predicted': predicted
    })

    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Plot the data
    fig = px.line(df, x='Date', y=['actual', 'predicted'], markers=marker)

    fig.update_layout(
        title_text=str_,
        plot_bgcolor='white',
        font_size=15,
        font_color='black',
        legend_title_text=''
    )

    fig.update_xaxes(title_text="Date", zeroline=False, showgrid=False)
    fig.update_yaxes(title_text='actual', secondary_y=False, zeroline=False, showgrid=False)
    fig.update_yaxes(title_text='predicted', secondary_y=True, zeroline=False, showgrid=False)

    fig.show()


# In[49]:


def process_and_sort_data(df_put_OUT):
    if df_put_OUT.empty:
        print(f'The DataFrame is empty.')
        return df_put_OUT
    
    random_index = np.random.choice(df_put_OUT.index)
    df_call1_filtered = df_put_OUT.loc[[random_index]]
    strike = df_call1_filtered['Strike_Price'].iloc[0]
    condition2 = df_put_OUT['Strike_Price'] == strike
    df_put_OUT = df_put_OUT.drop(df_put_OUT[condition2].index)

    return df_put_OUT

def test_portfolio(df, company_name):
    df['Symbol'] = company_name
    df['difference'] = df['future_option_one_day'] - df['Option_price']
    df.replace({'Option_type': {0: 'call', 1: 'put'}}, inplace=True)
    
    Date_counts = df['Strike_Price'].value_counts().to_frame()
    Date_counts.rename(columns={'Strike_Price': 'value_counts'}, inplace=True)
    Date_counts.index.name = 'Strike_Price'
    Date_counts_sorted = Date_counts.sort_index()
    strike_prices = Date_counts_sorted.index.tolist()
    
    df_call = df[df['Option_type'] == "call"]
#     df_put = df[df['Option_type'] == "put"]
    
    df_call_IN = df_call[(df_call['Strike_Price'] == strike_prices[0]) | (df_call['Strike_Price'] == strike_prices[1])]
    df_call_OUT = df_call[(df_call['Strike_Price'] == strike_prices[4]) | (df_call['Strike_Price'] == strike_prices[5])]
    df_call_NEAR = df_call[(df_call['Strike_Price'] == strike_prices[2]) | (df_call['Strike_Price'] == strike_prices[3])]
    
    final_df_call_OUT = process_and_sort_data(df_call_OUT)
    final_df_call_IN = process_and_sort_data(df_call_IN)
    final_df_call_NEAR = process_and_sort_data(df_call_NEAR)
    
    print("Final DataFrame (Call OUT):")
    final_df_call_OUT, metrics_call_OUT = designPortfolio(final_df_call_OUT,company_name,'call_OUT')
    print('________________________________________________________________________________________________________________')
#     display(final_df_call_OUT)
    print("Final DataFrame (Call IN):")
    final_df_call_IN, metrics_call_IN = designPortfolio(final_df_call_IN,company_name,'call_IN')
    
    print('________________________________________________________________________________________________________________')
#     display(final_df_call_IN)
    print("Final DataFrame (Call NEAR):")
    final_df_call_NEAR, metrics_call_NEAR = designPortfolio(final_df_call_NEAR,company_name,'call_NEAR')
    print('________________________________________________________________________________________________________________')
#     display(final_df_call_NEAR)
    
    final_df_test = pd.concat([final_df_call_OUT, final_df_call_IN, final_df_call_NEAR], axis=0)
    final_metrics = pd.concat([metrics_call_OUT, metrics_call_IN, metrics_call_NEAR], axis=0)
    return final_df_test,final_metrics


# In[50]:


def get_window_series(start_date, bucket_size, df):
    date_obj = datetime.strptime(start_date, '%m/%d/%Y')
    final_date = pd.Timestamp(date_obj)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    comparison_result = df['Date'].iloc[0] <= final_date
    if (comparison_result):
        df_test = df[df['Date'] >= final_date]
        date_list =df_test['Date']
    else:
        date_list = df[-512*12:]['Date']
    n_trading_days = date_list.count()
    n_trading_days = n_trading_days
    bucket_size = bucket_size*12
    num_buckets = int(n_trading_days / bucket_size)

    df = df.copy()
    total_rows = len(df)

    # Step 3: Create the buckets
    buckets = []
    for i in range(num_buckets, 0, -1):
        train_data = df.iloc[0:-i*bucket_size]
        if i > 1:
            valid_data = df.iloc[-i*bucket_size: (-i*bucket_size)+bucket_size]
        else:
            valid_data = df.iloc[-i*bucket_size:]

        buckets.append((train_data, valid_data))
    
    return buckets, date_list


# In[51]:


def my_standard_model_exp2_multiperiod(df, n_trading_days, included_features, buckets, date_list, current_point, buying_point, target, rf_model,company_name):
    print('Prediction From ', date_list.iloc[0], ' To ', date_list.iloc[-1])
    valid_prediction_list = []
    valid_y_list = []
    train_prediction_list = []
    train_y_list = []
    i = 0
    models_list = []

    for bucket in buckets:
        i += 1
        train, valid = bucket
        start_date = valid['Date'].iloc[0]
        end_date = valid['Date'].iloc[-1]
        X_train, y_train = train[included_features], train[target]
        X_test, y_test = valid[included_features], valid[target]

        rf_model.fit(X_train, y_train)
        models_list.append(rf_model)

        train_predictions = rf_model.predict(X_train)
        test_predictions = rf_model.predict(X_test)

        # Calculate error
        mae_test = mean_absolute_error(y_test, test_predictions)
        r2_test = r2_score(y_test, test_predictions)
        print('Bucket ', i, " Test-- MAE: ", mae_test, 'R2 Score: ', r2_test)
        print("Start date: ", start_date, 'End date: ', end_date)

        train_prediction_list.append(list(train_predictions))
        train_y_list.append(list(y_train))
        valid_prediction_list.append(list(test_predictions))
        valid_y_list.append(list(y_test))

    valid_y_list = [item for sublist in valid_y_list for item in sublist]
    valid_prediction_list = [item for sublist in valid_prediction_list for item in sublist]
    train_y_list = [item for sublist in train_y_list for item in sublist]
    train_prediction_list = [item for sublist in train_prediction_list for item in sublist]

    mae_train = mean_absolute_error(train_y_list, train_prediction_list)
    r2_train = r2_score(train_y_list, train_prediction_list)
    mae_test = mean_absolute_error(valid_y_list, valid_prediction_list)
    r2_test = r2_score(valid_y_list, valid_prediction_list)
    print("Total-Test-- MAE: ", mae_test, 'R2 Score: ', r2_test)

    # Truncate date_list to match the length of valid_y_list and valid_prediction_list
    if len(date_list) > len(valid_y_list):
        date_list = date_list[:len(valid_y_list)]

    draw_actual_vs_predict(date_list, valid_y_list, valid_prediction_list, 'Model performance in Testing', True)

    test_data = df[-n_trading_days:]

    # Truncate test_data to match the length of valid_prediction_list
    min_len = min(len(test_data), len(valid_prediction_list))
    test_data = test_data.iloc[:min_len]
    valid_prediction_list = valid_prediction_list[:min_len]
    valid_y_list = valid_y_list[:min_len]

    result_df = pd.DataFrame(columns=['Date', 'Option_price', 'predicted_future_point', 'future_option_one_day',
                                     'Stock_price','Strike_Price','Option_type'])
    result_df['Date'] = test_data['Date']
    result_df['Option_price'] = test_data[current_point]
    result_df['future_option_one_day'] = test_data[target]
    result_df['predicted_future_point'] = valid_prediction_list
    result_df['Stock_price'] = test_data['Stock_price']
    result_df['Strike_Price'] = test_data['Strike_Price']
    result_df['Option_type'] = test_data['Option_type']

    n = len(buckets)
    rows_per_part = len(result_df) // n
    df_parts = [result_df.iloc[i * rows_per_part: (i + 1) * rows_per_part] for i in range(n)]
    list_of_portfolio = []
    list_of_portfolio.append(0)
    list_of_matric = []

    for j, df_part in enumerate(df_parts, start=1):
        print('Bucket ', j)
        start_date = df_part['Date'].iloc[0].date()
        end_date = df_part['Date'].iloc[-1].date()
        portfolio_df,final_metrics = test_portfolio(df_part,company_name)
        final_metrics = final_metrics.assign(Start_Date=start_date)
        final_metrics = final_metrics.assign(End_Date=end_date)
        list_of_portfolio.append(portfolio_df)
        list_of_matric.append(final_metrics)
        print('------------------------------------------------------------')

    print('----Total Portfolio----')
    start_date2 = result_df['Date'].iloc[0].date()
#     print(start_date2)
    end_date2 = result_df['Date'].iloc[-1].date()
#     print(end_date2)
    total_portfolio_df,total_final_metrics = test_portfolio(result_df,company_name)
    total_final_metrics = total_final_metrics.assign(Start_Date=start_date2)
    total_final_metrics = total_final_metrics.assign(End_Date=end_date2)
    list_of_portfolio.append(total_portfolio_df)
    list_of_matric.append(total_final_metrics)

    return models_list, result_df, list_of_portfolio,list_of_matric


# In[52]:


features = ["Strike_Price", "Stock_price",'Option_price', "Rate",
                      "Rolling_Std",'Option_type', "implied_volatility",
                      "Vega", "delta", "gamma", 'sharpe_ratio']


# In[53]:


ML_algo_dict = {
     'AAPL': RandomForestRegressor(),
     'ADBE': Ridge(alpha=0.8),
     'AEYE': RandomForestRegressor(),
     'AMD': DecisionTreeRegressor(),
     'AMZN': DecisionTreeRegressor(),
     'ANET': RandomForestRegressor(),
     'ARKG': Ridge(alpha=0.8),
     'ARKK_ETF': Lasso(alpha=0.1),
     'ASML': RandomForestRegressor(),
     'BILL': RandomForestRegressor(),
     'CELH': RandomForestRegressor(),
     'CMG': RandomForestRegressor(),
     'COIN': DecisionTreeRegressor(),
     'COST': RandomForestRegressor(),
     'CRM': RandomForestRegressor(),
     'CRWD': Lasso(alpha=0.1),
     'CYBR': Lasso(alpha=0.1),
     'DDOG': Lasso(alpha=0.1),
     'DKNG': Ridge(alpha=0.8),
     'DT': RandomForestRegressor(),
     'ELF': Lasso(alpha=0.1),
     'FTI': Lasso(alpha=0.1),
     'FTNT': Lasso(alpha=0.1),
     'GOOGL': Ridge(alpha=0.8),
     'GTEK_ETF': RandomForestRegressor(),
     'HUBS': RandomForestRegressor(),
     'INTC': Lasso(alpha=0.1),
     'KLAC': RandomForestRegressor(),
     'LCID': Ridge(alpha=0.8),
     'LLY': RandomForestRegressor(),
     'LPLA': DecisionTreeRegressor(),
     'MA': DecisionTreeRegressor(),
     'MELI': LinearRegression(),
     'META': RandomForestRegressor(),
     'MLTX': Lasso(alpha=0.01),
     'MRVL': RandomForestRegressor(),
     'MSFT': LinearRegression(),
     'MSI': Lasso(alpha=0.01),
     'NIO': DecisionTreeRegressor(),
     'NVDA': DecisionTreeRegressor(),
     'ORCL': DecisionTreeRegressor(),
     'OXY': RandomForestRegressor(),
     'PANW': RandomForestRegressor(),
     'PATH': LinearRegression(),
     'RBLX': Lasso(alpha=0.1),
     'RIVN': Lasso(alpha=0.1),
     'ROIV': RandomForestRegressor(),
     'ROKU': LinearRegression(),
     'SMCI': RandomForestRegressor(),
     'SMH_ETF': LinearRegression(),
     'SOUN': DecisionTreeRegressor(),
     'SPCE': Lasso(alpha=0.1),
     'SQ': RandomForestRegressor(),
     'SYM': Lasso(alpha=0.1),
     'TEAM': DecisionTreeRegressor(),##
     'TSLA': RandomForestRegressor(),
     'TSM': RandomForestRegressor(),
     'TWLO': RandomForestRegressor(),
     'U': Lasso(alpha=0.1),
     'UBER': Lasso(alpha=0.1),
     'UNH': RandomForestRegressor(),
     'V': DecisionTreeRegressor(),
     'VKTX': Lasso(alpha=0.1),
     'VRT':DecisionTreeRegressor(),
     'WDAY': LinearRegression(),
     'XLE_ETF': RandomForestRegressor(),
     'XLF_ETF': LinearRegression(),
     'ZM': DecisionTreeRegressor()
}
# sorted_ML_algo_dict1 = {k: ML_algo_dict[k] for k in sorted(ML_algo_dict)}


# In[54]:


def track_record_10y(company):
    print('________________________________________________________________________________________________________________')
    print(company)
    df = pd.read_csv(f'D:/Quantum/Codes/Option_data_from_historical/options_data/companies_train/{company}_final_data_for_options.csv')
    df.replace({'Option_type':{'call':0,'put':1,'Call':0,'Put':1}},inplace=True)
    start_date = '01/01/2014'
    bucket_size = 60
    buckets, date_list = get_window_series(start_date, bucket_size, df)
    n_trading_days = date_list.count()
    models_list, result_df, list_of_portfolio,total_final_metrics = my_standard_model_exp2_multiperiod(
    df, 
    n_trading_days, 
    features,
    buckets, 
    date_list, 
    'Option_price', 
    'Option_price', 
    'future_option_one_day', 
    ML_algo_dict[company],
    company)
    
    combined_df = pd.concat(total_final_metrics, ignore_index=True)
    combined_df.to_csv(f"D:/Quantum/Codes/Option_data_from_historical/options_data/buckets/track_record_10y_{company}.csv", index=False)
    print('DONE')
    return combined_df
    


# In[55]:


# def best_models(company_name):
#     models = [{'name': RandomForestRegressor()},
#               {'name': DecisionTreeRegressor()},
#               {'name': LinearRegression()},
#               {'name': Ridge(alpha=0.8)},
#               {'name': Lasso(alpha=0.1)}]
    
#     model_results = {}
#     final_dfs = {}
    
#     for model in models:
#         model_name = model['name']
#         df = track_record_10y(company_name,model_name)
# #         df = df.iloc[3:].reset_index(drop=True)
# #         display(df)
#         model_return = df['portfolio_Return'].sum()
#         model_results[model_name] = model_return
#         final_dfs[model_name] = df
    
#     best_model = max(model_results, key=model_results.get)
#     final_df = final_dfs[best_model]
    
#     return final_df,best_model


# In[56]:


track_record_10y('AAPL')


# In[57]:


track_record_10y('ADBE')


# In[58]:


track_record_10y('AEYE')


# In[59]:


track_record_10y('AMD')


# In[60]:


track_record_10y('AMZN')


# In[61]:


track_record_10y('ANET')


# In[62]:


track_record_10y('ARKG')


# In[63]:


track_record_10y('ARKK_ETF')


# In[64]:


track_record_10y('ASML')


# In[65]:


track_record_10y('BILL')


# In[66]:


track_record_10y('CELH')


# In[67]:


track_record_10y('CMG')


# In[68]:


track_record_10y('COIN')


# In[69]:


track_record_10y('COST')


# In[70]:


track_record_10y('CRM')


# In[71]:


track_record_10y('CRWD')


# In[72]:


track_record_10y('CYBR')


# In[73]:


track_record_10y('DDOG')


# In[74]:


track_record_10y('DKNG')


# In[75]:


track_record_10y('DT')


# In[76]:


track_record_10y('ELF')


# In[77]:


track_record_10y('FTI')


# In[78]:


track_record_10y('FTNT')


# In[79]:


track_record_10y('GOOGL')


# In[80]:


track_record_10y('GTEK_ETF')


# In[81]:


track_record_10y('HUBS')


# In[82]:


track_record_10y('INTC')


# In[83]:


track_record_10y('KLAC')


# In[84]:


track_record_10y('LCID')


# In[85]:


track_record_10y('LLY')


# In[86]:


track_record_10y('LPLA')


# In[87]:


track_record_10y('MA')


# In[88]:


track_record_10y('MELI')


# In[89]:


track_record_10y('META')


# In[90]:


track_record_10y('MLTX')


# In[91]:


track_record_10y('MRVL')


# In[92]:


track_record_10y('MSFT')


# In[93]:


track_record_10y('MSI')


# In[94]:


track_record_10y('NIO')


# In[95]:


track_record_10y('NVDA')


# In[96]:


track_record_10y('ORCL')


# In[97]:


track_record_10y('OXY')


# In[98]:


track_record_10y('PANW')


# In[99]:


track_record_10y('PATH')


# In[100]:


track_record_10y('RBLX')


# In[101]:


track_record_10y('RIVN')


# In[102]:


track_record_10y('ROIV')


# In[103]:


track_record_10y('ROKU')


# In[104]:


track_record_10y('SMCI')


# In[105]:


track_record_10y('SOUN')


# In[106]:


track_record_10y('SPCE')


# In[107]:


track_record_10y('SMH_ETF')


# In[108]:


track_record_10y('SQ')


# In[109]:


track_record_10y('SYM')


# In[110]:


track_record_10y('TEAM')


# In[111]:


track_record_10y('TSLA')


# In[112]:


track_record_10y('TSM')


# In[113]:


track_record_10y('TWLO')


# In[114]:


track_record_10y('U')


# In[115]:


track_record_10y('UBER')


# In[116]:


track_record_10y('UNH')


# In[117]:


track_record_10y('V')


# In[118]:


track_record_10y('VKTX')


# In[119]:


track_record_10y('VRT')


# In[120]:


track_record_10y('WDAY')


# In[121]:


track_record_10y('XLE_ETF')


# In[122]:


track_record_10y('XLF_ETF')


# In[123]:


track_record_10y('ZM')


# In[ ]:




