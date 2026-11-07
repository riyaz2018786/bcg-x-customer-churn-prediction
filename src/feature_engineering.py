# BCG X Data Science Job Simulation - Task 3: Feature Engineering
import pandas as pd
import numpy as np

# 1. Load cleaned client data after EDA and convert date columns
df = pd.read_csv('./clean_data_after_eda.csv')
for col in ["date_activ", "date_end", "date_modif_prod", "date_renewal"]:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')

# 2. Load historical pricing data
price_df = pd.read_csv('./price_data.csv')
price_df["price_date"] = pd.to_datetime(price_df["price_date"], format='%Y-%m-%d')

# 3. Baseline feature: Difference between off-peak prices in December and preceding January
monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({
    'price_off_peak_var': 'mean',
    'price_off_peak_fix': 'mean'
}).reset_index()

jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

diff = pd.merge(
    dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}),
    jan_prices.drop(columns='price_date'),
    on='id'
)
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
diff = diff[['id', 'offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']]
df = pd.merge(df, diff, on='id')

# 4. Expand Dec vs. Jan price differences to Peak and Mid-Peak periods
monthly_all_prices = price_df.groupby(['id', 'price_date']).agg({
    'price_peak_var': 'mean',
    'price_mid_peak_var': 'mean',
    'price_peak_fix': 'mean',
    'price_mid_peak_fix': 'mean'
}).reset_index()

jan_all = monthly_all_prices.groupby('id').first().reset_index()
dec_all = monthly_all_prices.groupby('id').last().reset_index()

diff_all = pd.merge(
    dec_all.rename(columns={
        'price_peak_var': 'dec_peak_var',
        'price_mid_peak_var': 'dec_mid_var',
        'price_peak_fix': 'dec_peak_fix',
        'price_mid_peak_fix': 'dec_mid_fix'
    }),
    jan_all.drop(columns='price_date'),
    on='id'
)
diff_all['peak_diff_dec_january_energy'] = diff_all['dec_peak_var'] - diff_all['price_peak_var']
diff_all['midpeak_diff_dec_january_energy'] = diff_all['dec_mid_var'] - diff_all['price_mid_peak_var']
diff_all['peak_diff_dec_january_power'] = diff_all['dec_peak_fix'] - diff_all['price_peak_fix']
diff_all['midpeak_diff_dec_january_power'] = diff_all['dec_mid_fix'] - diff_all['price_mid_peak_fix']
diff_all = diff_all[[
    'id', 'peak_diff_dec_january_energy', 'midpeak_diff_dec_january_energy',
    'peak_diff_dec_january_power', 'midpeak_diff_dec_january_power'
]]
df = pd.merge(df, diff_all, on='id')

# 5. Average price spreads across periods & Max-Min monthly price variations
mean_prices = price_df.groupby('id').agg({
    'price_off_peak_var': 'mean',
    'price_peak_var': 'mean',
    'price_mid_peak_var': 'mean',
    'price_off_peak_fix': 'mean',
    'price_peak_fix': 'mean',
    'price_mid_peak_fix': 'mean'
}).reset_index()

mean_prices['off_peak_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_peak_var']
mean_prices['peak_mid_peak_var_mean_diff'] = mean_prices['price_peak_var'] - mean_prices['price_mid_peak_var']
mean_prices['off_peak_mid_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_mid_peak_var']
mean_prices['off_peak_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_peak_fix']
mean_prices['peak_mid_peak_fix_mean_diff'] = mean_prices['price_peak_fix'] - mean_prices['price_mid_peak_fix']
mean_prices['off_peak_mid_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_mid_peak_fix']
df = pd.merge(df, mean_prices, on='id')

price_range = price_df.groupby('id').agg({
    'price_off_peak_var': lambda x: x.max() - x.min(),
    'price_peak_var': lambda x: x.max() - x.min(),
    'price_mid_peak_var': lambda x: x.max() - x.min(),
    'price_off_peak_fix': lambda x: x.max() - x.min(),
    'price_peak_fix': lambda x: x.max() - x.min(),
    'price_mid_peak_fix': lambda x: x.max() - x.min()
}).reset_index().rename(columns={
    'price_off_peak_var': 'max_diff_off_peak_var',
    'price_peak_var': 'max_diff_peak_var',
    'price_mid_peak_var': 'max_diff_mid_peak_var',
    'price_off_peak_fix': 'max_diff_off_peak_fix',
    'price_peak_fix': 'max_diff_peak_fix',
    'price_mid_peak_fix': 'max_diff_mid_peak_fix'
})
df = pd.merge(df, price_range, on='id')

# 6. Date & Tenure Features (relative to 2016-01-01)
ref_date = pd.to_datetime('2016-01-01')
df['months_activ'] = ((ref_date - df['date_activ']).dt.days / 30.4375).round().astype(int)
df['months_to_end'] = ((df['date_end'] - ref_date).dt.days / 30.4375).round().astype(int)
df['months_modif_prod'] = ((ref_date - df['date_modif_prod']).dt.days / 30.4375).round().astype(int)
df['months_renewal'] = ((ref_date - df['date_renewal']).dt.days / 30.4375).round().astype(int)

df['month_activ'] = df['date_activ'].dt.month
df['month_end'] = df['date_end'].dt.month
df['month_modif_prod'] = df['date_modif_prod'].dt.month
df['month_renewal'] = df['date_renewal'].dt.month

# 7. Behavioral Consumption Ratios
df['cons_last_month_ratio'] = df['cons_last_month'] / (df['cons_12m'] + 1)
df['forecast_cons_ratio'] = df['forecast_cons_12m'] / (df['cons_12m'] + 1)
df['total_energy_cons_12m'] = df['cons_12m'] + df['cons_gas_12m']
df['elec_share_of_total_cons'] = df['cons_12m'] / (df['total_energy_cons_12m'] + 1)

# 8. Log10 Transformation of Skewed Columns
skewed_cols = [
    'cons_12m', 'cons_gas_12m', 'cons_last_month', 'forecast_cons_12m',
    'forecast_cons_year', 'forecast_discount_energy', 'forecast_meter_rent_12m',
    'imp_cons', 'net_margin', 'pow_max', 'total_energy_cons_12m'
]
for col in skewed_cols:
    df[col] = np.log10(df[col].clip(lower=0) + 1)

# 9. Categorical Encoding & Dropping Redundant Columns
df['has_gas'] = df['has_gas'].replace({'t': 1, 'f': 0}).astype(int)
df = pd.get_dummies(df, columns=['channel_sales', 'origin_up'], prefix=['channel', 'origin'], dtype=int)

cols_to_drop = [
    'date_activ', 'date_end', 'date_modif_prod', 'date_renewal',
    'margin_net_pow_ele',
    'var_year_price_off_peak', 'var_year_price_peak', 'var_year_price_mid_peak',
    'var_6m_price_off_peak', 'var_6m_price_peak', 'var_6m_price_mid_peak'
]
df = df.drop(columns=cols_to_drop)
print("Final engineered dataset shape:", df.shape)
