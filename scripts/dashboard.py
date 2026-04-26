import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

# Page config
st.set_page_config(page_title="Marketing Campaign Dashboard", layout="wide")
st.title("Marketing Campaign Analysis Dashboard")
st.markdown("RND4IMPACT — Data Science Program")

# Load data
@st.cache_data
def load_data():
    engine = create_engine('postgresql://postgres:postgres123@127.0.0.1:5432/marketing_campaign')
    return pd.read_sql('SELECT * FROM marketing_campaign', engine)

df = load_data()

# Train model
@st.cache_resource
def train_model(df):
    features = ['Income', 'Age', 'Recency', 'MntTotal',
                'NumWebPurchases', 'NumCatalogPurchases',
                'NumStorePurchases', 'Customer_Days',
                'NumDealsPurchases', 'Kidhome', 'Teenhome',
                'MntWines', 'MntMeatProducts', 'MntGoldProds',
                'AcceptedCmpOverall']
    X = df[features]
    y = df['Response']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scale = len(y_train[y_train==0]) / len(y_train[y_train==1])
    model = XGBClassifier(n_estimators=200, max_depth=4,
                          learning_rate=0.05, scale_pos_weight=scale,
                          base_score=0.5, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model, X_test, features

model, X_test, features = train_model(df)

# Section 1 - Key Metrics
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Campaign Acceptance Rate", f"{df['Response'].mean()*100:.1f}%")
col3.metric("Avg Total Spend", f"${df['MntTotal'].mean():.0f}")
col4.metric("Avg Income", f"${df['Income'].mean():.0f}")

# Section 2 - Spend by Category
st.header("Average Spend by Product Category")
spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
fig1, ax1 = plt.subplots(figsize=(8, 4))
df[spend_cols].mean().sort_values(ascending=False).plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_xlabel("Product Category")
ax1.set_ylabel("Average Spend")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# Section 3 - SHAP Feature Importance
st.header("Feature Importance (SHAP)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
fig2, ax2 = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  feature_names=features, show=False)
st.pyplot(fig2)

# Section 4 - Customer Predictor
st.header("Campaign Response Predictor")
st.markdown("Adjust the sliders to profile a customer and predict their campaign response.")

col1, col2, col3 = st.columns(3)
with col1:
    income = st.slider("Income", 0, 120000, 50000, step=1000)
    age = st.slider("Age", 18, 80, 45)
    recency = st.slider("Recency (days since last purchase)", 0, 100, 30)
    mnt_total = st.slider("Total Spend", 0, 2500, 500, step=50)
    mnt_meat = st.slider("Meat Spend", 0, 1500, 150, step=10)
with col2:
    num_web = st.slider("Web Purchases", 0, 20, 4)
    num_catalog = st.slider("Catalog Purchases", 0, 20, 2)
    num_store = st.slider("Store Purchases", 0, 20, 5)
    customer_days = st.slider("Days as Customer", 0, 4000, 2000, step=50)
    mnt_gold = st.slider("Gold Spend", 0, 300, 50, step=5)
with col3:
    num_deals = st.slider("Deal Purchases", 0, 15, 2)
    kidhome = st.slider("Kids at Home", 0, 3, 0)
    teenhome = st.slider("Teens at Home", 0, 3, 0)
    mnt_wines = st.slider("Wine Spend", 0, 1500, 300, step=10)
    accepted_overall = st.slider("Previous Campaigns Accepted", 0, 5, 1)

input_data = pd.DataFrame([[income, age, recency, mnt_total, num_web,
                             num_catalog, num_store, customer_days,
                             num_deals, kidhome, teenhome, mnt_wines,
                             mnt_meat, mnt_gold, accepted_overall]],
                           columns=features)

prob = model.predict_proba(input_data)[0][1]
prediction = "Will Accept" if prob >= 0.5 else "Will Not Accept"
color = "green" if prob >= 0.5 else "red"

st.markdown(f"### Prediction: :{color}[{prediction}]")
st.metric("Acceptance Probability", f"{prob*100:.1f}%")