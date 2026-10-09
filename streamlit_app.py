"""
Activity: Analysing EC2 Instances — Streamlit version
Run with:  streamlit run ec2_lab_app.py
Keep "Amazon EC2 Instance Comparison.csv" in the same folder as this script.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EC2 Instance Analysis", layout="wide")
st.title("Analysing Amazon EC2 Instances")

COST_COLUMNS = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost",
]
FILE_PATH = "Amazon EC2 Instance Comparison.csv"


# ---------------------------------------------------------------------------
# Step 2: Load the dataset
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


raw = load_data(FILE_PATH)

# Work on a copy so the cached raw data is never modified
data = raw.copy()


# ---------------------------------------------------------------------------
# Step 3: Clean the cost columns ('$0.0116 hourly' -> 0.0116)
# ---------------------------------------------------------------------------
for column in COST_COLUMNS:
    data[column] = pd.to_numeric(
        data[column].astype(str).str.replace("[$, hourly]", "", regex=True),
        errors="coerce",
    )


def detect_outliers(df, column):
    """IQR method: anything beyond 1.5 * IQR from Q1/Q3."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


def filter_instance_family(df, family):
    return df[df["Name"].str.startswith(family, na=False)]


def cost_boxplot(df, title, palette):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[COST_COLUMNS], palette=palette, showmeans=True, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Cost (USD)", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    fig.tight_layout()
    return fig


sns.set(style="whitegrid")

# ===========================================================================
# PART 1
# ===========================================================================
st.header("Part 1: EC2 Cost and Performance Analysis")

st.subheader("Step 2: Explore the dataset")
st.write(f"**Rows:** {raw.shape[0]}  |  **Columns:** {raw.shape[1]}")
info_df = pd.DataFrame({
    "Column": raw.columns,
    "Non-null count": raw.notnull().sum().values,
    "Dtype": raw.dtypes.astype(str).values,
})
st.dataframe(info_df)
st.write("First few rows:")
st.dataframe(raw.head())

st.subheader("Step 3: Clean the data")
st.write("Missing values in cost columns after conversion:")
st.dataframe(data[COST_COLUMNS].isnull().sum().rename("Missing values"))

st.subheader("Step 4: Summary statistics")
st.dataframe(data[COST_COLUMNS].describe())

st.subheader("Step 5: Cost distribution")
st.pyplot(cost_boxplot(data, "Cost Comparison of Amazon EC2 Instances (Hourly)", "Set2"))

st.subheader("Step 6: Outliers in On-Demand cost (IQR method)")
outliers_on_demand = detect_outliers(data, "On Demand")
st.write(f"{len(outliers_on_demand)} outliers found")
st.dataframe(outliers_on_demand)

st.subheader("Step 7: Reserved vs On-Demand (10 cheapest)")
cost_comparison = data[["Name", "On Demand", "Linux Reserved cost"]].dropna().sort_values("On Demand")
st.dataframe(cost_comparison.head(10))

st.subheader("Step 8: Compare instance families")
col_a, col_b = st.columns(2)
family_1 = col_a.text_input("First family", "T2")
family_2 = col_b.text_input("Second family", "T3")
fam1 = filter_instance_family(data, family_1)
fam2 = filter_instance_family(data, family_2)

if fam1.empty or fam2.empty:
    st.warning("One of the families matched no rows — check how names look in the 'Name' column above.")
else:
    col_a.write(f"**{family_1} Instance Costs Summary**")
    col_a.dataframe(fam1[COST_COLUMNS].describe())
    col_b.write(f"**{family_2} Instance Costs Summary**")
    col_b.dataframe(fam2[COST_COLUMNS].describe())

    col_a.pyplot(cost_boxplot(fam1, f"Cost Distribution for {family_1} Instances", "Blues"))
    col_b.pyplot(cost_boxplot(fam2, f"Cost Distribution for {family_2} Instances", "Greens"))

    st.write(f"**On-Demand vs Reserved: {family_1} and {family_2} (10 lowest-cost)**")
    comparison = pd.concat([
        fam1[["Name", "On Demand", "Linux Reserved cost"]],
        fam2[["Name", "On Demand", "Linux Reserved cost"]],
    ])
    st.dataframe(comparison.dropna().sort_values("On Demand").head(10))

# ===========================================================================
# PART 2
# ===========================================================================
st.header("Part 2: Predicting EC2 On-Demand Costs")

st.subheader("Step 3: Feature engineering")
reg = data.copy()
reg["Instance Memory"] = pd.to_numeric(
    reg["Instance Memory"].astype(str).str.replace(" GiB", "", regex=False),
    errors="coerce",
)
reg["vCPUs"] = pd.to_numeric(
    reg["vCPUs"].astype(str).str.extract(r"(\d+)", expand=False),
    errors="coerce",
)
st.dataframe(reg[["Instance Memory", "vCPUs"]].head())

st.subheader("Step 4: Handle missing data")
data_cleaned = reg.dropna(subset=["On Demand", "Instance Memory", "vCPUs"])
st.write(f"Rows kept: {len(data_cleaned)} of {len(reg)}")
st.dataframe(
    data_cleaned[["On Demand", "Instance Memory", "vCPUs"]].isnull().sum().rename("Missing values")
)

st.subheader("Step 5: Split the data (80/20)")
X = data_cleaned[["Instance Memory", "vCPUs"]]
y = data_cleaned["On Demand"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

st.subheader("Step 6: Train a linear regression model")
# The model is trained on the log of cost, memory and vCPUs. Converting the
# prediction back with np.exp() means a predicted cost can never be negative.
model = LinearRegression()
model.fit(np.log(X_train), np.log(y_train))
st.write(f"**Intercept:** {model.intercept_:.6f}")
st.dataframe(pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_}))
st.caption("Coefficients are on a log scale: a coefficient of 0.5 means doubling that feature "
           "multiplies the predicted cost by about 2^0.5 ≈ 1.41.")

st.subheader("Step 7: Evaluate the model")
y_pred = np.exp(model.predict(np.log(X_test)))
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.4f}")
m2.metric("MSE", f"{mse:.4f}")
m3.metric("RMSE", f"{rmse:.4f}")

st.subheader("Step 8: Actual vs predicted")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.7, color="b")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
ax.set_title("Actual vs Predicted On-Demand Costs")
ax.set_xlabel("Actual On-Demand Cost")
ax.set_ylabel("Predicted On-Demand Cost")
ax.set_xscale("log")
ax.set_yscale("log")
st.pyplot(fig)

st.subheader("Step 9: Predict a new instance")
c1, c2 = st.columns(2)
mem = c1.number_input("Instance Memory (GiB)", min_value=0.5, value=4.0, step=0.5)
cpus = c2.number_input("vCPUs", min_value=1, value=2, step=1)
new_instance = pd.DataFrame([[mem, cpus]], columns=["Instance Memory", "vCPUs"])
predicted_cost = np.exp(model.predict(np.log(new_instance)))
st.success(f"Predicted On-Demand Cost for {mem:g} GiB, {cpus} vCPUs: ${predicted_cost[0]:.4f} per hour")