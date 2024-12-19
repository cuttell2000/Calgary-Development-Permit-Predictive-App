import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle
from sklearn.preprocessing import LabelEncoder
import gzip



# Load cleaned data
@st.cache_data
def load_data():

    # api
    dtypes = {
        "category": "object",  # Replace with appropriate types (e.g., "str", "float64")
        "concurrent_loc": "object",
        "sdabnumber": "object",
        "sdabdecisiondate": "object"  # Use "object" for mixed types initially
    }

    develop_permit2_df = pd.read_csv("https://data.calgary.ca/resource/6933-unw5.csv?$limit=500000", dtype=dtypes)

    # Handle missing data
    # Drop columns with too many missing values (arbitrary threshold: 90% missing)
    columns_to_drop = ['concurrent_loc', 'sdabnumber', 'sdabhearingdate', 'sdabdecision', 'sdabdecisiondate']
    develop_permit2_df.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Fill missing values for essential columns
    develop_permit2_df.fillna({'ward' : develop_permit2_df['ward'].median()}, inplace=True)
    develop_permit2_df.fillna({'decision': 'Unknown'}, inplace=True) # 'Unknown' as placeholder
    develop_permit2_df.fillna({'communityname': 'Unknown'}, inplace=True)

    # Parse date columns into datetime format
    date_columns = ['applieddate', 'decisiondate', 'releasedate', 'mustcommencedate', 'canceledrefuseddate']
    for col in date_columns:
        develop_permit2_df[col] = pd.to_datetime(develop_permit2_df[col], errors='coerce')

    # Create derived columns
    develop_permit2_df['processing_time'] = (develop_permit2_df['decisiondate'] - develop_permit2_df['applieddate']).dt.days
    develop_permit2_df['approval_indicator'] = develop_permit2_df['decision'].apply(lambda x: 1 if x == 'Approval' else 0)

    # Filter out rows with negative 'process_time'
    develop_permit3_df = develop_permit2_df[develop_permit2_df['processing_time'] >= 0]

    # Use .loc to avoid SettingWithCopyWarning
    develop_permit3_df.loc[:, 'category'] = develop_permit3_df['category'].str.lower().str.strip()
    develop_permit3_df.loc[:, 'proposedusecode'] = develop_permit3_df['proposedusecode'].str.lower().str.strip()

    # return and load data
    return develop_permit3_df

data = load_data()

# load data
# Feature Engineering
data_ml = data.copy()

# # Encode categorical columns
# data_ml['status_encoded'] = data_ml['statuscurrent'].astype('category').cat.codes
# data_ml['quadrant_encoded'] = data_ml['quadrant'].astype('category').cat.codes

# Encode communityname using LabelEncoder
le_community = LabelEncoder()
data_ml['communityname_encoded'] = le_community.fit_transform(data_ml['communityname'])

# Encode statuscurrent using LabelEncoder
le_status = LabelEncoder()
data_ml['status_encoded'] = le_status.fit_transform(data_ml['statuscurrent'])

# resample the imbalanced class distribution of target variable apporval_indicator to make it balanced
# with the independent variables

# copy data
data_ml2 = data_ml.copy()

# from exploratory data analysis

# Class Distribution:
# approval_indicator
# 1    158610
# 0      5349
# Name: count, dtype: int64

# Class Percentages:
# approval_indicator
# 1    96.737599
# 0     3.262401
# Name: count, dtype: float64

# Check for class imbalance (threshold is subjective)
imbalance_threshold = 20  # Example: 20% difference

# Count the occurrences of each class in the target variable
class_counts = data_ml2['approval_indicator'].value_counts()

# Calculate the percentage of each class
class_percentages = class_counts / len(data_ml2) * 100

# Applying Random Oversampling for detected Potential Class Imbalance
if abs(class_percentages.iloc[0] - class_percentages.iloc[1]) > imbalance_threshold:
    print("\nWARNING: Potential Class Imbalance detected.")
    print("Applying Random Oversampling...")

    # Separate features (X) and target variable (y)
    X = data_ml2.drop('approval_indicator', axis=1)
    y = data_ml2['approval_indicator']

    # Initialize and apply RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    data_ml_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    data_ml_resampled['approval_indicator'] = y_resampled

    data_ml2 = data_ml_resampled
    
#  Applying Random Undersampling for detected Potential Class Imbalance
elif abs(class_percentages.iloc[0] - class_percentages.iloc[1]) < (100 - imbalance_threshold):
    print("\nWARNING: Potential Class Imbalance detected.")
    print("Applying Random Undersampling...")

    # Separate features (X) and target variable (y)
    X = data_ml2.drop('approval_indicator', axis=1)
    y = data_ml2['approval_indicator']

    # Initialize and apply RandomUnderSampler
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    data_ml_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    data_ml_resampled['approval_indicator'] = y_resampled

    data_ml2 = data_ml_resampled
    
else:
    print("\nClass balance appears to be appropriate.")

# Now data_ml2 holds the balanced dataset (either oversampled, undersampled, or original)
# Proceed with your model training using data_ml2

# Prepare data for modeling
# features = ['communityname_encoded', 'category', 'processing_time', 'quadrant_encoded', 'ward']
features = ['communityname_encoded', 'category', 'processing_time', 'status_encoded']
X = data_ml2[features].copy()

# Handle categorical data (One-Hot Encoding or Label Encoding as needed)
X['category'] = X['category'].fillna('Unknown')  # Replace NaN with 'Unknown'
X = pd.get_dummies(X, columns=['category'], drop_first=True)

# Target variable
y = data_ml2['approval_indicator']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.write(f"Model Accuracy: {accuracy:.2%}")

# Save model to variable for pickle
model_approval = model

# Save the models to files
with open('approval_model.pkl', 'wb') as f:
    pickle.dump(model_approval, f)

# Load models
with open('approval_model.pkl', 'rb') as f:
    approval_model = pickle.load(f)

# Compress the data
with gzip.open('approval_model.gz', 'wb') as f:
    pickle.dump(approval_model, f)

# Sidebar for navigation
st.sidebar.title("Calgary Development Permit Predictive Model")
options = st.sidebar.radio(
    "Select a model",
    [
        "Permit Approval Prediction"
    ]
)

# Main model
st.title("City of Calgary Development Permit Predictive Analytics")

st.write("""
         This dashboard allows you to predict whether a permit application will be approved based on input data.
""")

#


if options == "Permit Approval Prediction":
    st.header("Permit Approval Prediction")

    st.sidebar.header("Input Features")

    def permit_approval_pred():
        # User inputs for prediction
        st.subheader("Input Permit Details")

        # User Input: Community Name
        community_list = data_ml2['communityname'].unique()
        selected_community = st.selectbox("Select Community Name", community_list)
        

        # Display Associated Ward and Quadrant
        ward = data_ml2.loc[data_ml2['communityname'] == selected_community, 'ward'].iloc[0]
        quadrant = data_ml2.loc[data_ml2['communityname'] == selected_community, 'quadrant'].iloc[0]

        # Show the selected ward and quadrant for clarity
        st.sidebar.write(f"**Ward:** {ward}")
        st.sidebar.write(f"**Quadrant:** {quadrant}")

        # Other inputs
        input_category = st.selectbox("Select Category", data_ml2['category'].fillna('Unknown').unique())
        input_processing_time = st.slider("Processing Time (days)", int(data_ml2['processing_time'].min()), int(data_ml2['processing_time'].max()), 30)
        # input_quadrant = st.selectbox("Select Quadrant", data_ml2['quadrant'].unique())
        # input_ward = st.slider("Select Ward", int(data_ml2['ward'].min()), int(data_ml2['ward'].max()), 7)
        # status_encoded = st.selectbox("Status Encoded", data_ml2['status_encoded'].unique())

        statuscurrent_list = data_ml2['statuscurrent'].unique()
        selected_statuscurrent = st.selectbox("Select Current Status", statuscurrent_list)

        # Preprocess inputs
        input_data = {
            'communityname_encoded': [le_community.transform([selected_community])[0]],
            'processing_time': input_processing_time,
            # 'quadrant_encoded': data_ml2[data_ml2['quadrant'] == input_quadrant]['quadrant_encoded'].values[0],
            # 'ward': input_ward,
            'status_encoded': [le_status.transform([selected_statuscurrent])[0]]
        }

        # One-hot encode category
        for col in X.columns:
            if col.startswith('category_'):
                input_data[col] = 1 if f"category_{input_category}" in col else 0

        # return pd.DataFrame([input_data])
        return pd.DataFrame(input_data)
    
    input_df = permit_approval_pred()

    # Make prediction
    if st.sidebar.button("Predict Approval"):
        prediction = model.predict(input_df)
        prediction_prob = model.predict_proba(input_df)[0]

        # Display prediction result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"Approval Likely (Confidence: {prediction_prob[1]:.2%})")
        else:
            st.error(f"Approval Unlikely (Confidence: {prediction_prob[0]:.2%})")