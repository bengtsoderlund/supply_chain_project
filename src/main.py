
###################################
###  PREDICTING LATE SHIPMENTS  ###
###################################


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# SETTING PROJECT ROOT DIRECTORY DYNAMICALLY
try:
    BASE_DIR = Path(__file__).resolve().parent.parent  
except NameError:
    BASE_DIR = Path().resolve()

    if BASE_DIR.name == "src":
        BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
data_path = DATA_DIR / "DataCoSupplyChainDataset.csv"



###################  LOAD AND EXPLORE DATA  #######################
df = pd.read_csv(data_path, encoding='latin1')

# Dataset dimensions
print(f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]:,} columns.")

# Check column names and data types
print("\nColumn Names & Data Types:")
print(df.dtypes)

# Check missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]  # Only show columns with missing values
if not missing_values.empty:
    print("\nMissing Values:")
    print(missing_values)
else:
    print("\nNo missing values found.")



##################  DATA CLEANING AND FEATURE ENGENEERING  ####################


# REMOVE INCORRECT PRODUCT CATEGORY NAME
df = df[df['Category Name'] != 'As Seen on  TV!']

# REMOVE INCORRECT CUSTOMER STATES
df = df[~df['Customer State'].isin(['91732', '95758'])]

# DROP LEADING, LAGGING, AND DOUBLE SPACES OF STRING FEATURES
df['Customer Street'] = df['Customer Street'].str.strip().str.replace(r'\s+', ' ', regex=True)

# KEEP COMPLETE OR CLOSED ORDERS
df = df[df['Order Status'].isin(['COMPLETE', 'CLOSED'])]


### CREATE ADDITIONAL FEATURES

# CREATE TARGET VARIABLES
df['late'] = (df['Days for shipping (real)'] > df['Days for shipment (scheduled)']).astype(int)
df['very_late'] = (df['Days for shipping (real)'] > (df['Days for shipment (scheduled)'] + 2)).astype(int)


# TOTAL VALUE OF ENTIRE ORDER
df['Order Value'] = df.groupby('Order Id')['Order Item Total'].transform('sum')

# UNIQUE ITEMS PER ORDER
df['Unique Items per Order'] = df.groupby('Order Id')['Order Item Id'].transform('nunique')

# TOTAL NUMBER OF UNITS PER ORDER
df['Units per Order'] = df.groupby('Order Id')['Order Item Quantity'].transform('sum')

# TIME FEATURES
df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'], format='%m/%d/%Y %H:%M')
df['year'] = df['order date (DateOrders)'].dt.year
df['month'] = df['order date (DateOrders)'].dt.month
df['day'] = df['order date (DateOrders)'].dt.day
df['hour'] = df['order date (DateOrders)'].dt.hour
df['minute'] = df['order date (DateOrders)'].dt.minute
df['day_of_week'] = df['order date (DateOrders)'].dt.weekday

# DROP FEATURES
df = df.drop([
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Sales per customer',
    'Delivery Status',
    'Late_delivery_risk',
    'Category Name',
    'Customer Email',
    'Customer Fname',
    'Customer Id',
    'Customer Lname',
    'Customer Password',
    'Department Name',
    'Market',
    'Order Customer Id',
    'order date (DateOrders)',
    'Order Id',
    'Order Item Discount',
    'Order Item Id',
    'Order Item Profit Ratio',
    'Order Status',
    'Order Status',
    'Order Zipcode',
    'Product Category Id',
    'Product Description',
    'Product Image',
    'Product Name',
    'Product Status',
    'shipping date (DateOrders)',
    'Latitude',
    'Longitude'
], axis=1)


# DO TRAIN-TEST SPLIT (KEEP SAME TRAIN-TEST SPLIT FOR BOTH MODELS)
X = df.drop(columns=['late', 'very_late'])
y_late = df['late']
y_very_late = df['very_late']
X_train, X_test, y_late_train, y_late_test = train_test_split(X, y_late, random_state=42)

y_very_late_train = y_very_late.loc[y_late_train.index]
y_very_late_test = y_very_late.loc[y_late_test.index]



# SEPARATE FEATURE TYPES
numerical_features = ['Benefit per order', 'Order Item Product Price', 'Order Item Quantity', 'Sales', 'Order Item Total',
                     'Product Price', 'year', 'month', 'day', 'hour', 'minute', 'Order Value', 'Unique Items per Order', 
                     'Order Item Discount Rate', 'Units per Order', 'Order Profit Per Order']
onehot_features = ['Type', 'Customer Segment', 'Shipping Mode']
label_features = ['Category Id', 'Customer City', 'Customer Country', 'Customer State', 'Customer Street', 'Customer Zipcode'
                 , 'Department Id', 'Order City', 'Order Country', 'Order Item Cardprod Id', 'Order Region', 'Order State'
                 , 'Product Card Id']


# INITIALIZE SCALERS/ENCODERS
scaler = RobustScaler() # Alternatively: MinMaxScaler()
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)


# STEP 1 - APPLY MIN-MAX SCALER TO NUMERICAL FEATURES AND CONVERT BACK THE ORIGINAL DATAFRAME STRUCTURE

X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)



# STEP 2 - PROCESS LOW CARDINALITY CATEGORICAL FEATURES USING ONE-HOT ENCODING

X_train_onehot = pd.DataFrame(
    onehot_encoder.fit_transform(X_train[onehot_features]),
    columns=onehot_encoder.get_feature_names_out(onehot_features),
    index=X_train.index
)

X_test_onehot = pd.DataFrame(
    onehot_encoder.transform(X_test[onehot_features]),
    columns=onehot_encoder.get_feature_names_out(onehot_features),
    index=X_test.index
)



# STEP 3 - PROCESS HIGH CARDINALITY CATEGORICAL FEATURES USING LABEL ENCODING

X_train_label = pd.DataFrame(
    ordinal_encoder.fit_transform(X_train[label_features]),
    columns=label_features,
    index=X_train.index
)

X_test_label = pd.DataFrame(
    ordinal_encoder.transform(X_test[label_features]),
    columns=label_features,
    index=X_test.index
)


# CONCATENATE PRE-PROCESSED FEATURES
X_train_final = pd.concat([X_train_scaled, X_train_onehot, X_train_label], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test_onehot, X_test_label], axis=1)


# DROP ADDITIONAl FEATUREES

vars_to_drop = [
    'Customer City', # Basically becomes a customer id
    'Customer Street', # -""-
    'Customer Zipcode', # -""-
    'minute', # Irrelevant
    'hour', # Irrelevant
    'Order Item Product Price', # Perfectly correlated feature
    'Sales', # Perfectly correlated feature
    'Order Item Cardprod Id', # Perfectly correlated feature
    'Product Card Id', # Perfectly correlated feature
    'Benefit per order'] # Perfectly correlated feature


X_train_final = X_train_final.drop(vars_to_drop, axis=1)
X_test_final = X_test_final.drop(vars_to_drop, axis=1)



##################  EXPLORATORY DATA ANALYSIS  ####################

# CHECK CORRELATIONS OF FEATURES
plt.figure(figsize=(12, 10))
corr = X_train_final.corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap", fontsize=22, fontweight='bold', fontfamily='Georgia')
plt.xticks(rotation=45, ha='right')
plt.show()



##################  MODEL BUILDING 1 (PREDICT LATE SHIPMENTS)  ####################
# USE RANDOM FOREST
# OPTIMZE CLASSIFIER WITH REGARDS TO ACCURACY


# FIND OPTIMAL PARAMETERS USING RANDOM SERACH

"""
# Define the parameter distribution
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [25, 30, 50, 60, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True]
}

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train_final, y_late_train)

# Get the best estimator and its parameters
best_rf = random_search.best_estimator_
best_params = random_search.best_params_

# Predict on training and test sets using the best model
y_late_train_pred = best_rf.predict(X_train_final)
y_late_test_pred = best_rf.predict(X_test_final)

# Calculate recall scores and best params
train_accuracy = accuracy_score(y_late_train, y_late_train_pred)
test_accuracy = accuracy_score(y_late_test, y_late_test_pred)

# Print results
print("")
print("Accuracy Classifier")

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

print("")
print("Best Parameters:", best_params)


# BEST PARAMS FOR LAST RUN
#Best Parameters: {'bootstrap': True, 'max_depth': 50, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 104}
#Test Accuracy: 0.9214972180070814
"""

# USE BEST PARAMETERS IN RANDOM FOREST  CLASSIFIER

rf = RandomForestClassifier(
    random_state=42,
    bootstrap=True,
    max_depth=50,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=4,
    n_estimators=104,
)

rf.fit(X_train_final, y_late_train)

y_late_train_pred = rf.predict(X_train_final)
y_late_test_pred = rf.predict(X_test_final)

train_accuracy = accuracy_score(y_late_train, y_late_train_pred)
test_accuracy = accuracy_score(y_late_test, y_late_test_pred)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)




##################  CHECK FEATURE IMPORTANCE  ####################

importances = rf.feature_importances_
feature_names = X_train_final.columns
feature_importances = list(zip(feature_names, importances))

feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("Feature Importances")
for feature, importance in feature_importances:
    print(f"{feature}: {importance:.4f}")







##################  MODEL BUILDING 2 (PREDICT VERY LATE SHIPMENTS)  ####################
# USE RANDOM FOREST

"""
############  DO RANDOM SERACH  #######################
# Define the parameter distribution
param_dist = {
    'n_estimators': randint(100, 500),  # Focus around the current best
    'max_depth': [10, 20, 30, 40, 50, 60],  # Slightly restrict depth range
    'min_samples_split': randint(5, 20),  # Narrow range around 10
    'min_samples_leaf': randint(1, 10),  # Enforce slightly larger leaves
    'max_features': ['sqrt', 'log2', None],  # Stick to 'sqrt' since it's working well
    'bootstrap': [True],  # Bootstrap sampling
    'criterion': ['gini', 'entropy']
}

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=200,  # Increase iterations
    cv=5,
    scoring='recall',
    verbose=0,
    random_state=42,
    n_jobs=-1
)


# Fit the model
random_search.fit(X_train_final, y_very_late_train)

# Get the best estimator
best_rf = random_search.best_estimator_


# Predict on training and test sets using the best model
y_very_late_train_pred = best_rf.predict(X_train_final)
y_very_late_test_pred = best_rf.predict(X_test_final)

# Calculate recall scores and best params
train_recall = recall_score(y_very_late_train, y_very_late_train_pred)
test_recall = recall_score(y_very_late_test, y_very_late_test_pred)

### Adjust (reduce) decicion threshold
# Get predicted probabilities for the positive class (very late deliveries)
y_very_late_test_prob = best_rf.predict_proba(X_test_final)[:, 1]

# Define a custom decision threshold (e.g., 0.3)
threshold = 0.3  # Lower threshold to increase recall
y_very_late_test_pred_adjusted = (y_very_late_test_prob >= threshold).astype(int)

# Calculate recall score at the adjusted threshold
adjusted_test_recall = recall_score(y_very_late_test, y_very_late_test_pred_adjusted)
adjusted_test_precision = precision_score(y_very_late_test, y_very_late_test_pred_adjusted)
adjusted_test_f1 = f1_score(y_very_late_test, y_very_late_test_pred_adjusted)

# Print results
print("")
print("Recall Classifier")

print("")
print("Train Recall:", train_recall)
print("Test Recall:", test_recall)
print("")
print("Adjusted Test Recall", adjusted_test_recall)
print("Adjusted Test Precision", adjusted_test_precision)
print("Adjusted Test F1", adjusted_test_f1)

# OBTAIN BEST PARAMETERS
best_params = random_search.best_params_
print("")
print("Best Parameters:", best_params)
# Best Parameters: {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 30, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 271}
"""


#########  USE BEST PARAMETERS FOR RANDOM FOREST CLASSIFIER  ###############

rf = RandomForestClassifier(
    random_state=42,
    bootstrap=True,
    criterion = 'entropy',    
    max_depth=30,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=271
)

rf.fit(X_train_final, y_very_late_train)

# reduce decision threshold
threshold = 0.3
y_very_late_test_prob = rf.predict_proba(X_test_final)[:, 1]
y_very_late_test_pred = (y_very_late_test_prob >= threshold).astype(int)
test_recall = recall_score(y_very_late_test, y_very_late_test_pred)

print("Test Recall:", test_recall)



#####  DRAW CONFUSION MATRIX  #############
cm = confusion_matrix(y_very_late_test, y_very_late_test_pred)
print("\nConfusion Matrix:")
print(cm)

#######  CREATE PRECISION-RECALL CURVE  ###############
#Calculate precision, recall, and thresholds for the test set
precision, recall, thresholds = precision_recall_curve(y_very_late_test, y_very_late_test_prob)

# Calculate the Average Precision (AP) Score
average_precision = average_precision_score(y_very_late_test, y_very_late_test_prob)

# Plot Precision-Recall Curve
plt.plot(recall, precision, label=f'AP = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


##########  CHECK FEATURE IMPORTANCE  ###########

importances = rf.feature_importances_
feature_names = X_train_final.columns
feature_importances = list(zip(feature_names, importances))

feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("")
print("Feature Importances")
for feature, importance in feature_importances:
    print(f"{feature}: {importance:.4f}")







