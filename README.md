# Predicting Late Shipments: A Machine Learning Approach

In this project, I use machine learning to develop two predictive models for a global sports and outdoor equipment retailer to proactively identify high-risk shipments before delays occur. The data, which is publicly available, is provided by DataCo and contains detailed order and shipping information. Using this data, I train two Random Forest machine learning classifiers:

1. Late Order Model (Optimized for Accuracy) – Predicts whether an order will be late by at least 1 day.
2. Very Late Order Model (Optimized for Recall) – Predicts if an order will be late by at least 3 days.

## How to Run
1. Clone this repository: `git clone https://github.com/bengtsoderlund/supply_chain_project.git`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `notebooks/supply_chain_analysis.ipynb` in Jupyter Notebook and follow the steps.

## Results
- Both models perform very well in predicting late orders.
- The Late Order Model has an accuracy score of 92.1%.
- The Very Late Order Model has a recall score of 97.3%. This means that the classifier is able to flag 97.3% of all orders that will be at least 3 days late.
- Features that have a highly predictive power include shipping mode, order value, order city, order state, the day of the order, and the number of units ordered.
