import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt

st.title("ABN % Bucket Classification and Simulation")

uploaded_file = st.file_uploader("Upload Weekly Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Multi-select filter options
    language = st.multiselect("Select Language(s)", df['Language'].dropna().unique())
    req_media = st.multiselect("Select Req Media(s)", df['Req Media'].dropna().unique())
    usd = st.multiselect("Select USD(s)", df['USD'].dropna().unique())
    level = st.multiselect("Select Level(s)", df['Level'].dropna().unique())

    # Filter data
    df_filtered = df[
        df['Language'].isin(language) &
        df['Req Media'].isin(req_media) &
        df['USD'].isin(usd) &
        df['Level'].isin(level)
    ].copy()

    # Convert date and select week
    df_filtered['Date'] = pd.to_datetime(df_filtered['startDate per day'])
    week_start = st.date_input("Select Week Starting Date", value=df_filtered['Date'].min())
    week_end = week_start + pd.Timedelta(days=6)
    df_week = df_filtered[(df_filtered['Date'] >= pd.to_datetime(week_start)) & (df_filtered['Date'] <= pd.to_datetime(week_end))]

    # Define features and target
    features = ['Q2', 'Demand', 'Staffing', 'Requirement', 'Occ Assumption']
    target = 'ABN %'
    df_model = df_filtered[features + [target]].dropna()

    # Dynamic bucketing based on ABN % distribution
    abn_series = df_model[target]
    num_buckets = min(5, len(abn_series.unique()))
    quantiles = np.linspace(0, 1, num_buckets + 1)
    bucket_edges = abn_series.quantile(quantiles).round(4).unique()
    bucket_edges = np.sort(np.unique(bucket_edges))

    bucket_labels = []
    for i in range(len(bucket_edges) - 1):
        lower = bucket_edges[i]
        upper = bucket_edges[i + 1]
        if i == len(bucket_edges) - 2:
            label = f"{lower:.2f}%+"
        else:
            label = f"{lower:.2f}%-{upper:.2f}%"
        bucket_labels.append(label)

    def bucket_abn_dynamic(abn):
        for i in range(len(bucket_edges) - 1):
            if abn < bucket_edges[i + 1]:
                return bucket_labels[i]
        return bucket_labels[-1]

    df_model['ABN_bucket'] = df_model[target].apply(bucket_abn_dynamic)

    # Encode target
    le = LabelEncoder()
    df_model['ABN_encoded'] = le.fit_transform(df_model['ABN_bucket'])

    X = df_model[features]
    y = df_model['ABN_encoded']

    # Check data sufficiency
    if len(df_model) < 10 or len(np.unique(y)) < 2:
        st.error("Not enough data or class diversity to train the model. Please select a different week or filter.")
    else:
        # Stratified split
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())

        if min_class_count < 2:
            st.warning("Some ABN % buckets have fewer than 2 samples. Using random split instead of stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBClassifier(eval_metric='mlogloss')
            model.fit(X_train, y_train)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Safe cross-validation
            n_splits = min(3, min_class_count)
            if n_splits < 2:
                st.warning("Too few samples in some buckets for cross-validation. Skipping GridSearchCV.")
                model = xgb.XGBClassifier(eval_metric='mlogloss')
                model.fit(X_train, y_train)
            else:
                param_grid = {
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1],
                    'n_estimators': [100, 200],
                    'subsample': [0.8, 1.0]
                }
                skf = StratifiedKFold(n_splits=n_splits)
                grid_search = GridSearchCV(
                    xgb.XGBClassifier(eval_metric='mlogloss'),
                    param_grid,
                    cv=skf,
                    scoring='accuracy',
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_

        # Classification report
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            labels=np.unique(y_test),
            target_names=le.inverse_transform(np.unique(y_test)),
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df)

        # Plot metrics
        st.subheader("Classification Metrics Visualization")
        fig, ax = plt.subplots(figsize=(10, 5))
        buckets = report_df.index[:-3]
        ax.plot(buckets, report_df.loc[buckets, 'precision'], marker='o', label='Precision')
        ax.plot(buckets, report_df.loc[buckets, 'recall'], marker='s', label='Recall')
        ax.plot(buckets, report_df.loc[buckets, 'f1-score'], marker='^', label='F1-Score')
        ax.set_title("Precision, Recall, F1-Score by ABN % Bucket")
        ax.set_xlabel("ABN % Bucket")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Simulation
        st.subheader("Simulate ABN % Bucket Change")

        row_index = st.number_input("Select a row from test data for simulation", min_value=0, max_value=len(X_test)-1, value=0)
        base_row = X_test.iloc[row_index].copy()

        st.write("Base Row Values:")
        st.dataframe(pd.DataFrame([base_row]))

        modified_row = base_row.copy()
        st.write("Adjust Feature Values:")
        for feature in features:
            change_percent = st.slider(f"{feature} change (%)", -500, 500, 0)
            modified_row[feature] = base_row[feature] * (1 + change_percent / 100)

        original_pred = model.predict(pd.DataFrame([base_row]))[0]
        new_pred = model.predict(pd.DataFrame([modified_row]))[0]

        original_bucket = le.inverse_transform([original_pred])[0]
        new_bucket = le.inverse_transform([new_pred])[0]
        transitioned = original_bucket != new_bucket

        result_df = pd.DataFrame([modified_row])
        result_df['Original Bucket'] = original_bucket
        result_df['New Bucket'] = new_bucket
        result_df['Transitioned'] = transitioned

        st.subheader("Simulation Result")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Simulation Report", csv, "abn_simulation_result.csv", "text/csv")