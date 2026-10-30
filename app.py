import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Late Delivery Risk Analytics",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.joblib")
FEATURE_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.joblib")
IMPORTANCE_PATH = os.path.join(ARTIFACT_DIR, "feature_importance.csv")

DROP_COLS = [
    "Late_delivery_risk", "Days for shipping (real)",
    "Days for shipment (scheduled)", "Delivery Status",
    "Order Status", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode", "Customer Id",
    "Order Customer Id", "Customer City", "Customer State",
    "Customer Country", "Order City", "Order State",
    "Order Country", "Latitude", "Longitude", "Type"
]

st.title("Late Delivery Risk Prediction")
st.caption("Machine Learning-Based Supply Chain Analytics")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    st.error("Model files were not found. Run main.py first to train the model.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


@st.cache_data
def read_csv_file(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes), encoding="latin1")


def prepare_features(data):
    X = data.drop(columns=DROP_COLS, errors="ignore").copy()

    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan

    return X.reindex(columns=feature_columns)


def predict_probabilities(data):
    X = prepare_features(data)
    return model.predict_proba(X)[:, 1]


def risk_category(probability):
    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Medium"
    return "High"


def add_prediction_columns(data, probabilities):
    result = data.copy()
    result["Predicted_Late_Risk"] = probabilities * 100
    result["Risk_Category"] = [
        risk_category(p) for p in probabilities
    ]
    return result


pages = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Upload",
        "Risk Overview",
        "Predict New Order",
        "Bulk CSV Prediction"
    ]
)


if pages == "Dataset Upload":
    st.subheader("Upload Dataset")
    st.write(
        "Upload a logistics CSV file to analyze orders "
        "and generate late-delivery risk predictions."
    )

    uploaded_dataset = st.file_uploader(
        "Choose a logistics dataset",
        type=["csv"],
        key="overview_dataset"
    )

    if uploaded_dataset is not None:
        try:
            df = read_csv_file(uploaded_dataset.getvalue())

            if df.empty:
                st.error("The uploaded CSV file is empty.")
            else:
                st.session_state["overview_dataset_bytes"] = (
                    uploaded_dataset.getvalue()
                )
                st.session_state["overview_dataset_name"] = (
                    uploaded_dataset.name
                )

                st.success(
                    f"Dataset loaded successfully: {uploaded_dataset.name}"
                )

                st.write(
                    f"Rows: {df.shape[0]:,} | "
                    f"Columns: {df.shape[1]:,}"
                )

                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)

                st.info(
                    "Open Risk Overview from the sidebar "
                    "to view the analysis."
                )

        except Exception as e:
            st.error(f"Unable to read the uploaded dataset: {e}")

    elif "overview_dataset_bytes" in st.session_state:
        st.info(
            "Current dataset: "
            + st.session_state.get(
                "overview_dataset_name",
                "Uploaded dataset"
            )
        )

        if st.button("Clear Uploaded Dataset"):
            st.session_state.pop("overview_dataset_bytes", None)
            st.session_state.pop("overview_dataset_name", None)
            st.rerun()

    else:
        st.info(
            "Upload a CSV file to begin. The dataset should contain "
            "columns compatible with the trained model."
        )


elif pages == "Risk Overview":

    if "overview_dataset_bytes" not in st.session_state:
        st.warning(
            "Upload a dataset first using the Dataset Upload page."
        )
        st.stop()

    try:
        df = read_csv_file(
            st.session_state["overview_dataset_bytes"]
        )
    except Exception as e:
        st.error(f"Unable to read the uploaded dataset: {e}")
        st.stop()

    actual_target = (
        df["Late_delivery_risk"].copy()
        if "Late_delivery_risk" in df.columns
        else None
    )

    try:
        with st.spinner("Calculating order risk..."):
            probabilities = predict_probabilities(df)
    except Exception as e:
        st.error(
            "Unable to generate predictions. Check whether the CSV "
            f"contains the required model features. Details: {e}"
        )
        st.stop()

    results = add_prediction_columns(df, probabilities)

    threshold = st.sidebar.slider(
        "High-risk threshold (%)",
        10, 90, 50, 5,
        key="overview_threshold"
    )

    if "Shipping Mode" in results.columns:
        modes = sorted(
            results["Shipping Mode"].dropna().astype(str).unique()
        )

        selected_modes = st.sidebar.multiselect(
            "Shipping Mode",
            modes,
            default=modes,
            key="overview_modes"
        )

        results = results[
            results["Shipping Mode"].astype(str).isin(selected_modes)
        ]

    if "Order Region" in results.columns:
        regions = sorted(
            results["Order Region"].dropna().astype(str).unique()
        )

        selected_regions = st.sidebar.multiselect(
            "Order Region",
            regions,
            default=regions,
            key="overview_regions"
        )

        results = results[
            results["Order Region"].astype(str).isin(selected_regions)
        ]

    results["Predicted_Status"] = np.where(
        results["Predicted_Late_Risk"] >= threshold,
        "High Risk",
        "Lower Risk"
    )

    total = len(results)

    high = int(
        (results["Predicted_Late_Risk"] >= threshold).sum()
    )

    average_risk = (
        results["Predicted_Late_Risk"].mean()
        if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Orders Analyzed", f"{total:,}")
    c2.metric("High-Risk Orders", f"{high:,}")

    c3.metric(
        "High-Risk Share",
        f"{high / total * 100:.1f}%" if total else "0%"
    )

    c4.metric(
        "Average Predicted Risk",
        f"{average_risk:.1f}%"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Risk Category Distribution")

        st.bar_chart(
            results["Risk_Category"].value_counts().reindex(
                ["Low", "Medium", "High"],
                fill_value=0
            )
        )

    with right:
        st.subheader("Average Risk by Shipping Mode")

        if "Shipping Mode" in results.columns and total:
            mode_risk = results.groupby("Shipping Mode")[
                "Predicted_Late_Risk"
            ].mean().sort_values(ascending=False)

            st.bar_chart(mode_risk)

    st.subheader("Order-Level Risk Explorer")

    search = st.text_input("Search Order ID")
    visible = results

    if search and "Order Id" in results.columns:
        visible = results[
            results["Order Id"].astype(str).str.contains(
                search,
                case=False,
                na=False
            )
        ]

    display_columns = [
        col for col in [
            "Order Id",
            "Shipping Mode",
            "Market",
            "Order Region",
            "Customer Segment",
            "Predicted_Late_Risk",
            "Risk_Category",
            "Predicted_Status"
        ]
        if col in visible.columns
    ]

    if display_columns:
        st.dataframe(
            visible[display_columns].sort_values(
                "Predicted_Late_Risk",
                ascending=False
            ),
            use_container_width=True
        )
    else:
        st.dataframe(visible, use_container_width=True)

    st.download_button(
        "Download Risk Results",
        results.to_csv(index=False).encode("utf-8"),
        "late_delivery_risk_results.csv",
        "text/csv"
    )

    st.subheader("Feature Importance")

    if os.path.exists(IMPORTANCE_PATH):
        importance = pd.read_csv(
            IMPORTANCE_PATH
        ).head(20).sort_values("Importance")

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_xlabel("Importance")
        ax.set_title("Top 20 Model Features")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info(
            "Feature importance file was not found. "
            "Run main.py to regenerate model artifacts."
        )

    if actual_target is not None:
        st.subheader("Actual Historical Target Distribution")

        st.caption(
            "This chart shows the target values present "
            "in the uploaded dataset."
        )

        st.bar_chart(
            actual_target.value_counts().sort_index()
        )


elif pages == "Predict New Order":

    st.subheader("Single Order Risk Prediction")

    st.write(
        "Enter the order details to estimate its late-delivery risk."
    )

    reference_bytes = st.session_state.get(
        "overview_dataset_bytes"
    )

    if reference_bytes is None:
        st.warning(
            "Upload a reference dataset first using the Dataset Upload page."
        )
        st.stop()

    reference = read_csv_file(reference_bytes)
    reference_X = prepare_features(reference)

    with st.form("single_order_form"):
        values = {}

        for col in feature_columns:
            series = reference_X[col]

            if pd.api.types.is_numeric_dtype(series):
                default = (
                    float(series.median())
                    if series.notna().any()
                    else 0.0
                )

                values[col] = st.number_input(
                    col,
                    value=default,
                    format="%.4f"
                )

            else:
                options = sorted(
                    series.dropna().astype(str).unique().tolist()
                )

                if options:
                    values[col] = st.selectbox(
                        col,
                        options
                    )
                else:
                    values[col] = st.text_input(col)

        submitted = st.form_submit_button(
            "Predict Delivery Risk"
        )

    if submitted:
        new_order = pd.DataFrame(
            [values],
            columns=feature_columns
        )

        try:
            probability = float(
                model.predict_proba(new_order)[0, 1]
            )

            category = risk_category(probability)

            st.metric(
                "Estimated Late-Delivery Risk",
                f"{probability * 100:.2f}%"
            )

            st.subheader(f"Risk Category: {category}")

            if category == "High":
                st.warning(
                    "Prioritize this order for shipment review and monitoring."
                )

            elif category == "Medium":
                st.info(
                    "Monitor shipment progress and check for possible delays."
                )

            else:
                st.success(
                    "The model estimates a relatively lower delivery risk."
                )

            st.caption(
                "This is a model estimate, not a guaranteed outcome. "
                "Prediction quality depends on the input data "
                "and feature availability."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


else:

    st.subheader("Bulk CSV Prediction")

    st.write(
        "Upload a CSV file to estimate delivery risk for multiple orders."
    )

    uploaded_file = st.file_uploader(
        "Upload orders CSV",
        type=["csv"],
        key="bulk_csv"
    )

    if uploaded_file is not None:
        try:
            batch = read_csv_file(uploaded_file.getvalue())

            if batch.empty:
                st.error("The uploaded CSV file is empty.")
                st.stop()

            with st.spinner("Calculating predictions..."):
                batch_probabilities = predict_probabilities(batch)

            batch_results = add_prediction_columns(
                batch,
                batch_probabilities
            )

            batch_threshold = st.slider(
                "High-risk threshold (%)",
                10, 90, 50, 5,
                key="bulk_threshold"
            )

            batch_results["Predicted_Status"] = np.where(
                batch_results["Predicted_Late_Risk"] >= batch_threshold,
                "High Risk",
                "Lower Risk"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Orders",
                f"{len(batch_results):,}"
            )

            b.metric(
                "High-Risk Orders",
                f"{(batch_results['Predicted_Late_Risk'] >= batch_threshold).sum():,}"
            )

            c.metric(
                "Average Predicted Risk",
                f"{batch_results['Predicted_Late_Risk'].mean():.1f}%"
            )

            st.dataframe(
                batch_results.sort_values(
                    "Predicted_Late_Risk",
                    ascending=False
                ),
                use_container_width=True
            )

            st.download_button(
                "Download Bulk Predictions",
                batch_results.to_csv(index=False).encode("utf-8"),
                "bulk_late_delivery_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(
                f"Unable to process the CSV file: {e}"
            )


st.caption(
    "Risk scores are estimates generated by the trained model "
    "and are not guarantees."
)