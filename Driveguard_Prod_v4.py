import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
from smart_dict import SMART_DEFINITIONS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "driveguard_model_v4" / "driveguard_smote_v4.joblib"
SAMPLE_DIR = BASE_DIR / "driveguard_extracted_v4" / "website_samples"


@st.cache_resource
def load_deployment_bundle():
    """Load the v4 SMOTE model and its matching preprocessing objects."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_PATH}. Run DriveGuard_Train_v4.py first."
        )

    bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "imputer", "scaler", "feature_names", "threshold"}
    missing_keys = required_keys.difference(bundle)
    if missing_keys:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing_keys)}")
    return bundle

# ==========================================
# 1. PAGE CONFIGURATION & SAFE CSS
# ==========================================
st.set_page_config(
    page_title="DriveGuard | S.M.A.R.T. Failure Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe CSS: Targets typography without breaking Streamlit's Material Icons or Flexbox DOM
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

    html, body, p, div, h1, h2, h3, h4, h5, h6, span, label, li {
        font-family: 'Montserrat', sans-serif;
    }

    /* Subtle Mapua Green background tint */
    .stApp {
        background-color: #F8FAF9;
    }

    /* Academic Headers */
    .mapua-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0038A8; /* Mapua Blue */
        margin-bottom: -10px;
        letter-spacing: -0.5px;
    }
    .mapua-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #4B5563;
        padding-bottom: 15px;
        border-bottom: 2px solid #CE1126; /* Mapua Red */
        margin-bottom: 30px;
    }

    /* Diagnostic Status Cards */
    .status-card-healthy {
        background-color: #F0FDF4;
        border-left: 6px solid #007A33; /* Mapua Green */
        border-radius: 6px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-card-risk {
        background-color: #FEF2F2;
        border-left: 6px solid #CE1126; /* Mapua Red */
        border-radius: 6px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6B7280;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .status-value-healthy {
        font-size: 1.8rem;
        font-weight: 700;
        color: #007A33;
    }
    .status-value-risk {
        font-size: 1.8rem;
        font-weight: 700;
        color: #CE1126;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR NAVIGATION & ACADEMIC BRANDING
# ==========================================
# Using use_container_width perfectly scales the logo without breaking the sidebar
st.sidebar.image(
    "mapua_logo.png",
    use_container_width=True)
st.sidebar.markdown("### System Navigation")

app_mode = st.sidebar.radio(
    "Select Module:",
    [
        "1. Project Overview",
        "2. Diagnostic Screening",
        "3. Dataset & AI Model",
        "4. CpE Program Relevance"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("""
**DriveGuard Diagnostic Engine**  
**Group 8:** Neil Justin V. Bermoy & David D. Vallenas  
**Program:** B.S. Computer Engineering  
**Institution:** Mapúa Malayan Colleges Mindanao  
""")

# ==========================================
# 3. TAB 1: HOMEPAGE & OVERVIEW
# ==========================================
if app_mode == "1. Project Overview":
    st.markdown("<div class='mapua-title'>DriveGuard: S.M.A.R.T. Failure Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='mapua-subtitle'>Artificial Intelligence Framework for Proactive Hardware Diagnostics</div>",
        unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.subheader("The Problem Statement")
        st.write(
            "Unexpected hard disk drive (HDD) and solid-state drive (SSD) mechanical or controller failures "
            "cause irreversible data loss and severe system downtime. Traditional monitoring systems rely on "
            "static, threshold-based (TB) firmware alerts that typically notify users only after critical sectors "
            "or files have already been corrupted."
        )
        st.write(
            "**The DriveGuard Solution:** This system applies a Random Forest model to S.M.A.R.T. telemetry "
            "to screen for patterns associated with Backblaze failure-day records. The current dataset "
            "supports terminal failure-day classification, not a guaranteed multi-day early warning."
        )

        st.subheader("System Methodology")
        st.write(
            "1. **Data Acquisition:** S.M.A.R.T. operational telemetry derived from the public Backblaze repository.")
        st.write("2. **Preprocessing:** Extraneous metadata filtration, missing value imputation, and Min-Max scaling.")
        st.write("3. **Class Imbalance:** Compares cost-sensitive weighting with SMOTE applied only to training data.")
        st.write("4. **Classification:** Random Forest thresholds are selected on validation data using F2-score.")

    with col2:
        st.subheader("Target Beneficiaries")
        st.info(
            "**Standard PC Users:**\nProvide an additional S.M.A.R.T. screening result that can support backup and maintenance decisions.")
        st.success(
            "**Independent Repair Shops:**\nUse the screening output as one reference during drive assessment, alongside established diagnostic tools.")
        st.error(
            "**Small-Scale Data Centers:**\nExplore telemetry-based screening as a supplementary input for storage maintenance.")


# ==========================================
# 4. TAB 2: DIAGNOSTIC SCREENING (INPUT & PREDICTION)
# ==========================================
elif app_mode == "2. Diagnostic Screening":
    st.markdown("<div class='mapua-title'>Diagnostic Telemetry Screening</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='mapua-subtitle'>Upload S.M.A.R.T. CSV telemetry logs for model-based screening.</div>",
        unsafe_allow_html=True)

    st.warning(
        "Model limitation: Backblaze labels failure=1 on the last operational day before a recorded failure. "
        "This model screens for terminal failure-day patterns. It has not been validated as a multi-day warning system."
    )

    st.subheader("Backblaze Demonstration Samples")
    sample_files = sorted(SAMPLE_DIR.glob("driveguard_sample_*.csv")) if SAMPLE_DIR.exists() else []
    if sample_files:
        selected_sample = st.selectbox(
            "Choose a real held-out demonstration sample:",
            options=sample_files,
            format_func=lambda path: path.name
        )
        st.download_button(
            "Download Selected Backblaze Sample (.csv)",
            data=selected_sample.read_bytes(),
            file_name=selected_sample.name,
            mime="text/csv",
            use_container_width=True
        )
        st.caption(
            "These samples contain real Backblaze records from the separate website-demo drive partition. "
            "The failure column is included as a reference label and is not used as a model input."
        )
    else:
        st.warning(
            "Website sample files were not found. Make sure the driveguard_extracted_v4 folder "
            "is beside this Streamlit app."
        )

    st.divider()
    uploaded_file = st.file_uploader("Upload S.M.A.R.T. telemetry CSV:", type=["csv"])

    if uploaded_file is not None:
        try:
            bundle = load_deployment_bundle()
            model = bundle["model"]
            imputer = bundle["imputer"]
            scaler = bundle["scaler"]
            feature_names = list(bundle["feature_names"])
            threshold = float(bundle["threshold"])

            df = pd.read_csv(uploaded_file, low_memory=False)
            if df.empty:
                raise ValueError("The uploaded CSV contains no records.")
            if not feature_names:
                raise ValueError("The saved model bundle contains no feature names.")

            # Reindex to the exact SMART columns and order used during training.
            # Missing columns become NaN and are filled by the saved training-fitted imputer.
            # The 'failure' label and all metadata are excluded from predictors.
            X_input = df.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce")
            X_imputed = imputer.transform(X_input)
            X_scaled = scaler.transform(X_imputed)

            # Apply the threshold selected on the validation set, not sklearn's default 0.50.
            probabilities = model.predict_proba(X_scaled)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            has_actual_labels = "failure" in df.columns
            actual_labels = (
                pd.to_numeric(df["failure"], errors="coerce")
                if has_actual_labels else pd.Series([np.nan] * len(df), index=df.index)
            )

            result_rows = []
            for position, (row_index, row) in enumerate(df.iterrows()):
                serial = row.get("serial_number", f"Drive_ID_{position + 1}")
                actual = actual_labels.iloc[position]
                actual_text = (
                    "Failure-day label (1)" if actual == 1 else
                    "Healthy label (0)" if actual == 0 else
                    "Not provided"
                )
                score = float(probabilities[position])
                result_rows.append({
                    "Row": int(position + 1),
                    "Serial Number": str(serial),
                    "Prediction": "FLAGGED" if predictions[position] == 1 else "NOT FLAGGED",
                    "Model Score": f"{score * 100:.1f}%",
                    "Actual Dataset Label": actual_text,
                    "score_raw": score,
                    "original_row": row,
                })

            results_df = pd.DataFrame(result_rows)
            flagged_df = results_df[results_df["Prediction"] == "FLAGGED"]
            not_flagged_count = len(results_df) - len(flagged_df)

            st.subheader("Batch Screening Summary")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Records Analyzed", len(results_df))
            col_s2.metric("Flagged Records", len(flagged_df))
            col_s3.metric("Not Flagged", not_flagged_count)
            st.caption(f"Classification threshold: {threshold:.2f} (selected using validation data)")

            display_columns = [
                "Row", "Serial Number", "Prediction", "Model Score", "Actual Dataset Label"
            ]
            st.dataframe(results_df[display_columns], use_container_width=True, hide_index=True)

            if has_actual_labels:
                st.caption(
                    "The actual dataset label is shown only for comparison when present in the uploaded CSV. "
                    "It is not passed to the model and does not affect the prediction."
                )

            if not flagged_df.empty:
                st.divider()
                st.subheader("Flagged Record Review")
                selected_row_number = st.selectbox(
                    "Select a flagged record:",
                    options=flagged_df["Row"].tolist(),
                    format_func=lambda number: (
                        f"Row {number} | "
                        f"{flagged_df.loc[flagged_df['Row'] == number, 'Serial Number'].iloc[0]}"
                    )
                )
                selected_item = flagged_df.loc[flagged_df["Row"] == selected_row_number].iloc[0]
                row = selected_item["original_row"]

                col_verdict, col_detail = st.columns([1, 2.5], gap="large")
                with col_verdict:
                    st.error("FLAGGED FOR REVIEW")
                    st.metric("Serial Number", selected_item["Serial Number"])
                    st.metric("Model Score", selected_item["Model Score"])
                    st.caption(
                        "This score is a model output, not a calibrated guarantee of failure or a time-to-failure estimate."
                    )
                    st.write(
                        "Recommended next step: verify the drive with established diagnostic tools and ensure important data is backed up."
                    )

                with col_detail:
                    smart_breakdown = []
                    for col in row.index:
                        if col.lower().startswith("smart_") and col.lower().endswith("_raw"):
                            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                            if pd.notna(value) and float(value) > 0:
                                clean_name = col.replace("smart_", "").replace("_raw", "").upper()
                                smart_breakdown.append({
                                    "Attribute": f"S{clean_name}",
                                    "Raw Count": float(value)
                                })

                    if smart_breakdown:
                        attr_data = (
                            pd.DataFrame(smart_breakdown)
                            .sort_values(by="Raw Count", ascending=False)
                            .head(5)
                        )
                        fig = px.bar(
                            attr_data,
                            x="Attribute",
                            y="Raw Count",
                            color="Raw Count",
                            color_continuous_scale="Reds",
                            text="Raw Count",
                            title="Largest Non-zero SMART Raw Values (Reference Only)"
                        )
                        fig.update_layout(
                            height=300,
                            margin=dict(l=0, r=0, t=35, b=0),
                            yaxis_title="Reported Raw Value",
                            xaxis_title="SMART Attribute",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("##### SMART Attribute Dictionary")
                        for _, row_attr in attr_data.iterrows():
                            attr_code = row_attr["Attribute"].replace("S", "", 1).strip()
                            info = SMART_DEFINITIONS.get(attr_code, {
                                "name": f"S.M.A.R.T. {attr_code}",
                                "desc": "Monitored firmware attribute. Its meaning can vary by manufacturer and drive model.",
                                "cause": "Interpret together with the drive manufacturer's documentation and other diagnostics."
                            })
                            with st.expander(
                                f"Attribute {attr_code}: {info['name']} (Value: {row_attr['Raw Count']:,.0f})"
                            ):
                                st.write(f"**Definition:** {info['desc']}")
                                st.write(f"**Interpretation note:** {info['cause']}")
                        st.caption(
                            "The chart lists non-zero raw values. It is not a per-drive feature attribution "
                            "and does not prove which attribute caused the model to flag the record."
                        )
                    else:
                        st.info("No non-zero SMART raw values were available for this record.")
            else:
                st.success("No records crossed the selected model threshold.")
                st.caption("Not flagged does not guarantee that a drive is healthy or will not fail.")

        except Exception as e:
            st.error(f"Screening could not be completed: {e}")


# ==========================================
# 5. TAB 3: DATASET & AI MODEL
# ==========================================
elif app_mode == "3. Dataset & AI Model":
    st.markdown("<div class='mapua-title'>About Dataset & AI Model</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='mapua-subtitle'>Drive-level dataset extraction, training, and evaluation</div>",
        unsafe_allow_html=True
    )

    st.subheader("Backblaze Dataset")
    st.write(
        "The v4 experiment uses 65,993 unique physical drives and 205 drives with a recorded failure label. "
        "Exact duplicate rows were removed, and each physical drive contributes one snapshot to only one partition."
    )

    partition_df = pd.DataFrame([
        {"Partition": "Training", "Drives": 46194, "Failure Drives": 143, "Healthy Drives": 46051},
        {"Partition": "Validation", "Drives": 6600, "Failure Drives": 21, "Healthy Drives": 6579},
        {"Partition": "Testing", "Drives": 9899, "Failure Drives": 31, "Healthy Drives": 9868},
        {"Partition": "Website Demo", "Drives": 3300, "Failure Drives": 10, "Healthy Drives": 3290},
    ])
    st.dataframe(partition_df, use_container_width=True, hide_index=True)
    st.caption(
        "Failed drives use their failure-labelled terminal-day snapshot. Healthy drives use one seeded random snapshot. "
        "The website-demo drives are excluded from training, validation, and testing."
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.subheader("Preprocessing and Training")
        st.write("\n".join([
            "- 90 SMART fields were available; 56 were retained after removing fields with more than 95% missing values.",
            "- Median imputation and Min-Max scaling were fitted on training data only.",
            "- Two Random Forest configurations were compared: class weighting and SMOTE.",
            "- SMOTE was applied to training data only. Validation data was used to select each decision threshold by F2-score."
        ]))

    with col_b:
        st.subheader("Model Selection")
        st.write(
            "SMOTE Random Forest was selected because it achieved the higher validation F2-score "
            "(0.3937 versus 0.3333). The selected threshold is 0.41. The test set was not used to tune the threshold."
        )
        st.info(
            "The positive label represents the last operational day before a recorded failure. "
            "These results do not establish how many days in advance the model can warn of failure."
        )

    st.subheader("Untouched Test-Set Results")
    metrics_df = pd.DataFrame([
        {
            "Model": "Cost-Sensitive Random Forest",
            "Threshold": 0.43,
            "Accuracy": 0.991615,
            "Precision": 0.138889,
            "Recall": 0.322581,
            "F1": 0.194175,
            "ROC-AUC": 0.827366,
            "Average Precision": 0.145488,
            "TN": 9806, "FP": 62, "FN": 21, "TP": 10,
        },
        {
            "Model": "SMOTE Random Forest (Selected)",
            "Threshold": 0.41,
            "Accuracy": 0.992828,
            "Precision": 0.205882,
            "Recall": 0.451613,
            "F1": 0.282828,
            "ROC-AUC": 0.874771,
            "Average Precision": 0.154362,
            "TN": 9814, "FP": 54, "FN": 17, "TP": 14,
        },
    ])
    st.dataframe(
        metrics_df.style.format({
            "Threshold": "{:.2f}",
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "Average Precision": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True
    )
    st.write(
        "On the 9,899-drive test set, SMOTE identified 14 of the 31 failure-labelled drives and missed 17. "
        "It also flagged 54 healthy-labelled drives. Its precision was 20.59%, meaning that 14 of its 68 flagged "
        "test drives carried the failure-day label."
    )
    st.caption(
        "There were only 31 positive drives in the test set, so the reported metrics have substantial uncertainty. "
        "Accuracy is high in part because most drives are labelled healthy; precision, recall, average precision, "
        "and the confusion matrix are important for interpreting this imbalanced task."
    )


# ==========================================
# 6. TAB 4: PROGRAM RELEVANCE
# ==========================================
elif app_mode == "4. CpE Program Relevance":
    st.markdown("<div class='mapua-title'>Computer Engineering Synthesis</div>", unsafe_allow_html=True)
    st.markdown("<div class='mapua-subtitle'>Bridging Low-Level Hardware Diagnostics with Applied AI</div>",
                unsafe_allow_html=True)

    st.write(
        "DriveGuard directly aligns with the **B.S. Computer Engineering** program by bridging the gap between low-level computer architecture (storage controller diagnostics) and modern machine learning applications. "
    )

    st.subheader("Hardware Telemetry Integration")
    st.write(
        "This project works with S.M.A.R.T. telemetry fields published in the Backblaze dataset, including attributes "
        "that can describe reallocated sectors, pending sectors, temperature, and other drive-reported values. "
        "The current prototype analyzes exported telemetry records; it does not directly read firmware registers."
    )

    st.subheader("Hardware Reliability Prioritization")
    st.write(
        "In computing reliability engineering, **False Negatives** (classifying a failing drive as healthy) result in catastrophic data destruction. "
        "Recall is important because missed failure-labelled drives matter. However, precision and false-positive counts "
        "also matter because excessive alerts can make a screening system difficult to use."
    )