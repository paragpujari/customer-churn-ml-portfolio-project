"""
Customer Churn Prediction - Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.joblib"


@st.cache_resource
def load_model():
    """Load the trained pipeline and metadata"""
    if not MODEL_PATH.exists():
        st.error("Model file not found! Please run the training script first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH) if META_PATH.exists() else {}
    return model, meta


def main():
    st.title("📉 Customer Churn Prediction System")
    st.markdown("""
    **End-to-End Machine Learning Portfolio Project**  
    Predict whether a telecom customer is likely to churn based on their demographics, 
    account information, and service usage.
    """)
    
    model, meta = load_model()
    
    # Sidebar - About
    with st.sidebar:
        st.header("ℹ️ About this App")
        st.markdown("""
        This interactive app uses a **Random Forest Classifier** trained on the 
        Telco Customer Churn dataset.
        
        **Key Features used:**
        - Demographics (Gender, Senior Citizen, Partner, Dependents)
        - Account info (Tenure, Contract, Payment Method, Paperless Billing)
        - Services (Phone, Internet, Security, Support, Streaming)
        - Charges (Monthly & Total)
        
        **Model Performance (Test Set):**
        """)
        if meta.get('metrics'):
            m = meta['metrics']
            st.metric("F1 Score", f"{m.get('f1', 0):.3f}")
            st.metric("Recall (Churn)", f"{m.get('recall', 0):.3f}")
            st.metric("ROC-AUC", f"{m.get('roc_auc', 0):.3f}")
            st.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
        
        st.markdown("---")
        st.markdown("**How to use:** Fill in the customer details on the left and click **Predict Churn**.")
        st.markdown("Built with ❤️ using Scikit-learn + Streamlit")
    
    # Main form
    st.header("🔍 Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No")
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    
    with col2:
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col3:
        st.subheader("Services & Charges")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)
    
    st.subheader("Additional Services")
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col5:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with col6:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    # Predict button
    st.markdown("---")
    if st.button("🚀 Predict Churn", type="primary", use_container_width=True):
        # Build input dataframe (must match training columns exactly)
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless],
            'PaymentMethod': [payment],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        try:
            # Prediction
            proba = model.predict_proba(input_data)[0]
            prediction = model.predict(input_data)[0]
            churn_prob = proba[1]
            
            # Display results
            st.markdown("## 📊 Prediction Result")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK OF CHURN**")
                    st.markdown(f"**Probability of Churning:** `{churn_prob*100:.1f}%`")
                else:
                    st.success("✅ **LOW RISK OF CHURN**")
                    st.markdown(f"**Probability of Churning:** `{churn_prob*100:.1f}%`")
            
            with res_col2:
                st.progress(float(churn_prob))
                st.caption(f"Churn Probability: {churn_prob:.1%}")
            
            # Interpretation
            st.markdown("### 💡 Interpretation & Recommendations")
            if churn_prob > 0.6:
                st.warning("""
                **High churn risk detected.** Recommended actions:
                - Offer a loyalty discount or contract upgrade incentive
                - Reach out via preferred channel for retention offer
                - Review service quality / recent support tickets
                - Consider free month of premium features
                """)
            elif churn_prob > 0.4:
                st.info("""
                **Moderate risk.** Monitor this customer:
                - Send personalized engagement email
                - Highlight unused features that could increase stickiness
                - Survey satisfaction score
                """)
            else:
                st.success("""
                **Healthy customer.** Maintain relationship:
                - Continue excellent service
                - Occasional upsell of relevant add-ons
                - Include in loyalty program communications
                """)
            
            # Show input summary
            with st.expander("View Input Summary"):
                st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
        End-to-End ML Portfolio Project • Data: Telco Customer Churn • 
        Model: Random Forest Pipeline • Deployed with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
