# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from helper import train_models, comparison, weightage, churn, countplot
from preprocessor import preprocess

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

st.markdown("""
### 📁 Upload your CSV file (Telco Customer Churn format)
**Expected columns:**  
`customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`,  
`tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,  
`OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,  
`Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        
        # Display dataset info
        st.success("✅ File uploaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            if "Churn" in df.columns:
                churn_rate = (df["Churn"].value_counts().get("Yes", 0) / len(df)) * 100
                st.metric("Churn Rate", f"{churn_rate:.2f}%")
        
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📊 Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Data Types:**")
            st.dataframe(pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values.astype(str)
            }), use_container_width=True)
        
        with col2:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            if missing_df['Missing Count'].sum() == 0:
                st.success("✅ No missing values found!")
        
        st.markdown("---")
        
        # =========================
        # 🔹 Exploratory Data Analysis (EDA)
        # =========================
        st.markdown("## 🧠 Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            show_churn = st.checkbox("📈 Show Churn Distribution & Correlation")
        with col2:
            show_countplots = st.checkbox("📊 Show Count Plots for Categorical Features")

        # -------------------------------
        # Churn & Correlation Plots
        # -------------------------------
        if show_churn:
            with st.spinner("Generating churn & correlation plots..."):
                # Create figures
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.countplot(x=df["Churn"], palette="Set2", ax=ax1)
                ax1.set_title("Churn Distribution")
                ax1.set_xticklabels(["No", "Yes"])

                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax2)
                ax2.set_title("Correlation Heatmap")

                # Display plots in columns
                col_churn, col_corr = st.columns(2)
                with col_churn:
                    st.pyplot(fig1)
                with col_corr:
                    st.pyplot(fig2)

                plt.close("all")

        # -------------------------------
        # Count Plots for Categorical Features
        # -------------------------------
        if show_countplots:
            st.info("Generating count plots for categorical features...")
            progress = st.progress(0)

            # Get categorical columns
            object_cols = df.select_dtypes(include="object").columns.to_list()

            # Include 'SeniorCitizen' if numeric but categorical
            if "SeniorCitizen" not in object_cols and "SeniorCitizen" in df.columns:
                object_cols = ['SeniorCitizen'] + object_cols

            # Remove customerID safely
            if 'customerID' in object_cols:
                object_cols.remove('customerID')
            if 'TotalCharges' in object_cols:
                object_cols.remove('TotalCharges')

            total = len(object_cols)

            for idx, col_name in enumerate(object_cols):
                counts = df[col_name].value_counts().sort_index()

                # Create a bar plot with controlled width & size
                fig, ax = plt.subplots(figsize=(8, 3.5))  # wider figure for better spacing
                ax.bar(
                    counts.index.astype(str),
                    counts.values,
                    width=0.4,  # 🔧 adjust bar width here (0.4–0.8 looks good)
                    color=sns.color_palette("coolwarm", len(counts))
                )
                ax.set_title(f"{col_name} Count Plot", fontsize=12, fontweight='bold')
                ax.set_xlabel(col_name)
                ax.set_ylabel("Count")

                # Rotate and align category labels
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                # Show the plot in Streamlit
                st.pyplot(fig)
                plt.close(fig)

                # Update progress bar
                progress.progress((idx + 1) / total)

            st.success("✅ Count plots generated successfully!")


        # =========================
        # 🔹 Train Models
        # =========================
        st.markdown("## 🤖 Train Machine Learning Models")
        st.info("💡 Click the button below to train multiple ML models and compare their performance.")
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a minute ⏳"):
                try:
                    metrics_dict, best_model_name = train_models(df)
                    
                    st.balloons()
                    st.success(f"✅ Training Complete! Best Model: **{best_model_name}** (ROC-AUC: {metrics_dict[best_model_name]['roc_auc']:.4f})")
                    
                    # Store in session state
                    st.session_state['metrics_dict'] = metrics_dict
                    st.session_state['best_model_name'] = best_model_name
                    st.session_state['models_trained'] = True
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
        
        # Display results if models are trained
        if 'models_trained' in st.session_state and st.session_state['models_trained']:
            metrics_dict = st.session_state['metrics_dict']
            best_model_name = st.session_state['best_model_name']
            
            st.markdown("---")
            
            # 📊 Model performance comparison table
            st.subheader("📋 Model Performance Metrics")
            metrics_df = pd.DataFrame(metrics_dict).T.drop(columns=["report", "conf_matrix"])
            metrics_df = metrics_df.round(4)
            st.dataframe(
                metrics_df.style.background_gradient(cmap="Blues", subset=metrics_df.columns).format("{:.4f}"),
                use_container_width=True
            )
            
            # 📈 Visual comparison (bar charts)
            st.subheader("📊 Visual Performance Comparison")
            fig_comparison = comparison(metrics_dict)
            st.pyplot(fig_comparison)
            plt.close(fig_comparison)
            
            # 🧩 Confusion Matrices
            st.subheader("🧩 Confusion Matrices")
            cols = st.columns(2)
            for idx, (name, vals) in enumerate(metrics_dict.items()):
                with cols[idx % 2]:
                    st.write(f"**{name}**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(vals["conf_matrix"], annot=True, fmt="d", 
                               cmap="coolwarm", ax=ax, cbar_kws={'label': 'Count'})
                    ax.set_xlabel("Predicted", fontsize=11)
                    ax.set_ylabel("Actual", fontsize=11)
                    ax.set_title(f"{name} Confusion Matrix", fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            
            st.markdown("---")
            
            # =========================
            # 🔹 Feature Importance
            # =========================
            st.subheader("📌 Feature Importance Analysis")
            
            selected_model = st.selectbox(
                "🔍 Select model to view feature importance:",
                list(metrics_dict.keys()),
                key="feature_importance_select"
            )
            
            if st.button("📊 Show Feature Importance", key="show_importance"):
                model_file = f"{selected_model.replace(' ', '_').lower()}_model.pkl"
                try:
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                    
                    _, X_train_smote, _, _, _ = preprocess(df)
                    result = weightage(model, X_train_smote)
                    
                    if result:
                        importance_df, fig_weight = result
                        st.write("### 🏅 Top 15 Most Important Features")
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.dataframe(
                                importance_df.head(15).style.background_gradient(cmap="Greens"),
                                use_container_width=True
                            )
                        with col2:
                            st.pyplot(fig_weight)
                            plt.close(fig_weight)
                    else:
                        st.warning("⚠️ This model does not support feature importance visualization.")
                        
                except Exception as e:
                    st.error(f"❌ Could not load model or plot feature importance: {str(e)}")
            
            st.markdown("---")
            
            # =========================
            # 🔹 Make Predictions
            # =========================
            st.subheader("🔮 Make Predictions on Dataset")
            
            selected_model_name = st.selectbox(
                "🎯 Select model for predictions:",
                list(metrics_dict.keys()),
                key="pred_model"
            )
            
            if st.button("🧠 Generate Predictions", type="primary", key="predict_button"):
                try:
                    model_file = f"{selected_model_name.replace(' ', '_').lower()}_model.pkl"
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                    
                    # Load encoders
                    with open("encoders.pkl", "rb") as f:
                        encoders = pickle.load(f)
                    
                    # Preprocess the data for prediction
                    df_pred = df.copy()
                    if "customerID" in df_pred.columns:
                        customer_ids = df_pred["customerID"].copy()
                        df_pred.drop(columns="customerID", inplace=True)
                    else:
                        customer_ids = None
                    
                    if "TotalCharges" in df_pred.columns:
                        df_pred["TotalCharges"] = df_pred["TotalCharges"].replace({" ": "0.0", "": "0.0"})
                        df_pred["TotalCharges"] = pd.to_numeric(df_pred["TotalCharges"], errors='coerce').fillna(0.0)
                    
                    # Encode categorical columns
                    for col in df_pred.select_dtypes(include="object").columns:
                        if col != "Churn" and col in encoders:
                            df_pred[col] = encoders[col].transform(df_pred[col].astype(str))
                    
                    X_pred = df_pred.drop(columns=["Churn"], errors="ignore")
                    preds = model.predict(X_pred)
                    probs = model.predict_proba(X_pred)[:, 1]
                    
                    result_df = pd.DataFrame({
                        "Prediction": ["Churn" if p == 1 else "No Churn" for p in preds],
                        "Churn Probability": (probs * 100).round(2)
                    })
                    
                    if customer_ids is not None:
                        result_df.insert(0, "Customer ID", customer_ids.values)
                    
                    st.success(f"✅ Generated {len(result_df)} predictions using {selected_model_name}")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(result_df))
                    with col2:
                        churn_count = (preds == 1).sum()
                        st.metric("Predicted Churns", churn_count)
                    with col3:
                        churn_pct = (churn_count / len(preds)) * 100
                        st.metric("Churn Rate", f"{churn_pct:.2f}%")
                    
                    st.write("### 📄 Prediction Results (showing first 20 rows):")
                    st.dataframe(
                        result_df.head(20).style.background_gradient(cmap="RdYlGn_r", subset=["Churn Probability"]),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_{selected_model_name}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"⚠️ Error while predicting: {str(e)}")
            
            st.markdown("---")
            st.info("✅ All trained models and encoders are saved as `.pkl` files in your working directory.")
        
    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded CSV file is empty or invalid!")
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload a CSV file to begin the analysis and model training.")
    
    # Show example data format
    with st.expander("📖 View Expected Data Format"):
        st.markdown("""
        Your CSV should contain customer churn data with columns like:
        - **customerID**: Unique identifier
        - **Demographic info**: gender, SeniorCitizen, Partner, Dependents
        - **Service info**: tenure, PhoneService, InternetService, etc.
        - **Billing info**: Contract, PaymentMethod, MonthlyCharges, TotalCharges
        - **Target**: Churn (Yes/No)
        """)