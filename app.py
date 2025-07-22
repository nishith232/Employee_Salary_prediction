import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.data_prep import load_and_preprocess
from src.model import load_model, predict

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "salary_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "adult.csv")

# Add at the top after imports
st.markdown(
    '''<style>
    body {background: linear-gradient(120deg, #f6d365 0%, #fda085 100%) !important;}
    .main {background-color: #fffbe7 !important; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 2rem;}
    .stButton>button {background: #ff7e5f; color: white; border-radius: 8px; font-weight: bold;}
    .stSidebar {background: #f7cac9 !important;}
    .stMetric {background: #f9f871; border-radius: 8px;}
    .dialog-box {background: #fff; border: 3px solid #ff7e5f; border-radius: 16px; box-shadow: 0 8px 32px rgba(255,126,95,0.15); padding: 2rem; margin: 2rem 0; text-align: center;}
    .dialog-title {font-size: 2.2rem; font-weight: bold; color: #ff7e5f;}
    .dialog-pred {font-size: 2rem; font-weight: bold; color: #43cea2;}
    .dialog-conf {font-size: 1.2rem; color: #333;}
    .header-bar {background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%); color: white; padding: 1.5rem 0 1rem 0; border-radius: 0 0 24px 24px; margin-bottom: 2rem; text-align: center;}
    </style>''', unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor", 
    page_icon="💼",
    layout="wide"
)

# Replace st.title and st.markdown for header
st.markdown('<div class="header-bar"><h1>💼 Employee Salary Predictor</h1><h3>Predict if an employee\'s salary is above or below $50K</h3></div>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Options")

# Add preset examples
st.sidebar.subheader("🚀 Quick Test Examples")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("👔 High Income", help="CEO/Executive profile", key="high_income_btn"):
        st.session_state.use_high_income = True
        
with col2:
    if st.button("👤 Low Income", help="Entry-level profile", key="low_income_btn"):
        st.session_state.use_low_income = True

# Add debug mode toggle and data view in sidebar
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed debugging information")
st.session_state['debug_mode'] = debug_mode
show_sample_data = st.sidebar.checkbox("Show sample data", key="show_sample_checkbox")
show_visualizations = st.sidebar.checkbox("Show data visualizations", key="show_viz_checkbox")

def init_session_state():
    """Initialize session state for all widgets to prevent conflicts"""
    default_values = {
        'age_slider': 35, 'gender_select': 'Male', 'race_select': 'White',
        'workclass_select': 'Private', 'occupation_select': 'Tech-support', 'hours_slider': 40,
        'education_select': 'Bachelors', 'education_num_slider': 13,
        'marital_select': 'Married-civ-spouse', 'relationship_select': 'Husband',
        'capital_gain_input': 0, 'capital_loss_input': 0,
        'native_country_select': 'United-States', 'fnlwgt_input': 200000
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def handle_preset_buttons():
    """Check for preset examples and set session state values"""
    if 'use_high_income' in st.session_state and st.session_state.use_high_income:
        # High income defaults
        st.session_state.age_slider = 45
        st.session_state.education_select = "Masters"
        st.session_state.occupation_select = "Exec-managerial"
        st.session_state.hours_slider = 50
        st.session_state.capital_gain_input = 15000
        st.session_state.marital_select = "Married-civ-spouse"
        st.session_state.relationship_select = "Husband"
        st.session_state.gender_select = "Male"
        st.session_state.race_select = "White"
        st.session_state.workclass_select = "Private"
        st.session_state.education_num_slider = 14
        st.session_state.capital_loss_input = 0
        st.session_state.native_country_select = "United-States"
        st.session_state.fnlwgt_input = 200000
        st.session_state.use_high_income = False
    elif 'use_low_income' in st.session_state and st.session_state.use_low_income:
        # Low income defaults
        st.session_state.age_slider = 22
        st.session_state.education_select = "HS-grad"
        st.session_state.occupation_select = "Handlers-cleaners"
        st.session_state.hours_slider = 25
        st.session_state.capital_gain_input = 0
        st.session_state.marital_select = "Never-married"
        st.session_state.relationship_select = "Own-child"
        st.session_state.gender_select = "Male"
        st.session_state.race_select = "White"
        st.session_state.workclass_select = "Private"
        st.session_state.education_num_slider = 9
        st.session_state.capital_loss_input = 0
        st.session_state.native_country_select = "United-States"
        st.session_state.fnlwgt_input = 150000
        st.session_state.use_low_income = False

# Initialize and handle state
init_session_state()
handle_preset_buttons()

# --- Main Form for User Input ---
with st.form(key="employee_form"):
    st.subheader("📝 Enter Employee Details")
    
    # Use columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal")
        age = st.slider("Age", 17, 90, key="age_slider")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender_select")
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], key="race_select")
        
        st.subheader("👨‍👩‍👧‍👦 Family")
        marital_list = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
        marital_status = st.selectbox("Marital Status", marital_list, key="marital_select")
        relationship_list = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
        relationship = st.selectbox("Relationship", relationship_list, key="relationship_select")

    with col2:
        st.subheader("💼 Work")
        workclass = st.selectbox("Work Class", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"], key="workclass_select")
        occupation_list = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        occupation = st.selectbox("Occupation", occupation_list, key="occupation_select")
        hours_per_week = st.slider("Hours per Week", 1, 99, key="hours_slider")

        st.subheader("🎓 Education")
        education_list = ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"]
        education = st.selectbox("Education Level", education_list, key="education_select")
        education_num = st.slider("Years of Education", 1, 16, key="education_num_slider")
    
    with col3:
        st.subheader("💰 Financial & Other")
        capital_gain = st.number_input("Capital Gain", 0, 100000, step=100, key="capital_gain_input")
        capital_loss = st.number_input("Capital Loss", 0, 5000, step=50, key="capital_loss_input")
        country_list = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
        native_country = st.selectbox("Native Country", country_list, key="native_country_select")
        fnlwgt = st.number_input("Final Weight (Census)", 12285, 1484705, key="fnlwgt_input")

    # Form submission button
    submitted = st.form_submit_button("✨ Predict Salary")


# --- Prediction and Results Display ---
if submitted:
    # Create input dictionary from form values
    data = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "educational-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }
    input_df = pd.DataFrame([data])

    # Display user input
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📋 User Input Parameters")
        display_df = input_df.T
        display_df.columns = ['Value']
        display_df['Value'] = display_df['Value'].astype(str)
        st.write(display_df)

    with col2:
        st.subheader("📊 Quick Stats")
        st.metric("Age", input_df['age'].iloc[0])
        st.metric("Hours/Week", input_df['hours-per-week'].iloc[0])
        st.metric("Education Level", input_df['educational-num'].iloc[0])
        
        if input_df['educational-num'].iloc[0] >= 13 and input_df['hours-per-week'].iloc[0] >= 40:
            st.success("Higher income indicators")
        elif input_df['educational-num'].iloc[0] <= 9 and input_df['hours-per-week'].iloc[0] <= 30:
            st.info("Lower income indicators")

    # Load model and make prediction
    try:
        if debug_mode:
            st.subheader("🔍 Current Input Values (Debug)")
            st.write(input_df)

        model_package = load_model(MODEL_PATH)
        result = predict(model_package, input_df)

        # Display prediction results in a dialog box
        st.subheader("🎯 Prediction Results")
        col1, col2 = st.columns([2, 1])
        with col1:
            if result['prediction'] == ">50K":
                st.markdown(f'''<div class="dialog-box">
                    <div class="dialog-title">💰 Predicted Salary</div>
                    <div class="dialog-pred">{result['prediction']}</div>
                    <div class="dialog-conf">Confidence: {result['confidence']:.1f}%</div>
                </div>''', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'''<div class="dialog-box">
                    <div class="dialog-title">Predicted Salary</div>
                    <div class="dialog-pred">{result['prediction']}</div>
                    <div class="dialog-conf">Confidence: {result['confidence']:.1f}%</div>
                </div>''', unsafe_allow_html=True)
        
        with col2:
            # Probability chart
            prob_data = pd.DataFrame({
                'Income Level': list(result['probabilities'].keys()),
                'Probability': list(result['probabilities'].values())
            })
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(prob_data['Income Level'], prob_data['Probability'], color=['#ff7e5f', '#43cea2'])
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            ax.set_ylim(0, 1)
            for bar, prob in zip(bars, prob_data['Probability']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{prob:.3f}', ha='center', va='bottom')
            st.pyplot(fig)
            
        # Model information expander
        with st.expander("🔍 Model Information"):
            st.write(f"**Model Type:** {model_package.get('model_name', 'Random Forest')}")
            st.write(f"**Model Accuracy:** {model_package.get('test_accuracy', 0.86):.4f}")
            st.write(f"**Features Used:** {len(model_package['feature_columns'])}")

    except FileNotFoundError:
        st.error("🔗 **Model Not Found!** Please ensure `model/salary_model.pkl` exists.")
    except Exception as e:
        st.error(f"**An error occurred:** {e}")
        if debug_mode:
            import traceback
            st.code(traceback.format_exc())

# Load and display sample data if requested
if show_sample_data:
    try:
        X, y = load_and_preprocess(DATA_PATH)
        st.subheader("📁 Dataset Sample")
        sample_data = pd.concat([X.head(), y.head()], axis=1)
        st.write(sample_data)
        
        # Dataset statistics
        st.subheader("📈 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(X))
        with col2:
            st.metric("Features", len(X.columns))
        with col3:
            st.metric(">50K Count", (y == ">50K").sum())
        with col4:
            st.metric("≤50K Count", (y == "<=50K").sum())
            
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Additional visualizations if requested
if show_visualizations:
    try:
        X, y = load_and_preprocess(DATA_PATH)
        
        st.subheader("📊 Data Visualizations")
        
        # Income distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            y.value_counts().plot(kind='bar', ax=ax, color=['lightcoral', 'lightgreen'])
            ax.set_title('Income Distribution')
            ax.set_xlabel('Income Level')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            X['age'].hist(bins=30, ax=ax, alpha=0.7, color='skyblue')
            ax.set_title('Age Distribution')
            ax.set_xlabel('Age')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")

# Footer
st.markdown("---")
st.markdown("**Note:** This prediction is based on the Adult Census Income dataset and should be used for educational purposes only.")
