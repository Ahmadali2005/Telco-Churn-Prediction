import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title(" Telco Customer Churn Predictor")
st.write("أدخل بيانات العميل للتنبؤ باحتمالية إلغائه للاشتراك:")


@st.cache_resource
def load_assets():
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('model1.pkl')
    return preprocessor, model

preprocessor, model = load_assets()

st.sidebar.header("بيانات العميل")

gender = st.sidebar.selectbox("الجنس(Gender)", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("كبار السن (Senior Citizen)", [0, 1])
Partner = st.sidebar.selectbox("مرتبط (Partner)", ["Yes", "No"])
Dependents = st.sidebar.selectbox("يعول أفراد (Dependents)", ["Yes", "No"])
tenure = st.sidebar.slider("مدة الاشتراك بالأشهر (Tenure)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("خدمة الهاتف (Phone Service)", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("خطوط متعددة (Multiple Lines)", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("نوع الإنترنت (Internet Service)", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("أمان الإنترنت (Online Security)", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("نسخ احتياطي (Online Backup)", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("حماية الأجهزة (Device Protection)", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("الدعم الفني (Tech Support)", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("بث التلفزيون (Streaming TV)", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("بث الأفلام (Streaming Movies)", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("نوع العقد (Contract)", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("فواتير إلكترونية (Paperless Billing)", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("طريقة الدفع (Payment Method)", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])

MonthlyCharges = st.sidebar.number_input("الرسوم الشهرية (Monthly Charges)", value=70.0)
TotalCharges = st.sidebar.number_input("إجمالي الرسوم (Total Charges)", value=tenure * MonthlyCharges)


input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}])


if st.button("توقع حالة العميل"):
    
    transformed_data = preprocessor.transform(input_data)
    
    
    prediction = model.predict(transformed_data)[0]
    probability = model.predict_proba(transformed_data)[0][1] * 100

    st.markdown("---")
    st.subheader("نتيجة التنبؤ:")
    
    if prediction == 1:
        st.error(f" **العميل عرضة لإلغاء الاشتراك (Churn)!**")
        st.write(f"نسبة احتمال المغادرة: **{probability:.1f}%**")
        st.warning("**توصية:** يُفضل تقديم عرض خاص أو خصم على العقود الطويلة للحفاظ على العميل.")
    else:
        st.success(f" **العميل مستمر مع الشركة (No Churn).**")
        st.write(f"نسبة احتمال المغادرة: **{probability:.1f}%**")