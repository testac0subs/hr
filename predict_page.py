import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_department = data["le_department"]
le_education = data["le_education"]

def show_predict_page():
    st.title("HR Analytics")

    st.write("""### Enter the details of the employee""")

    department = (
        "Sales & Marketing",
        "Operations",
        "Technology",
        "Procurement",
        "Analytics",
        "Finance",
        "HR",
        "Legal",
        "R&D",
    )

    education = (
        "Less than a Bachelors",
        "Bachelors degree",
        "Masters degree",
    )
    length_of_service = (
        0,1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
    )
    no_of_trainings = (0,1,2,3,4,5,6,7,8,9,10)
    KPIs_met = (0,1)
    awards_won = (0,1)

    employee_id = st.number_input('Employee ID',min_value=1111, max_value=2000, value=1827, step=1)
    department = st.selectbox("Department", department)
    education = st.selectbox("Education Level", education)
    no_of_trainings = st.selectbox("Numbero of trainings", no_of_trainings)
    previous_year_rating = st.slider("Previous year rating",0, 1)
    length_of_service = st.selectbox("Length of service", length_of_service)
    KPIs_met = st.selectbox("Key performance indicator", KPIs_met)
    awards_won = st.selectbox("Awards won", awards_won)
    avg_training_score = st.number_input('Average training score',min_value=39, max_value=99, value=50, step=1)

    ok = st.button("Predict Outcome")
    if ok:
        # X = np.array([[country, education, expericence ]])
        X = np.array([[department,education,no_of_trainings,previous_year_rating,
        length_of_service,KPIs_met,awards_won,avg_training_score]])

        X[:, 0] = le_department.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(int)

        outcome = int(regressor.predict(X))
        if(outcome==0):
            str = "should be trained more"
        else:
            str = "can be promoted"
        st.subheader(f"The employee {employee_id} {str}")