import streamlit as st
import joblib
import numpy as np

# Set the page configuration
st.set_page_config(page_title="Portfolio App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Contact Us", "AI Project"])

# Homepage
if page == "Homepage":
    st.title("Welcome to My Portfolio!")
    st.image("https://w0.peakpx.com/wallpaper/607/199/HD-wallpaper-evening-pic-natura-thumbnail.jpg", caption="Your Amazing Banner Image", use_column_width=True)
    st.write("""
    Hi! I'm [Your Name], a passionate [Your Profession] with expertise in [Your Skills or Specializations].
    
    ### About Me
    I specialize in:
    - Data Science
    - Machine Learning
    - Web Development

    ### My Projects
    1. **Project 1**: [Description or Link]
    2. **Project 2**: [Description or Link]
    3. **Project 3**: [Description or Link]

    Feel free to explore and get in touch with me!
    """)

# Contact Us
elif page == "Contact Us":
    st.title("Contact Me")
    st.write("""
    I'd love to hear from you! Please fill out the form below or use the provided contact details to get in touch.
    """)
    
    # Contact Form
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            st.success("Thank you for reaching out! I'll get back to you soon.")
    
    # Additional Contact Info
    st.write("### Alternatively, you can reach me at:")
    st.write("- Email: pawat.pps@gmail.com")
    st.write("- LinkedIn: [Your LinkedIn Profile](https://linkedin.com)")
    st.write("- GitHub: [ppakpise](https://github.com/ppakpise)")

if page == "AI Project":
    st.title("My AI Project!")
    # Load the trained model
    model = joblib.load("titanic_model.pkl")

    # Input fields for user to provide passenger details
    st.header("Enter Passenger Details:")

    Pclass = st.selectbox("Passenger Class (Pclass):", [1, 2, 3], index=2)
    Sex = st.selectbox("Gender (Sex):", ["Male", "Female"])
    Sex = 0 if Sex == "Male" else 1
    Age = st.slider("Age:", min_value=0, max_value=100, value=30)
    SibSp = st.number_input("Number of Siblings/Spouses Aboard (SibSp):", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Number of Parents/Children Aboard (Parch):", min_value=0, max_value=10, value=0)
    Fare = st.number_input("Ticket Fare (Fare):", min_value=0.0, value=10.0, step=1.0)
    Embarked = st.selectbox("Port of Embarkation (Embarked):", ["Cherbourg", "Queenstown", "Southampton"])
    Embarked = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[Embarked]

    # Predict button
    if st.button("Predict Survival"):
        inputs = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
        # Make a prediction
        prediction_0 = model.predict(inputs)
        probability_0 = model.predict_proba(inputs)
        prediction = prediction_0[0]
        probability = probability_0[0][1]

        # Display the results
        if prediction == 1:
            st.success(f"The passenger is predicted to SURVIVE with a probability of {probability:.2f}.")
        else:
            st.error(f"The passenger is predicted NOT to survive with a probability of {1 - probability:.2f}.")

    # Footer
    st.write("### Note:")
    st.write("This prediction is based on a machine learning model trained on the Titanic dataset and may not reflect real-world outcomes.")



