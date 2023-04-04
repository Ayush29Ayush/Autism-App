from typing import Any, List, Union
from uuid import uuid4
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
import json
# le = LabelEncoder()
# le.fit_transform(pd.read_csv)

# le = LabelEncoder()
cols_to_transform = ['Ethnicity','Family_mem_with_ASD', 'Sex', 'Jaundice']
ref_data = pd.read_csv("Toddler Autism dataset July 2018.csv")

decision_map_reverse = {1: "Yes", 0: "No"}
decision_map = {"Yes": 1, "No": 0}

eth_index = {
        0: 'Hispanic',
        1: 'Latino',
        2: 'Native Indian',
        3: 'Others',
        4: 'Pacifica',
        5: 'White European',
        6: 'asian',
        7: 'black',
        8: 'middle eastern',
        9: 'mixed',
        10: 'south asian',
    }

def encode(data: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray, Any]:
    global ref_data

    columns: List[str] = [
        # 'Ethnicity',
        'Family_mem_with_ASD', 'Sex', 'Jaundice']
    col_dict = {
        # "Ethnicity": ['middle eastern','White European','Hispanic','black','asian', 'south asian', 'Native Indian', 'Others', 'Latino', 'mixed', 'Pacifica'],
        "Family_mem_with_ASD": ["no", "yes"],
        "Sex": ["f", "m"],
        "Jaundice": ["no", "yes"],
    }

    # data["Ethnicity"] = data['Ethnicity'].apply(lambda x: eth_index[x])
    for col in columns:
        le = LabelEncoder()
        le.fit(ref_data[col])
        data[col] = le.transform(data[col])
    
    return data


def load_models() -> List[Any]:
    return pickle.loads(open("./models.pkl", "rb").read())



if __name__ == "__main__":
    print(load_models())

st.write("""
# Autism Detection App

Check this app out to see if someone has autistic tendencies or undiagnosed autism.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown(f"""
## Guide:
- 1 indicates 'Yes', while 0 indicates 'No'
- For Ethnicity, refer the following map:
    
    {json.dumps(eth_index, indent=4)} 


""")

columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons',
    'Qchat-10-Score', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']

# Collects user input features into dataframe
def user_input_features():
    global columns
    
    features = [0 for _ in range(17)]
    features[0] = st.sidebar.selectbox("I often notice small sounds when others do not.", (0, 1))
    features[1] = st.sidebar.selectbox("When I’m reading a story, I find it difficult to work out the characters’ intentions.", (0, 1))
    features[2] = st.sidebar.selectbox("I find it easy to read between the lines when someone is talking to me.", (0, 1))
    features[3] = st.sidebar.selectbox("I usually concentrate more on the whole picture, rather than the small details.", (0, 1))
    features[4] = st.sidebar.selectbox("I know how to tell if someone listening to me is getting bored.", (0, 1))
    features[5] = st.sidebar.selectbox("I find it easy to do more than one thing at once.", (0, 1))
    features[6] = st.sidebar.selectbox("I find it easy to work out what someone is thinking or feeling just by looking at their face.", (0, 1))
    features[7] = st.sidebar.selectbox("If there is an interruption, I can switch back to what I was doing very quickly.", (0, 1))
    features[8] = st.sidebar.selectbox("I like to collect information about categories of things.", (0, 1))
    features[9] = st.sidebar.selectbox("I find it difficult to work out people’s intentions.", (0, 1))
    features[10] = st.sidebar.slider("Age in Months", min_value=0, max_value=50, step=1)
    features[11] = sum([x for x in features[:10]])
    features[12] = st.sidebar.selectbox("Sex", ("m", "f"))
    features[13] = st.sidebar.selectbox(f"Ethnicity:", (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ))
    features[14] = st.sidebar.selectbox("Jaundice", ("no", "yes"))
    # features[15] = st.sidebar.selectbox("Q14", ("no", "yes"))
    features[15] = st.sidebar.selectbox("Are there any family members with ASD?", ("yes", "no"))
    
    df_dict = {"Age_Mons": features[10]}
    df_dict = df_dict | {elem: features[idx] for idx, elem in enumerate(columns)}
    df = encode(pd.DataFrame(df_dict, index=[0]))

    # features = pd.DataFrame(data, index=[0])
    return df

df = user_input_features()
df = df[columns]


# Displays the user input features
st.subheader('User Input features')
st.write(df)

# Reads in saved classification model
models = pickle.load(open('models.pkl', 'rb'))

# Apply model to make predictions
predictions = [model[1].predict(df[columns]).tolist() for model in models]
# prediction_proba = [model.predict_proba(df) for model in models]


st.subheader('Prediction')
penguins_species = np.array(["Autism?"])

for idx, prediction in enumerate(predictions):
    st.write(f"According to the {models[idx][0][:-1]} model, do you have autism?  {decision_map_reverse[prediction[0]]}")

# st.subheader('Prediction Probability')
# st.write(prediction_proba)
