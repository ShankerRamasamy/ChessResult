import streamlit as st
import numpy as np 
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt


def main():
    st.title("Chess Results ML Webapp")   
    st.sidebar.write("For more info, please contact:")
    st.sidebar.write("<a href='mailto:shanxr@gmail.com'>Shanker Ramasamy </a>", unsafe_allow_html=True)
    
    st.sidebar.write("The <a href='https://www.kaggle.com/datasnaek/chess/download'>Chess Game Dataset </a>is available for download in Kaggle ", unsafe_allow_html=True)
if __name__ == '__main__':
    main()
    
@st.cache(persist= True)
def load():
    data= pd.read_csv("games.csv")
    data = data.drop(['id','created_at','last_move_at','white_id','black_id','moves'],axis=1)
    labelencoder = LabelEncoder()
    data['winner'] = labelencoder.fit_transform(data['winner'])
    data['increment_code'] = labelencoder.fit_transform(data['increment_code'])
    data['opening_eco'] = labelencoder.fit_transform(data['opening_eco'])
    data['opening_name'] = labelencoder.fit_transform(data['opening_name'])
    return data
df = load()

if st.sidebar.checkbox("Display data", False):
    st.write("This is a set of just over 20,000 games collected from the online chess platform <a href='https://lichess.org/'>*Lichess.org* </a>:sunglasses:", unsafe_allow_html=True)
    st.subheader("Loaded Chess dataset")
    st.write(df)

@st.cache(persist=True)
def split(df):
    X = df.drop('victory_status',axis=1)

    y = df['victory_status']

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
    
    scaler = MinMaxScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)
    
   

    
    return Xtrain_scaled, Xtest_scaled, ytrain, ytest
    
Xtrain_scaled, Xtest_scaled, ytrain, ytest = split(df)




    



st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Random Forest","None"))

if classifier != "None":
    st.sidebar.subheader("Hyperparameters")
    n_estimators= st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=100, key="n_estimators")
    max_depth = st.sidebar.slider("The maximum depth of tree", 10, 100, step =10, key="max_depth")
    
    
    
    
    if st.sidebar.button("Compute", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1 )
        model.fit(Xtrain_scaled, ytrain)
        accuracy = model.score(Xtest_scaled, ytest)
        ypred = model.predict(Xtest_scaled)
        st.write("Accuracy: ", accuracy.round(2))
        confusionMatrix = confusion_matrix(ytest,ypred)
        st.write("Confusion Matrix: ", confusionMatrix)
        st.write('Classification report:')
        report = classification_report(ytest, ypred,output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.write(df)
        
        X = df.drop('victory_status')
        model.feature_importances_

        important_factors = pd.DataFrame({'Factor': list(X.columns), 'Importance': model.feature_importances_})

        important_factors.sort_values(by=['Importance'], ascending=False,inplace=True)

        print(important_factors)
        
        st.write(important_factors)
        

    
st.text("Reference material for creating this webapp from: Dr.Yu Yong Poh, analyticsvidhya.com")
