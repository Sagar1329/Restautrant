import streamlit as st
from PIL import  Image

img=Image.open('images/title.jpg')

st.image(img,use_column_width=True)


st.title('Restaurant Review')
import pandas as pd 
df=pd.read_csv('https://raw.githubusercontent.com/Sagar1329/restautrant/main/Restaurant_Reviews.tsv',sep='\t')
x=df['Review'].values
y=df['Liked'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

#METHOD 2 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
text_model = make_pipeline(CountVectorizer(),SVC())

text_model.fit(x_train,y_train)
ip=st.text_input('Please provide feedback ')
y_pred1 = text_model.predict([ip])
if st.button('Submit the feedback'):
  if y_pred1==1:
    st.title('POSITIVE')
    st.subheader("Thanks for your feedback,Glad that you liked our restaurent")
  else:
    st.title("NEGATIVE")
    st.subheader("Thanks for the feedback,we hope to improve and provide good expirence for you next time ")
