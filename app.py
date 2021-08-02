import os
import sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join('./pages')))
import home
import data
import insights
import prediction


PAGES = {
  "Home": home,
  "Data": data,
  "Insights": insights,
  "Prediction": prediction
}

selection = st.sidebar.radio("Go to page", list(PAGES.keys()))
page = PAGES[selection]
page.app()
