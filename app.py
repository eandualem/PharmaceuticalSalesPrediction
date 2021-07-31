import os
import sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join('./pages')))
import page1
import page2


PAGES = {
  "Home": page1,
  "Dashboard": page2,
}

selection = st.sidebar.radio("Go to page", list(PAGES.keys()))
page = PAGES[selection]
page.app()
