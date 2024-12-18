import streamlit as st

def _deploy_pre_streamlit_config():
    st.set_page_config(
        page_title="Uvais Karni: Technical Portfolio",  # Change the title to match your app's purpose
        page_icon="./assets/ai-file.png",  # Ensure you have a favicon image in the correct path
        layout="wide",  # Using wide layout for more space
        initial_sidebar_state="expanded",  # Expanding sidebar by default
    )

    # Customizing the sidebar and hiding Streamlit default elements (menu and footer)
    hide_streamlit_style = """
    <style>
    /* This is to hide the hamburger menu completely */
    # MainMenu {visibility: hidden;}
    /* This is to hide Streamlit footer */
    footer {visibility: hidden;}
    /* You can also hide the "Get Help" button and other Streamlit UI elements */
    .stApp {padding-top: 0px;}  /* Optional: To remove some padding on top */
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
