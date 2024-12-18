import streamlit as st
from lib._deploy_config import _deploy_pre_streamlit_config


def _config_sidebar_main_page():
    pass


def main():
    _config_sidebar_main_page()

    # Markdown for page styling and structure
    st.markdown(
        """
        <style>
            .header {
                font-size: 45px;
                text-align: center;
                font-weight: bold;
                color: #2E2B5F;
                margin-top: 40px;
                font-family: 'Helvetica', sans-serif;
            }

            .subheader {
                font-size: 25px;
                color: #2E2B5F;
                text-align: center;
                margin-top: 10px;
                font-family: 'Arial', sans-serif;
            }

            .section {
                font-size: 18px;
                color: #444;
                text-align: left;
                padding-left: 20px;
                margin-top: 20px;
            }

            .content {
                font-size: 16px;
                color: #666;
                text-align: justify;
                padding-left: 20px;
                padding-right: 20px;
            }

            .image-container {
                text-align: center;
                margin-top: 40px;
            }

            .image-container img {
                border-radius: 50%;
                width: 200px;
                height: 200px;
                border: 5px solid #4B0082;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
            }

            .fun-fact {
                font-size: 16px;
                color: #999;
                text-align: center;
                margin-top: 30px;
                font-style: italic;
            }

            .links {
                font-size: 16px;
                color: #4B0082;
                text-align: center;
                margin-top: 20px;
            }

            .links a {
                text-decoration: none;
                color: #4B0082;
                font-weight: bold;
            }

            .links a:hover {
                color: #6A5ACD;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # Header and Profile Picture
    st.markdown(
        """
        <div class="header">
            Welcome to My AI Journey 🚀
        </div>
        """, unsafe_allow_html=True
    )

    st.image("./assets/Profile_Pic.jpg", width=300)

    # About Me Section
    st.markdown("""
        ## About Me

        Hi, I'm **Uvais Karni**, an AI Developer with a passion for **Machine Learning**, **Computer Vision**, and **Embedded Systems**. 
        I am committed to creating scalable and efficient solutions that push the boundaries of technology and solve real-world problems. 
        Whether it's **automating processes** or building **AI-powered systems**, I strive to deliver innovative solutions that drive progress.
        """)

    # # Background Section
    # st.markdown("""
    #     ## 📄 Background

    #     I hold a **Master’s in Computer Science** with a specialization in **Artificial Intelligence** and **Image Analysis** from **Uppsala University**. 
    #     My career journey has involved working on diverse projects, ranging from **multi-object tracking systems** for **automotive applications** 
    #     to **AI-driven recruitment assistants**. My work spans **real-time video processing**, **predictive modeling**, and **cloud deployment**, 
    #     where I leverage AI and machine learning technologies to build scalable systems that meet the challenges of modern industries.
    #     """)

    # # Skills Section
    # st.markdown("""
    #     ## 💻 Skills

    #     Here's the tech I use to create all these AI wonders:

    #     - **Programming**: Python, C++, R, MATLAB (The essential languages for AI and ML development)
    #     - **Computer Vision**: Object Detection, Segmentation, Multi-Object Tracking (Tools: OpenCV, GStreamer, TensorFlow)
    #     - **Machine Learning**: Supervised/Unsupervised Learning, Predictive Insights (Frameworks: TensorFlow, PyTorch, OpenVINO, ONNX)
    #     - **Cloud & Data Infrastructure**: AWS (ECR, EKS, Redshift), OpenStack, Big Data (Spark, Hadoop)
    #     - **Software Development**: CI/CD, Docker, Git, Streamlit, Flask (For backend development and automation)
    #     """)

    # # Projects Section
    # st.markdown("""
    #     ## 🔧 Projects

    #     ### **AI-Powered Interview Assistant (Ongoing)**
    #     I built a **RAG framework** with **LLM** to help assess **candidate fit** by answering questions from their profiles. This AI assistant streamlines the recruitment process, making it easier for recruiters to evaluate candidates' experience and qualifications more efficiently.

    #     ### **Multi-Hand Gesture Recognition Using Faster R-CNN**
    #     Developed a **real-time multi-hand gesture recognition system** using **Faster R-CNN** to provide an interactive and hands-free user experience, perfect for **gaming, robotics**, or **virtual environments**.

    #     ### **Reinforcement Learning in Unity 3D**
    #     Trained AI agents using the **PPO algorithm** for **reinforcement learning** in **Unity 3D**. This project demonstrates how AI can be applied to create interactive agents capable of learning from their environment.

    #     ### **Registration Plate Localization & Character Classification**
    #     Designed an **AI model** to **localize registration plates** and **classify characters**, for applications like **smart parking systems** and **vehicle identification**.
    #     """)

    # Education Section
    st.markdown("""
        ## 📚 Education

        **Master’s in Computer Science**  
        **Uppsala University** – Graduated in 2022  
        Specialization: Artificial Intelligence, Image Analysis, Computer Vision

        **Bachelor's in Computer Science and Engineering**  
        **Meenakshi College of Engineering** – Graduated in 2018
        """)

    # Links Section
    st.markdown("""
        ## 🔗 Links

        Connect with me on these platforms:

        - [LinkedIn](https://www.linkedin.com/in/uvais-karni-531a06147/)
        - [GitHub](https://github.com/uvaiskarni)
        """)

    # Fun Fact Section
    st.markdown("""
        ## 💭 Fun Fact 💭

        When I'm not building the next cool AI project, you can find me tinkering with **gadgets** or learning about **quantum computing** (just for fun 😜).

        **Thanks for visiting my page!** Let's make some cool stuff together!
        """)

    # Footer/Contact Info Section (Optional)
    st.markdown("""
        ## 📞 Contact Information

        - **Email**: uvais_karni@outlook.com
        - **Phone**: +46764469651
    """)


if __name__ == "__main__":
    _deploy_pre_streamlit_config()
    main()
