import streamlit as st
from multiapp import MultiApp
from apps import proarroz, parroz#, model # import your app modules here
from PIL import Image

im = Image.open("./apps/rice.gif")
st.set_page_config(page_title='ML-DL-Arroz-App', layout="wide", page_icon=im)
apps = MultiApp()

st.markdown("""
    # Arroz App""")
with st.expander("Descipción"):
    st.markdown("""
    En esta App podrás predecir la producción de Arroz teniendo en cuenta variables, como clima, fertilización, tipo de Semilla
    y suelo, al saber cual es tu producción podrás predecir cual va hacer el precio de venta por Ton en Colombia. Selecciona
    Navegador para seleccionar si vas a Calcular la producción o predicir el precio. En cada App se explica como usarla.
    """)
#This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed by [Praneel Nihar](https://medium.com/@u.praneel.nihar). Also check out his [Medium article](https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4).

row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    # image = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/ya.jpeg')
    # st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](https://ia.smartecorganic.com.co/index.php/contact/)')
with row1_2:
    # st.write("""
    # # Producción de Arroz App
    # Esta App utiliza algoritmos de Machine Learning y Deep Learning para predecir la productividad de arroz!
    # """)
    with st.expander("Contact us 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name = st.text_input('Name')
            mail = st.text_input('Email')
            q = st.text_area("Query")

            submit_button = st.form_submit_button(label='Send')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com", 587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n' + name + '\n' + mail + '\n' + q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()
st.markdown('____________')

# Add all your application here
apps.add_app("Productividad", proarroz.app)
apps.add_app("Predicción", parroz.app)
#app.add_app("Model", model.app)
# The main app
apps.run()
