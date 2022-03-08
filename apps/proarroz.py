import math
import time
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#from sklearn.linear_model._base import _base
#from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from datetime import date
import streamlit as st
from pandas_datareader import data as pdr
from datetime import date
from PIL import Image
import base64
import pydeck as pdk

def app():

    #im = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/arroz.png')
    im2 = Image.open('ya.jpeg')
    #st.set_page_config(page_title='ML-DL-Arroz-App', layout="wide", page_icon=im)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    row1_1, row1_2 = st.columns((2, 3))

    with row1_1:
        image = Image.open('ya.jpeg')
        st.image(image, use_column_width=True)
        # st.markdown('Web App by [Manuel Castiblanco](https://github.com/mcastiblanco1251)')
    with row1_2:
        st.write("""
        # Producción de Arroz App
        Esta App utiliza algoritmos de Machine Learning y Deep Learning para predecir la productividad de arroz!
        """)
        # with st.expander("Contact us 👉"):
        #     with st.form(key='contact', clear_on_submit=True):
        #         name = st.text_input('Name')
        #         mail = st.text_input('Email')
        #         q = st.text_area("Query")
        #
        #         submit_button = st.form_submit_button(label='Send')
        #         if submit_button:
        #             subject = 'Consulta'
        #             to = 'macs1251@hotmail.com'
        #             sender = 'macs1251@hotmail.com'
        #             smtpserver = smtplib.SMTP("smtp-mail.outlook.com", 587)
        #             user = 'macs1251@hotmail.com'
        #             password = '1251macs'
        #             smtpserver.ehlo()
        #             smtpserver.starttls()
        #             smtpserver.ehlo()
        #             smtpserver.login(user, password)
        #             header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
        #             message = header + '\n' + name + '\n' + mail + '\n' + q
        #             smtpserver.sendmail(sender, to, message)
        #             smtpserver.close()

    st.header('Aplicación')
    st.markdown('____________________________________________________________________')
    app_des = st.expander('Descripción App')
    with app_des:
        st.write("""Esta aplicación se desarrollo en la región arrocera del Hulia, específicamente en Yaguará,
        para lo cual los datos son válidos para el área cercana a este punto. La modelación tiene objetivo predecir la
        productividad Ton por Ha, teniendo en cuenta las siguientes variables:""")
        st.write("""
        - **Clima**: El historial de temperatura, precipitación, radiación.
        - **Tipo de Suelo**: De los diferentes lotes de cultivo mediante el análisis de suelo se definieron los tipos de suelos
        - **Fertilizacion**: Los fertilizantes mayores para los diferentes lotes-tipo de suelo.
        - **Tipo de Semillas**: Las diferentes tipos de semilla utilizados.""")
        st.write("""
        El objetivo de la app es integrar las variables ambientales, tipos de suelos y semilla, en el impacto de la
        fertilización ya que es uno de los costos más apreciables en el cultivo y como afecta la productividad.
        Se utilizo algoritmos de Machine-Deep Learning para generar el modelo. Para regiones diferentes le recomendamos
        ponerse en contacto con [Manuel Castiblanco](https://ia.smartecorganic.com.co)
        Este modelo fue gracias a la contribución de la Ing. Carolina Castiblanco.
        """)
        st.image(im2)  # , use_column_width=True)

    #st.sidebar.subheader('Entradas de Usuario')
    uploaded_file = 0#st.sidebar.file_uploader("Upload file CSV ", type=["csv"])

    if uploaded_file !=0:
        stock = pd.read_csv(uploaded_file)
    else:
        def user_input_features():

            df = pd.read_csv('./apps/Consolidado_Arroz_FY.csv')
            ac = pd.get_dummies(df, columns=['Semilla Variedades', 'Suelo'])
            st.sidebar.write('Fertilización')
            n = st.sidebar.slider('Nitrogeno Kg/Ha', float(ac.N.min()), float(ac.N.max()), float(ac.N.mean()))
            p = st.sidebar.slider('Fósforo Kg/Ha', float(ac.P.min()), float(ac.P.max()), float(ac.P.mean()))
            k = st.sidebar.slider('Potasio Kg/Ha', float(ac.K.min()), float(ac.K.max()), float(ac.K.mean()))
            st.sidebar.write('Condiciones Climáticas')
            r = st.sidebar.slider('Radiación MJ/m²/d', float(ac['Radiación'].min()), float(ac['Radiación'].max()),
                                  float(ac['Radiación'].mean()))
            t = st.sidebar.slider('Temperatura °C', float(ac.Temperatura.min()), float(ac.Temperatura.max()), float(ac.Temperatura.mean()))
            pr = st.sidebar.slider('Precipitación-mm/año', float(ac['Precipitación'].min()),
                                   float(ac['Precipitación'].max()), float(ac['Precipitación'].mean()))
            st.sidebar.write('Tipo Semilla')
            se = sorted(list(ac.columns[8:16]))
            se1 = st.sidebar.multiselect('Semilla Variedad', sorted(list(ac.columns[8:16])), sorted(list(ac.columns[8:9])))
            st.sidebar.write('Tipo Suelo')
            su = sorted(list(ac.columns[16:42]))
            su1 = st.sidebar.multiselect('Suelo Variedad', sorted(list(ac.columns[16:42])), sorted(list(ac.columns[16:17])))
            data = {
                'Nitrogeno Kg/Ha': n,
                'Fósforo Kg/Ha': p,
                'Potasio Kg/Ha': k,
                'Radiación MJ/m²/d': r,
                'Temperatura°C': t,
                'Precipitación-mm/año': pr}

            # 'Semilla Variedad': se1,
            # 'Suelo Variedad': su}
            def valorp(a):
                b = {}
                for col in a:
                    b[col] = 1
                    dic = b
                z = list(dic.items())
                df = pd.DataFrame(z).T
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
                return df

            def valorn(a):
                b = {}
                for col in a:
                    b[col] = 0
                    dic = b
                z = list(dic.items())
                df = pd.DataFrame(z).T
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
                return df

            def cri(a, b):
                ss = set(a)
                fs = set(b)
                inter = list(ss.intersection(fs))
                un = list(ss.union(fs))
                rest = list(ss.union(fs) - ss.intersection(fs))
                if not rest == []:
                    z = valorp(inter)
                    w = valorn(rest)
                    df = pd.concat([z, w], axis=1)

                else:
                    z = valorp(inter)
                    df = pd.DataFrame(z)

                return df

            df1 = pd.DataFrame(data, index=[0])
            df2 = cri(se, se1)
            df2.sort_index(axis=1, inplace=True)
            df3 = cri(su, su1)
            df3.sort_index(axis=1, inplace=True)
            df_1 = pd.concat([df1, df2], axis=1)
            df = pd.concat([df_1, df3], axis=1)
            return df


        input_df = user_input_features()
    st.subheader('Configuración de Datos de Entrada')
    input_df

    #load_model = pickle.load(open('./apps/linear_model.sav', 'rb'))
    df = pd.read_csv("./apps/Consolidado_Arroz_FY.csv")
    df1=pd.get_dummies(df,columns=['Semilla Variedades','Suelo'])
    X=df1.drop('Ren (Tn/Ha)',axis=1)
    X=X.drop('Ciclo',axis=1)
    y = df1[['Ren (Tn/Ha)']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    regression_model = LinearRegression()
    load_model=regression_model.fit(X_train, y_train)
    prediction = load_model.predict(input_df)

    np.random.seed(42)
    map_data = pd.DataFrame(
        np.random.randn(24, 2) / [30, 30] + [2.667, -75.517],
        columns=['lat', 'lon'])

    st.subheader('Ubicación del Modelo')
    loc = pd.DataFrame(list(input_df.columns[16:42]), columns=['loc'])
    dfg = loc.join(map_data)
    st.write('Localizaión')
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state={"latitude": map_data.lat.mean(),
                            "longitude": map_data.lon.mean(), "zoom": 11, "pitch": 10},
        layers=[
            pdk.Layer(
                "TextLayer",
                data=dfg,
                get_position=["lon", "lat"],
                get_text="loc",
                get_color=[0, 0, 0, 200],
                get_size=16,
                get_alignment_baseline="'bottom'",
            ),

            pdk.Layer(
                "HexagonLayer",
                data=dfg,
                get_position=["lon", "lat"],
                radius=500,
                elevation_scale=5,
                elevation_range=[0, 1000],
                extruded=True,
            ),
        ],
    ))
    row2_1, row2_2 = st.columns((1, 2))
    with row2_1:
        file_ = open('./apps/Ha.gif', "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="rice" width="150" height="100"/>',
            unsafe_allow_html=True
        )
    with row2_2:
        st.subheader(f'Productividad esperada Ton/Ha:{prediction}')
