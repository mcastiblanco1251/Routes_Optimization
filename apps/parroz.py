import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import streamlit as st
from PIL import Image
import datetime
from selenium import webdriver
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

def app():
    #im = Image.open("C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/rice.gif")
    #st.set_page_config(page_title='Precio-Predict-LSTM', layout="wide", page_icon=im)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Actualización precio
    def fecha(today):
        if today[5:7]=='01':
            f=today[0:4]+'-12'+'-01'
        elif today[5:7]=='02':
            f=today[0:4]+'-01'+'-01'
        elif today[5:7]=='03':
            f=today[0:4]+'-02'+'-01'
        elif today[5:7]=='04':
            f=today[0:4]+'-03'+'-01'
        elif today[5:7]=='04':
            f=today[0:4]+'-03'+'-01'
        elif today[5:7]=='05':
            f=today[0:4]+'-04'+'-01'
        elif today[5:7]=='06':
            f=today[0:4]+'-05'+'-01'
        elif today[5:7]=='07':
            f=today[0:4]+'-06'+'-01'
        elif today[5:7]=='08':
            f=today[0:4]+'-07'+'-01'
        elif today[5:7]=='09':
            f=today[0:4]+'-08'+'-01'
        elif today[5:7]=='10':
            f=today[0:4]+'-09'+'-01'
        elif today[5:7]=='11':
            f=today[0:4]+'-10'+'-01'
        elif today[5:7]=='12':
            f=today[0:4]+'-11'+'-01'
        return f

    def actual_p():
    # Extrae precio
        browser=webdriver.Chrome('./apps/Chromedriver/chromedriver')
        url='https://fedearroz.com.co/es/fondo-nacional-del-arroz/investigaciones-economicas/estadisticas-arroceras/precios-del-sector-arrocero/'
        browser.get(url)
        time.sleep(2)
        html_code=browser.page_source
        soup=BeautifulSoup(html_code, 'lxml')
        browser.quit()
        table=soup.find('table')
        mes= table.find_all('tr',class_=['odd','even'])
        a=[]
        for i in range (12):
            p=mes[int(i)].find_all('td')
            for j in range (8):
                pr=p[j].text.replace('.','')
                a.append(pr)
            a
        b=[]
        for i in range(8):
            for j in range(12):
                c=i+j*8
                p=a[c]
                b.append(p)
            b
        b1=' '.join(b).split()
        p=b1[len(b1)-1]

        #Extrae fecha
        df = pd.read_csv("./apps/p_a_paddy.csv")
        try:
            df["Fecha"]= pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
            df["Fecha"]= pd.to_datetime(df["Fecha"], format="%Y-%m-%d")
        except:
            pass
        today = time.strftime("%Y-%m-%d")
        x=fecha(today)
        ad=[[x,p]]
        z=str(df.Fecha[len(df)-1])[0:10]
        if x==z:
            pass
        else:
            df2 = pd.DataFrame(ad)
            df2.to_csv("./apps/p_a_paddy.csv", index=False, mode='a', header=False)
        return


    row1_1, row1_2 = st.columns((2,3))
    with row1_1:
        image = Image.open('./apps/price.jpg')
        st.image(image, use_column_width=True)
        #st.markdown('Web App by [Manuel Castiblanco](https://github.com/mcastiblanco1251)')
    with row1_2:
        st.write("""
        # Predicción Precio Arroz App
        Esta applicaicón usa algoritmos de  Deep Learning LSTM para predecir!
        """)
        # with st.expander("Contact us 👉"):
        #     with st.form(key='contacts', clear_on_submit=True):
        #         name=st.text_input('Name')
        #         mail = st.text_input('Email')
        #         q=st.text_area("Query")
        #
        #         submit_button = st.form_submit_button(label='Send')
        #         if submit_button:
        #             subject = 'Consulta'
        #             to = 'macs1251@hotmail.com'
        #             sender = 'macs1251@hotmail.com'
        #             smtpserver = smtplib.SMTP("smtp-mail.outlook.com",587)
        #             user = 'macs1251@hotmail.com'
        #             password = '1251macs'
        #             smtpserver.ehlo()
        #             smtpserver.starttls()
        #             smtpserver.ehlo()
        #             smtpserver.login(user, password)
        #             header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
        #             message = header + '\n'+name + '\n'+mail+'\n'+ q
        #             smtpserver.sendmail(sender, to, message)
        #             smtpserver.close()

    st.header('Aplicación')
    st.write('_______________________________________________________________________________________________________')
    app_des=st.expander('Descripción App')
    with app_des:
        st.markdown("""
        Esta aplicación se basa en algoritmos de modelos LSTM de aprendizaje profundo para predecir acciones. Las redes de memoria a
        largo y corto plazo, generalmente llamadas simplemente "LSTM", son un tipo especial de RNN, capaz de aprender las dependencias
        a largo plazo. Fueron introducidos por Hochreiter y Schmidhuber (1997), y muchas personas los perfeccionaron y popularizaron en
        sus trabajos posteriores. Funcionan tremendamente bien en una gran variedad de problemas y ahora se utilizan ampliamente.

        Los LSTM están diseñados explícitamente para evitar el problema de la dependencia a largo plazo. Recordar información durante
        largos períodos de tiempo es prácticamente su comportamiento predeterminado, ¡no es algo que les cueste aprender!

        Todas las redes neuronales recurrentes tienen la forma de una cadena de módulos repetidos de red neuronal.
            """)


    #Actualiza los precios
    #actual_p()

    uploaded_file =0 #st.sidebar.file_uploader("Upload files CSV ", type=["csv"])

    if uploaded_file !=0:
        df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            type=st.sidebar.selectbox('Clase de Arroz', ('Paddy Verde', 'Arroz Blanco', 'Paddy Seco USA' ))
            df = pd.read_csv("./apps/p_a_paddy.csv")
            #try:
            #df["Fecha"]= pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
            #df["Fecha"]= pd.to_datetime(df["Fecha"], format="%Y-%m-%d")
            #except:
            #    pass
            return df

        df=user_input_features()
        df['$/Tonelada']=df["$/Tonelada"].astype(float)
        today= time.strftime("%Y-%m-%d")
        date_day = pd.date_range(start='1996-01-01', end=today, freq='M')
        df=df.set_index(date_day)
        def actual_pc(df):
            today= time.strftime("%Y-%m-%d")
            x=fecha(today)
            n=df['Fecha'][len(df)-1]
            if x==n:
                pass
            else:
                actual_p()
        actual_pc(df)
        #df=df.set_index('Fecha')
        #year=df['Fecha'][len(df)-1][5:9]
        #st.subheader(f'Mes a Pronosticar del')# {year}')


    row2_1, row2_2, = st.columns((2,2))

    with row2_1:
        st.subheader('Precio arroz 1996-Actual')
        plt.figure(figsize=(16,15))
        plt.title('Precio Arroz Paddy Mensual')
        plt.plot(df['$/Tonelada'])
        plt.xlabel('Mes',fontsize=18)
        plt.ylabel('Precio $COP/Ton',fontsize=18)
        plt.show()
        st.pyplot(plt.show())

    with row2_2:
        st.subheader('Tabla de Precios')
        st.write(df['$/Tonelada'].tail(10))

    data = df.filter(["$/Tonelada"])
    #data

    #Converting the dataframe to a numpy array
    dataset = data.values
    #dataset

    #Get /Compute the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) *.8)
    #training_data_len

    #Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler

    scaled_data = scaler.fit_transform(dataset)
    #scaled_data

    #Create the scaled training data set
    train_data = scaled_data[0:training_data_len, : ]
    #train_data.shape

    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])

    #Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler

    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #x_train.shape, y_train.shape

    #Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    #st.write(x_train)

    #Build the LSTM network model
    model = Sequential()
    #
    model.add(LSTM(units=150, return_sequences=True,input_shape=(x_train.shape[1],1)))
    #model.add(LSTM(units=25, return_sequences=True))
    #model.add(LSTM(units=25, return_sequences=True))
    model.add(LSTM(units=150, return_sequences=False))
    model.add(Dense(units=150))
    model.add(Dense(units=1))


    #Compile the model

    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    #st.subheader('Model')
    #st.write(str(model.summary()))

    #Train the model
    model.fit(x_train, y_train, batch_size=8, epochs=50)

    #Test data set
    test_data = scaled_data[training_data_len - 60: , : ]
    #test_data

    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    #Convert x_test to a numpy array
    x_test = np.array(x_test)
    #x_test

    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


    #Getting the models predicted price values
    predictions = model.predict(x_test)
    scaler=scaler.fit(dataset)
    predictions = scaler.inverse_transform(predictions)#Undo scaling
    #predictions

    #Calculate/Get the value of RMSE
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    #rmse

    import sklearn
    from sklearn.metrics import r2_score
    r2=sklearn.metrics.r2_score(predictions, y_test)

    from sklearn.metrics import mean_absolute_error
    mae=mean_absolute_error(predictions, y_test)

    from sklearn.metrics import mean_squared_error
    mse=mean_squared_error(predictions, y_test)


    # latest_iteration = st.empty()
    # bar = st.progress(0)
    # iter=50
    # for i in range(iter):
    #     latest_iteration.text(f'Progress {i*(150//iter)}%')
    #     bar.progress(i *(150//iter))
    #     time.sleep(0.1)


    st.subheader('Evaluación de Exactitud del Modelo')
    #acc={'RMSE': rmse, 'R2':r2, 'MSE':mse, 'MAE':mae}
    st.write(pd.DataFrame({'RMSE': rmse, 'R2':r2, 'MSE':mse,'MAE':mae}, columns=['RMSE', 'R2', 'MSE', 'MAE'], index=['Acurracy']))


    if r2>=0.90:
        st.write('**Gran Desempeño!!!**')
    elif r2>=0.8 and r2<0.9:
        st.write('**Aceptable Desempeño**')
    elif r2>0.6 and r2<0.8:
        st.write('**Regular Desempeño**')
    elif r2<0.6:
        st.write('**Bajo Desempeño- Rechazar Modelo**')

    # training metrics
    #scores = model.evaluate(x_train, y_train, verbose=1, batch_size=200)
    #print('Accurracy: {}'.format(scores[1]))

    #Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #valid

    #st.subheader('Company name: ' + name)
    st.subheader('Predicción')
    row3_1, row3_2, = st.columns((2,2))

    with row3_1:
    #Visualize the data
        st.subheader('Gráfico Precio Real y Predicción')
        plt.figure(figsize=(16,15))
        plt.title('Modelo')
        plt.xlabel('Fecha', fontsize=18)
        plt.ylabel('Precio Arroz Paddy $COP/Ton', fontsize=18)
        plt.plot(train['$/Tonelada'])
        plt.plot(valid[['$/Tonelada', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()
        #plt.show()
        st.pyplot(plt.show())
    #Show the valid and predicted prices

    with row3_2:
        st.subheader('Precios Reales y Predicción')
        st.write(valid.tail(10))

    #Get the quote
    stock_quote = df#pdr.get_data_yahoo(stock, start='2012-01-01', end=end)
    #Create a new dataframe
    new_df = stock_quote.filter(['$/Tonelada'])
    #Get teh last 60 day closing price
    last_60_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_60_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    string = ' '.join(str(x) for x in pred_price)

    today = time.strftime("%Y-%m-%d")
    st.subheader(f'Precio Futuro para {today[0:7]} ${string}COP') #{df['Fecha'][len(df)-1]}
    #st.sidebar.text_input('Future Price:', str(pred_price))

    #Contact Form

    with st.expander('Ayuda? 👉'):
        st.markdown(
                " Necesitas Ayuda? contacte a [Manuel Castiblanco](https://ia.smartecorganic.com.co/index.php/contact/)")
