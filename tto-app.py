import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from copy import copy
import datetime
import pickle
from geopy.geocoders import  Photon
from datetime import date
import streamlit as st
from PIL import Image
import pydeck as pdk
import altair as alt
import time
import email
import smtplib
import joblib

im = Image.open("rutas.jpg")

st.set_page_config(page_title='Optimizacion', layout="wide", page_icon=im)
#st.set_option('deprecation.showPyplotGlobalUse', False)

# LAYING OUT THE TOP SECTION OF THE APP

row1_1, row1_2 = st.columns((2,3))

with row1_1:
    image = Image.open('opt.jpg')
    st.image(image, use_container_width=True)
    st.markdown('Web App by [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)')
with row1_2:
    st.write("""
    # Ruta Optima App
    Esta app usa machine learning y alrgoritmo genético para encontrar la ruta óptima!
    """)
    with st.expander("Contact us 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name=st.text_input('Nombre')
            mail = st.text_input('Email')
            q=st.text_area("Consulta")

            submit_button = st.form_submit_button(label='Enviar')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com",587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n'+name + '\n'+mail+'\n'+ q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()


st.header('Aplicación')
st.write('_______________________________________________________________________________________________________')


#loaded_model = pickle.load(open('xgb_model.sav', 'rb'))


st.sidebar.subheader('Archivo con los sitios a optimizar')


loaded_model=joblib.load("xgb_model.json")

uploaded_file = st.sidebar.file_uploader("Cargue archivo CSV si tienes los puntos a Optimizar p.e.: {L1: Latitud:0.0, Longitud:0.0}", type=["csv"])

st.sidebar.subheader('Características de Entrada')
if uploaded_file is not None:
    map_data = pd.read_csv(uploaded_file)
else:
    def user_input_features():

        c=pd.read_csv('worldcities.csv')
        city=st.sidebar.selectbox('Seleccione la ciudad a Optimizar',(c.city_ascii))
        n=c[c['city_ascii']==city].index.item()
        lat=c['lat'][n]
        long=c['lng'][n]
        p = st.sidebar.number_input('Número de Puntos en la Ruta', value=10, min_value=0)#st.sidebar.slider('No de Puntos en la ruta', 5,20,10)
        np.random.seed(42)
        map_data = pd.DataFrame(
        np.random.randn(p, 2) / [20,20] + [lat, long],
        columns=['lat', 'lon'])
        return map_data

    def iter():
        iter = st.sidebar.number_input('Número de Iteraciones Optimización', value=1, min_value=0)#st.sidebar.slider('No de Iteraciones Optimización', 10,100,10)
        return iter

    def date():
        my_date = st.sidebar.date_input("Fecha")
        return my_date

map_data=user_input_features()
iter=iter()
my_date=date()

#DataFrame general inicial



geolocator =  Photon(user_agent="tto", timeout=None)
test_locations={}
loc_a=[]
for i in range(len(map_data)):
    b=map_data.lat[i],map_data.lon[i]
    location = geolocator.reverse(b)
#    addresses.append(location.address)
    #coord[i]={'L'+i:b}
    test_locations[f'Punto L{i}']=(map_data.lat[i],map_data.lon[i])#, 'Coord':b}
    loc_a.append(location.address)

#st.write(list(addresses.keys()), addresses)
loc=pd.DataFrame(list(test_locations.keys()), columns=['Loc'])
add=pd.DataFrame(loc_a, columns=['Dirección'])
df1=loc.join(add)
dfg=df1.join(map_data)


app_des=st.expander('Descripción de la App')
with app_des:
    st.markdown("""
    ¿Cuál es la relación entre el aprendizaje automático y la optimización? - Por un lado, la optimización matemática se utiliza en el aprendizaje automático durante el entrenamiento del modelo, cuando intentamos minimizar el costo de los errores entre nuestro modelo y nuestros puntos de datos. Por otro lado, ¿qué sucede cuando se usa el aprendizaje automático para resolver problemas de optimización?

    Considere esto: un conductor de UPS con 25 paquetes tiene 15 billones de rutas   posibles para elegir. Y si cada conductor recorre solo una milla más de lo necesario cada día, la compañía estaría perdiendo $ 30 millones al año. Si bien UPS tendría todos los datos de sus camiones y rutas, no hay forma de que puedan ejecutar 15 billones de cálculos por cada conductor con 25 paquetes. Sin embargo, este problema  se puede abordar con algo llamado "algoritmo genético".

    El problema aquí es que este algoritmo requiere tener algunas entradas, digamos el tiempo de viaje entre cada par de ubicaciones. Pero, ¿qué pasa si usamos el poder predictivo del aprendizaje automático para potenciar el algoritmo genético? En términos simples, podemos usar el poder del aprendizaje automático para pronosticar los tiempos de viaje entre cada dos ubicaciones y usar el algoritmo genético para encontrar la mejor ruta de viaje para nuestro camión de reparto.

    Esta App es basada en trajados hechos por Zach Miller (evolutionary_algorithm_traveling_salesman) y Vladmir Vlazovskiy (route-optimizer-machine-learning)

    *Puntos a tener en cuenta*
    - Puede cargar cargar archivos *.csv donde esten los puntos con Latitud y Longitud que desee optimizar.
    - A mayor numero de puntos de rutas el tiempo de optimización es mayor.
    - A mayor numero de iteraciones el tiempo de optimizción es mayor.
    - La fecha tiene en cuenta en el modelo el cálculo de los tiempos ya que es diferente para los diferentes meses del año.
        """)



# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
st.subheader('Representación Inicial')
row2_1, row2_2, = st.columns((2,2))

with row2_1:
    st.write('Localización Incial')
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state={"latitude": dfg.lat.mean(),
                            "longitude": dfg.lon.mean(), "zoom": 11, "pitch": 50},
        layers=[
            pdk.Layer(
            "TextLayer",
            data=dfg,
            get_position=["lon", "lat"],
            get_text="Loc",
            get_color=[0, 0, 0, 200],
            get_size=16,
            get_alignment_baseline="'bottom'",
            ),

            pdk.Layer(
            "HexagonLayer",
            data=dfg,
            get_position=["lon", "lat"],
            radius=500,
            elevation_scale=50,
            #colorRage(0, 0, 0, 200),
            elevation_range=[0, 3000],
            extruded=True,

            ),
        ],
    ))


with row2_2:
    st.write('Tabla Resumen',dfg)

def create_guess(points):
    """
    Creates a possible path between all points, returning to the original.
    Input: List of point IDs
    """
    guess = copy(points)
    np.random.shuffle(guess)
    guess.append(guess[0])
    return list(guess)
#
#st.write(create_guess(list(test_locations.keys())))

def plot_cities(city_coordinates, annotate=True):
    """
    Makes a plot of all cities.
    Input: city_coordinates; dictionary of all cities and their coordinates in (x,y) format
    """
    names = []
    x = []
    y = []
    fig=plt.figure(dpi=250)
    for ix, coord in city_coordinates.items():
        names.append(ix)
        x.append(coord[0])
        y.append(coord[1])
        if annotate:
            plt.annotate(ix, xy=(coord[0], coord[1]), xytext=(20, -20),
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='w', alpha=0.5),
                        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.scatter(x,y,c='r',marker='o')
    return(fig)

#plot_cities(test_locations)


def plot_guess(city_coordinates, guess, guess_in_title=False):
    """
    Takes the coordinates of the cities and the guessed path and
    makes a plot connecting the cities in the guessed order
    Input:
    city_coordinate: dictionary of city id, (x,y)
    guess: list of ids in order
    """
    #plot_cities(city_coordinates)
    fig, ax=plt.figure()#dpi=250)
    for ix, current_city in enumerate(guess[:-1]):
        x = [city_coordinates[guess[ix]][0],city_coordinates[guess[ix+1]][0]]
        y = [city_coordinates[guess[ix]][1],city_coordinates[guess[ix+1]][1]]
        ax=plt.plot(x,y,'c--',lw=1)
    plt.scatter(city_coordinates[guess[0]][0],city_coordinates[guess[0]][1], marker='x', c='b')
    if guess_in_title:
        plt.title("Current Guess: [%s]"%(','.join([str(x) for x in guess])))
    else:
        print("Current Guess: [%s]"%(','.join([str(x) for x in guess])))
    return(fig)

path = create_guess(list(test_locations.keys()))
#st.write(path)#print(path)
# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
st.subheader('Representación esquematica Inicial')

row3_1, row3_2, = st.columns((2,2))
with row3_1:
    #fig=plot_cities(test_locations)
    st.pyplot(plot_cities(test_locations))
    st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 10px;">Esquema X-Y de los puntos de Ruta a Optimizar</p>', unsafe_allow_html=True)
with row3_2:
    #fig=
    st.pyplot(plot_guess(test_locations, path))
    st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 10px;">Esquema X-Y ruta inicial: {}</p>'.format(path), unsafe_allow_html=True)

st.subheader('Machine Learning & Optimización')

st.write('**Generación de Rutas y Cálculo de tiempo por ruta**')

def create_generation(points, population=100):
    """
    Makes a list of guessed point orders given a list of point IDs.
    Input:
    points: list of point ids
    population: how many guesses to make
    """
    generation = [create_guess(points) for _ in range(population)]
    return generation

test_generation = create_generation(list(test_locations.keys()), population=10)

def travel_time_between_points(point1_id, point2_id, hour, date, passenger_count = 1,
                               store_and_fwd_flag = 0, pickup_minute = 0):
    """
    Given two points, this calculates travel between them based on a XGBoost predictive model
    """

    model_data = {'passenger_count': passenger_count,
                  'pickup_longitude' : point1_id[1],
                  'pickup_latitude' : point1_id[0],
                  'dropoff_longitude' : point2_id[1],
                  'dropoff_latitude' : point2_id[0],
                  'store_and_fwd_flag' : store_and_fwd_flag,
                  'pickup_month' : my_date.month,
                  'pickup_day' : my_date.day,
                  'pickup_weekday' : my_date.weekday(),
                  'pickup_hour': hour,
                  'pickup_minute' : pickup_minute,
                  'latitude_difference' : point2_id[0] - point1_id[0],
                  'longitude_difference' : point2_id[1] - point1_id[1],
                  'trip_distance' : 0.621371 * 6371 * (abs(2 * np.arctan2(np.sqrt(np.square(np.sin((abs(point2_id[0] - point1_id[0]) * np.pi / 180) / 2))),
                                  np.sqrt(1-(np.square(np.sin((abs(point2_id[0] - point1_id[0]) * np.pi / 180) / 2)))))) + \
                                     abs(2 * np.arctan2(np.sqrt(np.square(np.sin((abs(point2_id[1] - point1_id[1]) * np.pi / 180) / 2))),
                                  np.sqrt(1-(np.square(np.sin((abs(point2_id[1] - point1_id[1]) * np.pi / 180) / 2)))))))
                 }

    df = pd.DataFrame([model_data], columns=model_data.keys())

    pred = np.exp(loaded_model.predict(xgb.DMatrix(df))) - 1

    return pred[0]

coordinates = test_locations

def fitness_score(guess):
    """
    Loops through the points in the guesses order and calculates
    how much distance the path would take to complete a loop.
    Lower is better.
    """
    score = 0
    for ix, point_id in enumerate(guess[:-1]):
        score += travel_time_between_points(coordinates[point_id], coordinates[guess[ix+1]], 11, my_date)
    return score

def check_fitness(guesses):
    """
    Goes through every guess and calculates the fitness score.
    Returns a list of tuples: (guess, fitness_score)
    """
    fitness_indicator = []
    for guess in guesses:
        fitness_indicator.append((guess, fitness_score(guess)))
    return fitness_indicator

with st.expander('Tabla Ruta y Tiempo de Trayecto'):
    r_t=check_fitness(test_generation)
    ml=pd.DataFrame(r_t, columns=['Ruta', 't Trayecto(min)'])
    ml

st.subheader('Optimizacion de Generación')

def get_breeders_from_generation(guesses, take_best_N=10, take_random_N=5, verbose=False, mutation_rate=0.1):
    """
    This sets up the breeding group for the next generation. You have
    to be very careful how many breeders you take, otherwise your
    population can explode. These two, plus the "number of children per couple"
    in the make_children function must be tuned to avoid exponential growth or decline!
    """
    # First, get the top guesses from last time
    fit_scores = check_fitness(guesses)
    sorted_guesses = sorted(fit_scores, key=lambda x: x[1]) # sorts so lowest is first, which we want
    new_generation = [x[0] for x in sorted_guesses[:take_best_N]]
    best_guess = new_generation[0]

    if verbose:
        # If we want to see what the best current guess is!
        print(best_guess)

    # Second, get some random ones for genetic diversity
    for _ in range(take_random_N):
        ix = np.random.randint(len(guesses))
        new_generation.append(guesses[ix])

    # No mutations here since the order really matters.
    # If we wanted to, we could add a "swapping" mutation,
    # but in practice it doesn't seem to be necessary

    np.random.shuffle(new_generation)
    return new_generation, best_guess

def make_child(parent1, parent2):
    """
    Take some values from parent 1 and hold them in place, then merge in values
    from parent2, filling in from left to right with cities that aren't already in
    the child.
    """
    list_of_ids_for_parent1 = list(np.random.choice(parent1, replace=False, size=len(parent1)//2))
    child = [-99 for _ in parent1]

    for ix in range(0, len(list_of_ids_for_parent1)):
        child[ix] = parent1[ix]
    for ix, gene in enumerate(child):
        if gene == -99:
            for gene2 in parent2:
                if gene2 not in child:
                    child[ix] = gene2
                    break
    child[-1] = child[0]
    return child

def make_children(old_generation, children_per_couple=1):
    """
    will be left out.
    Pairs parents together, and makes children for each pair.
    If there are an odd number of parent possibilities, one

    Pairing happens by pairing the first and last entries.
    Then the second and second from last, and so on.
    """
    mid_point = len(old_generation)//2
    next_generation = []

    for ix, parent in enumerate(old_generation[:mid_point]):
        for _ in range(children_per_couple):
            next_generation.append(make_child(parent, old_generation[-ix-1]))
    return next_generation

current_generation = create_generation(list(test_locations.keys()),population=500)
print_every_n_generations = 5

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(iter+1):
    latest_iteration.text(f'Progreso de Optimización {i*(100//iter)}%')
    bar.progress(i *(100//iter))
    time.sleep(0.1)
    if not i % print_every_n_generations:
        print("Generation %i: "%i, end='')
        print(len(current_generation))
        is_verbose = True
    else:
        is_verbose = False
    breeders, best_guess = get_breeders_from_generation(current_generation,
                                                        take_best_N=250, take_random_N=100,
                                                        verbose=is_verbose)
    current_generation = make_children(breeders, children_per_couple=3)

st.subheader('Encontrando la Ruta Optima')

def evolve_to_solve(current_generation, max_generations, take_best_N, take_random_N,
                    mutation_rate, children_per_couple, print_every_n_generations, verbose=False):
    """
    Takes in a generation of guesses then evolves them over time using our breeding rules.
    Continue this for "max_generations" times.
    Inputs:
    current_generation: The first generation of guesses
    max_generations: how many generations to complete
    take_best_N: how many of the top performers get selected to breed
    take_random_N: how many random guesses get brought in to keep genetic diversity
    mutation_rate: How often to mutate (currently unused)
    children_per_couple: how many children per breeding pair
    print_every_n_geneartions: how often to print in verbose mode
    verbose: Show printouts of progress
    Returns:
    fitness_tracking: a list of the fitness score at each generations
    best_guess: the best_guess at the end of evolution
    """
    fitness_tracking = []
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(max_generations+1):
        latest_iteration.text(f'Progreso de Optimización {i*(100//max_generations)}%')
        bar.progress(i *(100//max_generations))
        time.sleep(0.1)
        if verbose and not i % print_every_n_generations and i > 0:
            print("Generation %i: "%i, end='')
            print(len(current_generation))
            print("Current Best Score: ", fitness_tracking[-1])
            is_verbose = True
        else:
            is_verbose = False
        breeders, best_guess = get_breeders_from_generation(current_generation,
                                                            take_best_N=take_best_N, take_random_N=take_random_N,
                                                            verbose=is_verbose, mutation_rate=mutation_rate)
        fitness_tracking.append(fitness_score(best_guess))
        current_generation = make_children(breeders, children_per_couple=children_per_couple)

    return fitness_tracking, best_guess

current_generation = create_generation(list(test_locations.keys()),population=500)
fitness_tracking, best_guess = evolve_to_solve(current_generation, iter, 150, 70, 0.5, 3, 5, verbose=True)

st.subheader('La Ruta Optima Encontrada')


#Graficar la respuesta optima
a=[]
for i in range (len(best_guess)):
    for j in range(len(list(test_locations))):
        if best_guess[i]==list(test_locations.items())[j][0]:
            p=list(test_locations.items())[j][1]
            a.append(p)

df1=pd.DataFrame(a, columns=['lat', 'lon'])
#df1
df2=pd.DataFrame(best_guess, columns=['Loc'])
#df2.T

df3=df2.join(df1)
#df3

lon2=[]
lat2=[]
for i in range(len(df2)-1):
    lat=df3.lat[i+1]
    lon=df3.lon[i+1]
    lon2.append(lon)
    lat2.append(lat)
df4=df3[:-1]
df4['lon2']=lon2
df4['lat2']=lat2

#st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 20px;"> Ruta a Optima</p>', unsafe_allow_html=True)

st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 15px;">Ruta: {}</p>'.format(df2.Loc), unsafe_allow_html=True)

#df4

row4_1, row4_2, = st.columns((2,2))
with row4_1:

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state={"latitude": df3.lat.mean(),
                            "longitude": df3.lon.mean(), "zoom": 11, "pitch": 50},
        layers=[
            pdk.Layer(
            "TextLayer",
            data=df4,
            get_position=["lon", "lat"],
            get_text="Loc",
            get_color=[0, 0, 0, 200],
            get_size=16,
            get_alignment_baseline="'bottom'",
            ),

            pdk.Layer(
                "ArcLayer",
                data=df4,
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[255, 0, 0, 255],
                get_target_color=[0, 255, 0, 255],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                getHeight=1,
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        ],
    ))

with row4_2:
    st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 15px;"> Esquema X - Y Ruta a Optima</p>', unsafe_allow_html=True)
    st.pyplot(plot_guess(test_locations, best_guess))

# Contact Form

with st.expander('Necesita Ayuda? 👉'):
    st.markdown(
            "Tiene problemas en entender la App? contacte [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)")
