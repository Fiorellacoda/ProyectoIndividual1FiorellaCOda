from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import uvicorn

# Diccionario para mapear los nombres de los meses en español a números de mes
meses = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 
    'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8, 
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

# Diccionario para convertir días de la semana en español a números
dias = {
    'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6
}

# Cargar el CSV 
archivo_csv = r'C:\Users\Lauta\Downloads\dataset_modificado78188.csv'
df = pd.read_csv(archivo_csv, parse_dates=['release_date'])
#crear la app 
app = FastAPI()

class MesRequest(BaseModel):
    mes: str

class DiaRequest(BaseModel):
    dia: str

class TituloRequest(BaseModel):
    titulo: str

class ActorRequest(BaseModel):
    actor: str

class DirectorRequest(BaseModel):
    director: str
# Defino la funcion de similitud al coseno asi en un futuro el codigo que llame al funciona,iemto de la API queda mas prolijo
def calcular_similitud(title, data):
    #creo la matriz que sera necesaria para el calculo
    vectorizer = TfidfVectorizer()
    tfidf_matrix_genres = vectorizer.fit_transform(data['genres'])
#creo una nueva matriz de una sola columna
    vote_average = data['vote_average'].values.reshape(-1, 1)
    #uno la matriz vote_average con tfidf_matrix_genres
    features = np.hstack([tfidf_matrix_genres.toarray(), vote_average])
#calculo la matriz de similitud utilizando cosine_similarity
    similarity_matrix = cosine_similarity(features)
#verifico si el titulo solicitado existe
    if title not in data['title'].values:
        raise ValueError("El título no se encuentra en los datos.")
#solicito el indice del titulo 
    product_index = data[data['title'] == title].index[0]
    #obtengo los resultados de las comparaciones 
    product_similarities = similarity_matrix[product_index]
    #devuelve los indices de los titulos mas similares 
    similar_indices = np.argsort(-product_similarities)[1:6]
   #Muestro los resultados pero con excepcion del primero y el septimo
    return data.loc[similar_indices, 'title'].tolist()

@app.get("/cantidad_filmaciones_mes")
#def cantidad_filmaciones_mes( Mes ):
def cantidad_filmaciones_mes(mes: str = Query(..., description="Ingrese un mes.")):
    #Convierto el valor de mes a minúsculas para facilitar la comparación.
    mes = mes.lower()
#aqui se valida si el mes escrito es un mes o no
    if mes not in meses:
        #si no es mes aparecera mes no valido 
        raise HTTPException(status_code=400, detail="Mes no válido. Por favor ingrese un mes en español.")
    mes_numero = meses[mes]
   #
    peliculas_mes = df[df['release_date'].dt.month == mes_numero]
    #se cuenta la cantidad de peliculas
    cantidad = len(peliculas_mes)
    #se arroja el resultado
    return {"mes": mes, "cantidad": cantidad}

@app.get("/cantidad_filmaciones_dia")
#def cantidad_filmaciones_dia( Dia ):
def cantidad_filmaciones_dia(dia: str = Query(..., description="Ingrese un día de la semana.")):
    dia_num = dias.get(dia.lower())
    if dia_num is not None:
        count = df[df['release_date'].dt.weekday == dia_num].shape[0]
        return {"dia": dia, "cantidad": count}
    raise HTTPException(status_code=400, detail="Día no válido")

@app.get("/score_titulo")
def score_titulo(titulo_de_la_filmación: str = Query(..., description="Ingrese el título de la filmación.")):
    # Busca el titulo 
    filmacion = df[df['title'].str.contains(titulo_de_la_filmación, case=False, na=False)]
    
    if not filmacion.empty:
        #se busca toda la informacion que necesita la funcion
        fila = filmacion.iloc[0]
        titulo = fila['title']
        anio_estreno = fila['release_year']
        score = fila['vote_average']
        #arroja el resultado
        return {
            "mensaje": f"La película '{titulo}' fue estrenada en el año {anio_estreno} con un score/popularidad de {score}."
        }
    #si no se encuentra la pelicula se dira pelicula no encontrada
    raise HTTPException(status_code=404, detail="Filmación no encontrada.")

# Para ejecutar la API, puedes usar:
# uvicorn nombre_del_archivo:app --reload


@app.get("/votos_titulo")
#def votos_titulo( titulo_de_la_filmación ):
def votos_titulo(titulo: str = Query(..., description="Ingrese el título de la película.")):
    film = df[df['title'].str.lower() == titulo.lower()]
    if not film.empty:
        votos = film.iloc[0]['vote_count']
        if votos >= 2000:
            titulo = film.iloc[0]['title']
            año = film.iloc[0]['release_date'].year
            promedio = film.iloc[0]['vote_average']
            return {"titulo": titulo, "año": año, "votos": votos, "promedio": promedio}
        raise HTTPException(status_code=400, detail="La película no cumple con la condición de tener al menos 2000 valoraciones")
    raise HTTPException(status_code=404, detail="Película no encontrada")

@app.get("/get_actor")
def get_actor(nombre_actor: str = Query(..., description="Ingrese el nombre del actor.")):
    # Busca las películas del actor solicitado
    peliculas_actor = df[df['cast'].str.contains(nombre_actor, case=False, na=False)]
    
    if not peliculas_actor.empty:
        # Calcula el número de películas, retorno total y promedio de retorno
        num_peliculas = peliculas_actor.shape[0]
        retorno_total = peliculas_actor['return'].sum()
        promedio_retorno = retorno_total / num_peliculas  # Promedio

        return {
            "nombre_actor": nombre_actor,
            "num_peliculas": num_peliculas,
            "retorno_total": retorno_total,
            "promedio_retorno": promedio_retorno
        }
    
    raise HTTPException(status_code=404, detail="Actor no encontrado.")

@app.get("/get_director")
def get_director(nombre_director: str = Query(..., description="Ingrese el nombre del director.")):
    # Busca las películas del director
    peliculas = df[df['crew'].str.contains(nombre_director, case=False, na=False)]
    
    if not peliculas.empty:
        # Calcula el éxito total del director
        exito_total = peliculas['return'].sum()
        
        # Se prepara la lista con todos los datos necesarios para la funcion 
        detalles_peliculas = []
        for index, fila in peliculas.iterrows():
            detalles_peliculas.append({
                "titulo": fila['title'],
                "fecha_lanzamiento": fila['release_date'],
                "retorno_individual": fila['return'],
                "costo": fila['budget'],
                "ganancia": fila['return'] - fila['budget']  # Ganancia = retorno - costo
            })
        #devuelve los resultados
        return {
            "nombre_director": nombre_director,
            "exito_total": exito_total,
            "peliculas": detalles_peliculas
        }
    
    raise HTTPException(status_code=404, detail="Director no encontrado.")


@app.get("/similitud")
def obtener_similitud(title: str = Query(..., description="Ingrese el título de la película.")):
    try:
        similares = calcular_similitud(title, df)
        return {"tittle": title, "similares": similares}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9300)

