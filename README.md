# ScikitLearn + FastAPI + Docker
Deployment of ML models using Python's Scikit-Learn + FastAPI + Docker

# Dataset
## Breast Cancer Wisconsin (Diagnostic) Data Set

Para esta clase usaremos este dataset

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)


### Información  de atributos:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)
s:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)


# Virtual Environment
`$ pip install virtualenv`

Crear un nuevo ambiente virtual
`$ virtualenv venv`

Activar el ambiente virtual

`$ source venv/bin/activate`

# Paquetes de Python


`$ pip install -r requirements.txt`

# Entreno

Correr este ambiente para entrenar el modelo

`$ python code/train.py`

# Applicacion web

Probar la aplicación de fastapi con:

`$ uvicorn main:app`

# Docker

Creando la imagen Docke

`$ docker build . -t sklearn_fastapi_docker`

Test la aplicación usando Docker

`$ docker run -p 8000:8000 sklearn_fastapi_docker`

# Test!
Test con los archivos  enn tests/example_calls.txt desde la terminal
