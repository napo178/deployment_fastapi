# deployment_fastapi

Deployment Machine Learning Models

Como correr el sistema


1. Clone the repository

git clone (url del repositorio)

2. Install poetry

pip install poetry

3. Crear ambiente de poetry

Poetry new proyect

4. Agregar librerias

Poetry add Fastapi sklearn


5. Correr docker 

docker build -t mlprod .



docker run -p 8000:80 mlprod
