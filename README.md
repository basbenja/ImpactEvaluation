# Impact Evaluation

Authors:
- Bas Peralta, Benjamín
- Giuliodori, David Augusto
- Domínguez, Martín Ariel
- Rodríguez, Alejandro

# Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing
    virtual environments and Python versions.

# Install dependencies
**Having installed `uv`, `cd` into the project directory and run:**
```bash
uv sync
```
This command will automatically create a virtual environment and install al the
dependencies declared in `pyproject.toml`.

# Modules
This repository consists of the following modules:
- `data_generation`: Responsible for generating synthetic data for impact evaluation.

## `data_generation`
The configuration for data generation is specified in the
`data_generation/config.py` file.

To generate synthetic data, run the `data_generation/main.ipynb` notebook.

# Método experimental
### Esquema general
Cada conector recibe el panel de datos crudo, y las listas de ids de
entrenamiento y prueba. El conector se encarga de transformar el panel en el
formato que espera el modelo específico, y devuelve los datos de entrenamiento y
prueba listos para ser usados.

### Complejidades de los datos
Ahora bien, en nuestros datos hay algunas "complejidades":
- Primero, no todos los individuos tienen la misma cantidad de períodos
observados.
- Segundo, hay diferentes cohortes. Los tratados son tratados de una cohorte,
los controles son controles de una determinada cohorte, y los NiNis son NiNis
de todas las cohortes.

### Estrategia general de entrenamiento y test
La idea es que la red aprenda quiénes fueron los tratados de cada cohorte (y
quiénes no) para después poder predecir a los que serían controles de cada una.
Entonces, para esto lo que vamos a hacer es:
- En el conjunto de entrenamiento, vamos a usar los tratados de todas las cohortes
y algunos NiNis. De acá, la red aprende quiénes son 1s (tratados) y quiénes son 0s (NiNis).
- En el conjunto de test, vamos a usar los controles de cada cohorte y el resto
de NiNis. Lo que vamos a testear entonces es si la red es capaz de predecir
quiénes son 1s (controles) y quiénes son 0s (NiNis).

#### Conjunto de entrenamiento
En un escenario real, tenemos toda la información de los tratados, es decir conocemos
su historia, y la cohorte a la que pertenecen. Entonces, usamos toda esta información
para ellos.

De los no tratados, no sabemos quiénes son realmente controles y quiénes son
NiNis. Lo que vamos a hacer para facilitar un poco el entrenamiento es que todos
los 0s que metemos en el de entrenamiento, realmente son NiNis. Es decir, no
vamos a meter controles en el conjunto de entrenamiento (notar que lo que pasaría
en la realidad es que podríamos llegar a meter individuos que sean controles como 0s).

Así, el conjunto de entrenamiento quedaría conformado de esta forma:
- **Todos los tratados**, usando la mayor cantidad de períodos pre-tratamiento
posibles de cada uno + la cohorte a la que pertencen como feature extra. La
etiqueta de cada uno de estos individuos es 1.
- **Un subconjunto de NiNis**, repitiendo a cada uno tantas veces como cohortes
haya, y usando en cada repetición la mayor cantidad de períodos pre-inicio de
cohorte posibles + la cohorte a la que se le asigna en esa repetición como
feature extra. La etiqueta de cada uno de estos individuos (de todas las
repeticiones) es 0. La idea de esto es comunicarle de alguna forma a la red, que
los que son 0s, son 0s para todas las cohortes.

#### Conjunto de test
- **Todos los controles**, usando la misma estrategia que para los NiNis en el
entrenamiento, pero ahora asignando 1 a la cohorte a la que realmente
pertenecen. La idea de repetir esto es que en la realidad, no sabemos quiénes
son controles, entonces lo que queremos es que la red aprenda a predecir quiénes
son controles para cada cohorte, y no sólo para una cohorte específica.
- **El resto de NiNis**, usando la misma estrategia que para los NiNis en el
entrenamiento.

### Métrica
Lo que vamos a querer es que la red identifique a los controles en su cohorte
particular. Por ahora, no nos vamos a fijar que ponga 0s en las cohortes en las
que no son controles, aunque esto también sería algo interesante de evaluar.
