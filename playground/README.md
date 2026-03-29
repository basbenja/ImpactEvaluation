Mi estrategia va a ser esta:

1. Generar el conjunto de datos (acá van a estar disciminados quiénes son
   tratados en cada cohorte, quiénes son controles de cada cohorte, y quiénes no
   son ni tratados ni controles en ninguna cohorte. A estos últimos los llamo
   NiNi).
2. Agarro el conjunto de datos crudo y genero train y test. Acá simplemente
   obtengo los IDs de los que van a estar en train y de los que van a estar en
   test. El train va a estar formado por tratados + algunos NiNi, y el test por
   controles + el resto de NiNi. Las features van a ser los pasos de tiempo de
   las diferentes caracteristicas + la cohorte a la que pertence el tratado /
   control.
3. Agarro el train y test y los estructuro para la red neuronal o modelo que
   quiera probar.

Por ahora, quiero hacer una conexión de todo y después voy a pasar a usar mi
generador de datos real.

Para empezar, necesito que me des un csv en formato de datos panel que tenga las
siguientes columnas:
1. ID (identificador de la empresa)
2. t (hace referencia al período de tiempo)
3. y_1 (variable observable de cada empresa)
4. y_2 (variable observable de cada empresa)
5. tratado
6. control
7. cohorte en la que es tratado o control

Además, agrega una complejidad extra. Hace que la cantidad de periodos de las
variables observables varie de empresa a empresa. Por ejemplo, si modelas 20
periodos (con inicios de cohorte en 8, 9 y 10), hacé que por ejemplo de una
empresa pueda ver y_1  e y_2 a partir del periodo 5, de otra a partir del
periodo 3, y asi. Esto seria un proxy de la antiguedad de la empresa