import pandas as pd
import numpy as np

def get_ode_data(file_name: str) -> pd.DataFrame:
    """Funcion que obtiene los datos de las edos simuladas

    Args:
        file_name (str): noimbre del archivo .csv

    Returns:
        pd.DataFrame: DataFrame de las ODEs simuladas
    """
    data = pd.read_csv(file_name, sep=';')
    data['sol.t'] = data['sol.t'].apply(lambda x: np.float_(x.replace('[','').replace(']','').split(',')))
    data['sol.y'] = data['sol.y'].apply(lambda x: np.float_(x.replace('[','').replace(']','').split(',')))
    
    # Imprimimos los datos que encontro
    print('El tamano de los datos de polinomio en S y t es: {}'.format(data[data['funcion'] == 'polinom_st'].shape[0]))
    print('El tamano de los datos de polinomio en S es    : {}'.format(data[data['funcion'] == 'polinom_s'].shape[0]))
    print('El tamano de los datos de polinomio en t es    : {}'.format(data[data['funcion'] == 'polinom_t'].shape[0]))
    print('El numero total de los datos es                : {}'.format(data.shape[0]))

    return data

