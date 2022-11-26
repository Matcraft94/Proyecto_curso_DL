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

def get_seqs(data: np.ndarray, seq_len: int) -> np.ndarray:
    """Prepara la cadena de datos con su pareja de validacion

    Args:
        data (np.ndarray): Simulaciones de las EDOs
        seq_len (int): tamano de la cadena de 

    Returns:
        np.ndarray: _description_
    """
    X_train = []
    Y_train = []
    count = 0
    for i in range(seq_len, len(data)):
        try:
            tmp = data[i]
            # Extrameos media y std
            tmp_mean = np.mean(tmp)
            tmp_std = np.std(tmp)

            # extraemos y estandarizamos datos de entrenamiento
            tmp = data[i][:seq_len]
            tmp = (tmp - tmp_mean) / tmp_std
            X_train.append(tmp)
            
            # extraemos y estandarizamos datos de validacion
            tmp = data[i][seq_len:]
            tmp = (tmp - tmp_mean) / tmp_std
            Y_train.append(tmp)
        except:
            count += 1

    
    # transofrmamos a ndarray
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train