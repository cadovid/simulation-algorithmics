import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

t = 1 # Numero de experimentos
n = 150 # Numero de muestras
ival_low = np.array([]) # Datos de intervalo de confianza en comparacion de sistemas
ival_high = np.array([]) # Datos de intervalo de confianza en comparacion de sistemas

for i in range(t):
    """ _________________________________________________________
        
        Muestras de variable Gaussiana (Normal) X 
        _________________________________________________________
    """
    
    # Generacion de muestras
    mu_X = 1
    std_X = 1
    _X = np.random.normal(mu_X, std_X, n)
    
    """ _________________________________________________________
        
        Muestras de variable Gaussiana (Normal) Y 
        _________________________________________________________
    """
    
    # Generacion de muestras
    mu_Y = 1
    std_Y = 1
    _Y = np.random.normal(mu_Y, std_Y, n)
    
    for i, val in enumerate(_Y):
        if val <= 0:
            while True:
                val = np.random.normal(mu_Y, std_Y)
                _Y[i] = val
                if not val <= 0:
                    break
    
    """______________________________________________________
        
        Resultados y analisis de salida
       ______________________________________________________
    """
    
    # Representacion de resultados obtenidos
    data = {'X_Variable': _X,
            'Y_Variable': _Y}
    df = pd.DataFrame(data)
    #print(df)
    
    # Calculo de la distribucion t-Student con un nivel de confianza dado
    def t(alpha, gl):
        return scipy.stats.t.ppf(1-(alpha/2), gl)
    
    # Calculo de intervalos de confianza
    mu_X = df['X_Variable'].mean()
    mu_Y = df['Y_Variable'].mean()
    #print('Media de X_Variable: ' + str(mu_X))
    #print('Media de Y_Variable: ' + str(mu_Y))
    std_X = df['X_Variable'].std()
    std_Y = df['Y_Variable'].std()
    #print('Intervalo de confianza de 95%, X_Variable: [' + str(mu_X - t(0.05, n-1) * (std_X/math.sqrt(n))) + ', ' + str(mu_X + t(0.05, n-1) * (std_X/math.sqrt(n))) + ']')
    #print('Intervalo de confianza de 95%, Y_Variable: [' + str(mu_Y - t(0.05, n-1) * (std_Y/math.sqrt(n))) + ', ' + str(mu_Y + t(0.05, n-1) * (std_Y/math.sqrt(n))) + ']')
    
    # # Representacion del histograma de las muestras X
    # fig, ax = plt.subplots(figsize=(6,4))
    # sns.distplot(_X, kde=False, label='X Value')
    # ax.set_xlabel("Value of the variable", fontsize=12)
    # ax.set_ylabel("Frequency", fontsize=12)
    # plt.axvline(mu_X, color='red')
    # plt.legend()
    # plt.tight_layout()
    
    # # Representacion del histograma de las muestras Y
    # fig, ax = plt.subplots(figsize=(6,4))
    # sns.distplot(_Y, kde=False, label='Y Value')
    # ax.set_xlabel("Value of the variable", fontsize=12)
    # ax.set_ylabel("Frequency", fontsize=12)
    # plt.axvline(mu_Y, color='red')
    # plt.legend()
    # plt.tight_layout()
    
    # Comparacion de los sistemas por medio de test t-Student (Test no pareado)
    mu = mu_X - mu_Y # Diferencia de medias
    std = (std_X/math.sqrt(n)) + (std_Y/math.sqrt(n)) # Varianza entre las medias
    #gl = n # Estimacion 1 de grados de libertad: Distinta varianza
    #gl = n+n-2 # Estimacion 2 de grados de libertad: Misma varianza
    gl = ((std_X**2/n)+(std_Y**2/n))**2/(((std_X**2/n)**2/(n-1))+((std_Y**2/n)**2/(n-1))) # Estimacion 3 de grados de libertad: Aproximacion de Welch
    #print('Intervalo de confianza de 95%, Comparacion de sistemas: [' + str(mu - t(0.05, gl) * std) + ', ' + str(mu + t(0.05, gl) * std) + ']')

    # # Representacion de valores de los experimentos realizados
    # ival_low = np.append(ival_low, mu - t(0.05, gl) * std)
    # ival_high = np.append(ival_high, mu + t(0.05, gl) * std)
    
    # Test simetrico con nivel de confianza 95% y alpha 0.05
    print('La media aceptable resultado del test simetrico para X_Variable se contiene en el intervalo: [' + str(1 - 1.96 * (1/math.sqrt(n))) + ', '+ str(1 + 1.96 * (1/math.sqrt(n))) + ']')
    z_X = math.sqrt(n)*(abs(mu_X-1)/1) # Desviacion de la muestra, para aceptacion de hipotesis: < 1.96
    print('La desviacion de la muestra X_Variable por test bilateral es: ' + str(z_X))
    if z_X < 1.96:
        print('La hipotesis del experimento X_Variable se acepta')
    else:
        print('La hipotesis del experimento X_Variable se rechaza')
        
    # Test asimetrico con nivel de confianza 95% y alpha 0.05
    print('La media aceptable resultado del test asimetrico para Y_Variable ha de ser menor o igual a: ' + str(1 + 1.645 * (1/math.sqrt(n))) )
    z_Y = math.sqrt(n)*(abs(mu_Y-1)/1) # Desviacion de la muestra, para aceptacion de hipotesis: < 1.645
    print('La desviacion de la muestra Y_Variable por test unilateral es: ' + str(z_Y))
    if z_Y < 1.645:
        print('La hipotesis del experimento Y_Variable se acepta')
    else:
        print('La hipotesis del experimento Y_Variable se rechaza')

# fig = plt.figure()
# plt.plot(ival_low, '-o', c='red')
# plt.plot(ival_high, '-o', c='blue')
# plt.axhline(0, color='black')
# plt.xlabel('Number of experiments')
# plt.ylabel('Confidence Interval')
# plt.show()