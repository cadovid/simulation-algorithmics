import math
import random
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

t = 50 # Numero de experimentos
s = 1000 # Numero de simulaciones por experimento (o muestras)
n = 1000 # Numero de lanzamientos por simulacion
ival_low = np.array([]) # Datos de intervalo de confianza en comparacion de sistemas
ival_high = np.array([]) # Datos de intervalo de confianza en comparacion de sistemas

for i in range(t):
    """ _________________________________________________________
    
        Estimacion de pi por el metodo de la circunferencia
        _________________________________________________________
    """
    sol_1 = np.array([]) # Contenido de las estimaciones de pi
    e_1 = np.array([]) # Contenido de los errores relativos de pi
    for i in range(s):
        # Parametros iniciales y simulacion de dianas en circuferencia de radio unidad
        success = 0
        for i in range(n):
            x = random.uniform(0, 1) # Coordenada x del primer cuadrante
            y = random.uniform(0, 1) # Coordenada y del primer cuadrante
            if (x**2 + y**2 <= 1):
                success += 1
                
        # Estimacion de pi
        pi = 4 * success/n
        error_relative = ((abs(math.pi-pi))/math.pi)*100
        sol_1 = np.append(sol_1, pi)
        e_1 = np.append(e_1, error_relative)
        
    
    """ _____________________________________________________________
    
        Estimacion de pi mediante el problema de Buffon y la aguja
        _____________________________________________________________
    """
    sol_2 = np.array([]) # Contenido de las estimaciones de pi
    e_2 = np.array([]) # Contenido de los errores relativos de pi
    drops = n
    l = 1 # Longitud de la aguja
    d = 1 # Distancia entre lineas del papel
    for i in range(s):
        # Parametros iniciales y simulacion del problema de Buffon y la aguja
        hits = 0
        for i in range(n):
            center = random.uniform(0, d/2) # Centro de la aguja
            theta = random.uniform(0, math.pi) # Angulo de caida de la aguja
            distance = math.sin(theta)*l/2 # Distancia desde el centro de la aguja hasta su extremo
            if center <= distance :
                hits += 1
    
        # Estimacion de pi
        pi = 2*(l/d)*(drops/hits)
        error_relative = ((abs(math.pi-pi))/math.pi)*100
        sol_2 = np.append(sol_2, pi)
        e_2 = np.append(e_2, error_relative)
        
        
    """______________________________________________________
    
        Resultados y analisis de salida
        _____________________________________________________
    """
    # Representacion de resultados obtenidos
    data = {'Circunference': sol_1,
            'Circunference, e': e_1,
            'Buffon': sol_2,
            'Buffon, e': e_2}
    df = pd.DataFrame(data)
    #print(df)
    
    # Calculo de la distribucion t-Student con un nivel de confianza dado
    def t(alpha, gl):
        return scipy.stats.t.ppf(1-(alpha/2), gl)
    
    # Calculo de intervalos de confianza
    mu_1 = df['Circunference'].mean()
    mu_2 = df['Buffon'].mean()
    std_1 = df['Circunference'].std()
    std_2 =df['Buffon'].std()
    #print('Intervalo de confianza de 95%, Circunference: [' + str(mu_1 - t(0.05, s-1) * (std_1/math.sqrt(s))) + ', ' + str(mu_1 + t(0.05, s-1) * (std_1/math.sqrt(s))) + ']')
    #print('Intervalo de confianza de 95%, Buffon: [' + str(mu_2 - t(0.05, s-1) * (std_2/math.sqrt(s))) + ', ' + str(mu_2 + t(0.05, s-1) * (std_2/math.sqrt(s))) + ']')
    
    # Comparacion de los sistemas por medio de test t-Student (Test no pareado)
    mu = mu_1 - mu_2 # Diferencia de medias
    std = (std_1/math.sqrt(s)) + (std_2/math.sqrt(s)) # Desviacion estandar entre las medias
    #gl = s # Estimacion 1 de grados de libertad: Distinta varianza
    #gl = s+s-2 # Estimacion 2 de grados de libertad: Misma varianza
    gl = ((std_1**2/s)+(std_2**2/s))**2/(((std_1**2/s)**2/(s-1))+((std_2**2/s)**2/(s-1))) # Estimacion 3 de grados de libertad: Aproximacion de Welch
    #print('Intervalo de confianza de 95%, Comparacion de sistemas: [' + str(mu - t(0.05, gl) * std) + ', ' + str(mu + t(0.05, gl) * std) + ']')
    
    # Representacion de valores de los experimentos realizados
    ival_low = np.append(ival_low, mu - t(0.05, gl) * std)
    ival_high = np.append(ival_high, mu + t(0.05, gl) * std)

plt.plot(ival_low, '-o', c='red')
plt.plot(ival_high, '-o', c='blue')
plt.axhline(0, color='black')
plt.xlabel('Number of experiments')
plt.ylabel('Confidence Interval')
plt.show()