from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
from random import randint
import numpy as np
import time
import numba as nb
import math

#Anzahl Spalten/Zeilen -> n^2 ist die Anzahl der Fluidelemente
n=None

#Zeitänderung pro Durchgang in s
dt=None

#Breite des Elements(dy=dx) in km,m,cm,mm,etc.
dx=None

#Geschwindigkeitsfeld in m/s
v_feld_koordinaten=None
v_feld=None

#Dichtefeld in kg/m^3
d_feld_koordinaten=None
d_feld=None

#dynamische Viskosität
viskosität=None

#Variablen für die gau?sche Normalverteilung (Initialisierung der Felder)
sigma = None
muu = None

#Geschwindigkeits- und Dichtefeldkoordinaten festlegen
def coord_v(n, dx):
    x, y = np.meshgrid(np.linspace(0, dx*n, n), np.linspace(0, dx*n, n)) 
    v_feld_koordinaten = np.array([x,y], dtype=np.float32)
    return v_feld_koordinaten

def coord_d(n, dx):
    x,y = np.meshgrid(np.linspace(0, dx*n, n), np.linspace(0, dx*n, n))
    d_feld_koordinaten = np.array([x,y], dtype=np.float32)
    return d_feld_koordinaten


# Felder initialisieren

# Geschwindigkeitsfeld initialisieren
def init_v(n, v_feld_koordinaten):
    x, y = v_feld_koordinaten[0]*0, v_feld_koordinaten[1] * 0
    # Randbedingungen festlegen
    for i in range(n):
        for j in range(n):
            if i == 0:
                x[i][j] = 0.0
                y[i][j] = 0.0
            if j == 0:  
                x[i][j] = 0.0
                y[i][j] = 0.0
            if j == n-1:
                x[i][j] = 0.0
                y[i][j] = 0.0
            if i == n-1:
                x[i][j] = 1.0
                y[i][j] = 0.0
    v_feld = np.array([x,y], dtype=np.float32)
    return v_feld

# Dichtefeld initialisieren
def init_d(n, d_feld_koordinaten):
    x, y = np.zeros_like(d_feld_koordinaten)
    d_feld = np.array([x,y])
    return d_feld[0]


# Diffusion und Advektion
def diff_adv_v_feld(n, dt, dx, viskosität, v_feld, d_feld):
    # Zweidimensionales Koordinatensystem in eindimensionales Koordinatensystem umwandeln: (i,j)->k=i+(j-1)n
    v_feld_x_1d = v_feld[0].flatten()
    v_feld_y_1d = v_feld[1].flatten()
    d_feld_1d = d_feld.flatten()
    
    # c und d berechnen für jedes einzelne Feld
    c_x = np.empty_like(v_feld_x_1d)
    d_x = np.empty_like(v_feld_x_1d)
    
    c_y = np.empty_like(v_feld_y_1d)
    d_y = np.empty_like(v_feld_y_1d)
    
    for i in range(len(v_feld_x_1d)):
        c_x[i] = v_feld_x_1d[i] * dt / 2*dx
        d_x[i] = viskosität * dt / dx**2
        
        c_y[i] = v_feld_x_1d[i] * dt / 2*dx
        d_y[i] = viskosität * dt / dx**2
        
    #print(np.reshape(c_x, (n,n)))
    #print(np.reshape(d_x, (n,n)))
    # Matrix erstellen
    
    # Zuerst mehrere "leere" Matrizen mit der Größe: n^2 x n^2 erstellen
    m_x = np.identity(n**2)
    m_2_x = np.identity(n**2)
    m_y = np.identity(n**2)
    m_2_y = np.identity(n**2)
    
    m_x = m_x.astype(np.float32)
    m_2_x = m_2_x.astype(np.float32)
    m_y = m_y.astype(np.float32)
    m_2_y = m_2_y.astype(np.float32)

    
    # Matrix mit den c- und den d-Werten
    # i = Zeilen, j = Spalten
    k = 0
    for i in range(n**2):
        for j in range(n**2):
            if i >= n+1 and i < n**2-n:
                if (i+1)%n != 0 and i%n != 0: 
                    if j == i-n:
                        m_x[i][j] = -c_x[k]/2-d_x[k]
                        m_2_x[i][j] = c_x[k]/2+d_x[k]
                        m_y[i][j] = -c_y[k]/2-d_y[k]
                        m_2_y[i][j] = c_y[k]/2+d_y[k]
                    if j == i-1:
                        m_x[i][j] = -c_x[k]/2-d_x[k]
                        m_2_x[i][j] = c_x[k]/2+d_x[k]
                        m_y[i][j] = -c_y[k]/2-d_y[k]
                        m_2_y[i][j] = c_y[k]/2+d_y[k]
                    if j == i:
                        m_x[i][j] = 2*(1+2*d_x[k])  
                        m_2_x[i][j] = 2*(1-2*d_x[k])
                        m_y[i][j] = 2*(1+2*d_y[k])  
                        m_2_y[i][j] = 2*(1-2*d_y[k])
                    if j == i+1:
                        m_x[i][j] = c_x[k]/2-d_x[k]
                        m_2_x[i][j] = -c_x[k]/2+d_x[k]
                        m_y[i][j] = c_y[k]/2-d_y[k]
                        m_2_y[i][j] = -c_y[k]/2+d_y[k]
                    if j == i+n:
                        m_x[i][j] = c_x[k]/2-d_x[k]
                        m_2_x[i][j] = -c_x[k]/2+d_x[k]
                        m_y[i][j] = c_y[k]/2-d_y[k]
                        m_2_y[i][j] = -c_y[k]/2+d_y[k]    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    if i == 3+(10-1)*n or i == 3+(11-1)*n or i == 3+(12-1)*n or i == 3+(13-1)*n:
                        if j == i-n:
                            m_x[i][j] = 0
                            m_2_x[i][j] = 0
                            m_y[i][j] = 0
                            m_2_y[i][j] = 0
                        if j == i-1:
                            m_x[i][j] = 0
                            m_2_x[i][j] = 0
                            m_y[i][j] = 0
                            m_2_y[i][j] = 0
                        if j == i:
                            m_x[i][j] = 1  
                            m_2_x[i][j] = 1
                            m_y[i][j] = 1
                            m_2_y[i][j] = 1
                        if j == i+1:
                            m_x[i][j] = 0
                            m_2_x[i][j] = 0
                            m_y[i][j] = 0
                            m_2_y[i][j] = 0
                        if j == i+n:
                            m_x[i][j] = 0
                            m_2_x[i][j] = 0  
                            m_y[i][j] = 0
                            m_2_y[i][j] = 0
                                  
        k+=1
    

        
    #print(m_x)
       
       
    # Inverse der Matrix bilden, damit die Werte für die nächste Iteration berechnet werden können
    m_x_inverse = np.linalg.inv(m_x)
    m_y_inverse = np.linalg.inv(m_y)
    
    # Gleichungen lösen
    v_feld_x_1d_m = np.dot(m_2_x, v_feld_x_1d)
    v_feld_y_1d_m = np.dot(m_2_y, v_feld_y_1d)

    v_feld_x_1d_n = np.dot(m_x_inverse, v_feld_x_1d_m)
    v_feld_y_1d_n = np.dot(m_y_inverse, v_feld_y_1d_m)
    
    # in das 2d-Koordinatensystem umwandeln
    v_feld_x_n = np.reshape(v_feld_x_1d_n, (n, n))
    v_feld_y_n = np.reshape(v_feld_y_1d_n, (n, n))
    
   
    
    v_feld_n = nb.typed.List([v_feld_x_n, v_feld_y_n])

    return v_feld_n
                    



#Main
n = 20
dt = 0.1
dx = 0.1
viskosität = 10**-3

# Koordinaten der Felder festlegen
v_feld_koordinaten = coord_v(n, dx)
d_feld_koordinaten = coord_d(n, dx)

# Felder initialiseren
v_feld = init_v(n, v_feld_koordinaten)
d_feld = init_d(n, d_feld_koordinaten)



# Test
# Anzahl der Iterationen, i*dt ergibt die Zeit in Sekunden, bei Nutzung der Animation = 0
i = 50


print("##################")
print("Anzahl an Iterationen:", i)
print("##################")

# Schleife zur Berechnung der Felder
t=0

for i in range(i):
    start = time.perf_counter()
    v_feld = diff_adv_v_feld(n, dt, dx, viskosität, v_feld, d_feld)
    #d_feld = diff_adv_d_feld(n, dt, dx, viskosität, v_feld, d_feld)
    stop = time.perf_counter()
    t+=stop-start
    print(".")
    
print(t/100*1000)

#d_feld = np.sqrt(v_feld[0]**2 + v_feld[1]**2)     








# Felder darstellen
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 8))

# Geschwindigkeitsfeld
ax1.set_title("Geschwindigkeitsfeld")
ax1.quiver(v_feld_koordinaten[0], v_feld_koordinaten[1], v_feld[0], v_feld[1], pivot="middle", scale=10)
ax1.set_xlabel("x in [m]")
ax1.set_ylabel("y in [m]")


# Dichtefeld
ax2.set_title("Dichtefeld")
ax2.pcolormesh(d_feld, cmap=cm.gray) #, shading="gouraud"
plt.xticks([])
plt.yticks([])
diff_min_max = str(np.round(np.amax(d_feld) - np.amin(d_feld), 3))
ax2.set_xlabel("x in [m] \n Größte Differenz: " + diff_min_max + r" $kg/m^2$")
ax2.set_ylabel("y in [m]")




# Felder animieren
def animate(i, n, dt, dx, viskosität, v_feld_koordinaten, v_feld, d_feld, d_min, d_max):
    plt.cla()
    v_feld = diff_adv_v_feld(n, dt, dx, viskosität, v_feld, d_feld)
    #d_feld = diff_d(n, dt, dx, viskosität, d_feld)
    #diff_min_max = str(np.round(np.amax(d_feld) - np.amin(d_feld), 3))
    #ax2.set_xlabel("x in [m] \n Größte Differenz: " + diff_min_max + r" $kg/m^2$" + "\n" + str(i) + " s")
    ax1.quiver(v_feld_koordinaten[0], v_feld_koordinaten[1], v_feld[0], v_feld[1], pivot="middle")
    #ax2.pcolormesh(v_feld[0], cmap=cm.gray, vmin=d_min, vmax=d_max)

    
 
# Um die Animation zu nutzen "ani" nicht mehr auskommentieren
#ani = animation.FuncAnimation(plt.gcf(), animate, fargs=(n, dt, dx, viskosität, v_feld_koordinaten, v_feld, d_feld, d_min, d_max), interval=dt*1000)


plt.show()

