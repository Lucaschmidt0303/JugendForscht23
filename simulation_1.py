from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
from random import randint
import numpy as np
import time
import numba as nb

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
    v_feld_koordinaten=[]
    x, y = np.meshgrid(np.linspace(0, dx*n, n), np.linspace(0, dx*n, n)) 
    v_feld_koordinaten.append(x)
    v_feld_koordinaten.append(y)
    return v_feld_koordinaten

def coord_d(n, dx):
    d_feld_koordinaten=[]
    x,y = np.meshgrid(np.linspace(0, dx*n, n), np.linspace(0, dx*n, n))
    d_feld_koordinaten.append(x)
    d_feld_koordinaten.append(y)
    return d_feld_koordinaten


# Felder initialisieren

# Geschwindigkeitsfeld initialisieren
def init_v(n, v_feld_koordinaten, sigma, muu):
    v_feld = []
    x, y = np.exp(-( (v_feld_koordinaten[0]-muu)**2 / ( 2.0 * sigma**2 ) ) ), np.exp(-( (v_feld_koordinaten[1]-muu)**2 / ( 2.0 * sigma**2 ) ) )
    v_feld = np.array([x,y])
    return v_feld

# Dichtefeld initialisieren
def init_d(n, d_feld_koordinaten, sigma, muu):
    d_feld=[]
    x, y = np.exp(-( (d_feld_koordinaten[0]-muu)**2 / ( 2.0 * sigma**2 ) ) ), np.exp(-( (d_feld_koordinaten[1]-muu)**2 / ( 2.0 * sigma**2 ) ) )
    d_feld.append((x+y)/2)
    return d_feld[0]


#Diffusion Geschwindigkeitsfeld
#@nb.njit()
def diff_v(n, dt, viskosität, v_feld, d_feld):
    v_feld_n = v_feld[:]
    for x in range(1, n-1):
        for y in range(1, n-1):
            #Durchschnittliche Geschwindigkeit der benachbarten Fluidelemente berechnen
            v_umliegend=[]
            a=4
            if(x==0 and y==0):
                x1 = (v_feld[0][x+1][y]+v_feld[0][x][y+1])
                y1 = (v_feld[1][x+1][y]+v_feld[1][x][y+1])
                a-=2
            if(x==n-1 and y==0):
                x1 = (v_feld[0][x-1][y]+v_feld[0][x][y+1])
                y1 = (v_feld[1][x-1][y]+v_feld[1][x][y+1])
                a-=2
            if(x==0 and y==n-1):
                x1 = (v_feld[0][x+1][y]+v_feld[0][x][y-1])
                y1 = (v_feld[1][x+1][y]+v_feld[1][x][y-1])
                a-=2
            if(x==n-1 and y==n-1):
                x1 = (v_feld[0][x-1][y]+v_feld[0][x][y-1])
                y1 = (v_feld[1][x-1][y]+v_feld[1][x][y-1])
                a-=2
            if(x==0 and not(y==0 or y==n-1)):
                x1 = (v_feld[0][x+1][y]+v_feld[0][x][y+1]+v_feld[0][x][y-1])
                y1 = (v_feld[1][x+1][y]+v_feld[1][x][y+1]+v_feld[1][x][y-1])
                a-=1
            if(x==n-1 and not(y==0 or y==n-1)):
                x1 = (v_feld[0][x-1][y]+v_feld[0][x][y+1]+v_feld[0][x][y-1])
                y1 = (v_feld[1][x-1][y]+v_feld[1][x][y+1]+v_feld[1][x][y-1])
                a-=1
            if(y==0 and not(x==0 or x==n-1)):
                x1 = (v_feld[0][x-1][y]+v_feld[0][x+1][y]+v_feld[0][x][y+1])
                y1 = (v_feld[1][x-1][y]+v_feld[1][x+1][y]+v_feld[1][x][y+1])
                a-=1
            if(y==n-1 and not(x==0 or x==n-1)):
                x1 = (v_feld[0][x-1][y]+v_feld[0][x+1][y]+v_feld[0][x][y-1])
                y1 = (v_feld[1][x-1][y]+v_feld[1][x+1][y]+v_feld[1][x][y-1])
                a-=1
            if(x>0 and x<n-1 and y>0 and y<n-1):
                x1 = (v_feld[0][x-1][y] + v_feld[0][x+1][y] + v_feld[0][x][y-1] + v_feld[0][x][y+1])
                y1 = (v_feld[1][x-1][y] + v_feld[1][x+1][y] + v_feld[1][x][y-1] + v_feld[1][x][y+1])
            x1/=a
            y1/=a
            v_umliegend.append(x1)
            v_umliegend.append(y1)
            
            
            #Berechnung des Parameters k
            k = viskosität / d_feld[x][y]
            
            #Berechnung der Geschwindigkeitsänderung dv
            dv=[]
            x2 = k*(v_umliegend[0] -v_feld[0][x][y])*dt
            y2 = k*(v_umliegend[1] -v_feld[1][x][y])*dt
            dv.append(x2)
            dv.append(y2)
            #print(v_umliegend[0]-v_feld[0][x][y])
            
            #Berechnung der Werte für das Geschwindigkeitsfeld für die nächste Iteration
            v_feld_n[0][x][y] = v_feld[0][x][y] + dv[0]
            v_feld_n[1][x][y] = v_feld[1][x][y] + dv[1]
            #print(dv)
            
    return v_feld_n

#Diffusion Dichtefeld
def diff_d(n, dt, dx, viskosität, d_feld):
    d_feld_n = d_feld
    for x in range(n):
        for y in range(n):
            #Durchschnittliche Dichte der benachbarten Fluidelemente berechnen
            a=4
            if(x==0 and y==0):
                d_umliegend=(d_feld[x+1][y]+d_feld[x][y+1])
                a-=2
            if(x==n-1 and y==0):
                d_umliegend=(d_feld[x-1][y]+d_feld[x][y+1])
                a-=2
            if(x==0 and y==n-1):
                d_umliegend=(d_feld[x+1][y]+d_feld[x][y-1])
                a-=2
            if(x==n-1 and y==n-1):
                d_umliegend=(d_feld[x-1][y]+d_feld[x][y-1])
                a-=2
            if(x==0 and not(y==0 or y==n-1)):
                d_umliegend = (d_feld[x+1][y]+d_feld[x][y-1]+d_feld[x][y+1])
                a-=1
            if(x==n-1 and not(y==0 or y==n-1)):
                d_umliegend = (d_feld[x-1][y]+d_feld[x][y-1]+d_feld[x][y+1])
                a-=1
            if(y==0 and not(x==0 or x==n-1)):
                d_umliegend = (d_feld[x-1][y]+d_feld[x+1][y]+d_feld[x][y+1])
                a-=1
            if(y==n-1 and not(x==0 or x==n-1)):
                d_umliegend = (d_feld[x-1][y]+d_feld[x+1][y]+d_feld[x][y-1])
                a-=1
            if(x>0 and x<n-1 and y>0 and y<n-1):
                d_umliegend=(d_feld[x-1][y]+d_feld[x+1][y]+d_feld[x][y-1]+d_feld[x][y+1])
            d_umliegend /= a
           
            
            #Berechnung des Parameters k, calculate dt
            k = viskosität * dt / dx**2
            
            #Berechnung der Dichteänderung dd
            dd = (k*(d_umliegend-d_feld[x][y]))*dt
            
            #Berechnung der Werte für das Dichtefeld für die nächste Iteration
            d_feld_n[x][y] = d_feld[x][y] + dd
    return d_feld_n






#Main
n = 20
dt = 0.1
dx = 0.1
viskosität=10**-1
sigma = dx/2 * 10
muu = dx*n/2

# Koordinaten der Felder festlegen
v_feld_koordinaten = coord_v(n, dx)
d_feld_koordinaten = coord_d(n, dx)

# Felder initialiseren mit gaußscher Normalverteilung
v_feld = init_v(n, v_feld_koordinaten, sigma, muu)
d_feld = init_d(n, d_feld_koordinaten, sigma, muu)



# Test
# Anzahl der Iterationen, i*dt ergibt die Zeit in Sekunden, bei Nutzung der Animation = 0
i = 1000



print("##################")
print("Anzahl an Iterationen:", i)
print("##################")

# Schleife zur Berechnung der Felder
t = 0
for i in range(i):
    start=time.perf_counter()
    v_feld = diff_v(n, dt, viskosität, v_feld, d_feld)
    d_feld = diff_d(n, dt, dx, viskosität, d_feld)
    stop=time.perf_counter()
    t += stop-start
    
print(t/100*1000)

#print(v_feld[0])
print(viskosität * dt / dx**2)



# Mindest- und Maximalwert für das Dichtefeld -> Mindestwert=schwarz, Maximalwert=weiß
d_min = np.amin(d_feld)
d_max = np.amax(d_feld)



# Felder darstellen
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 8))

# Geschwindigkeitsfeld
ax1.set_title("Geschwindigkeitsfeld")
ax1.quiver(v_feld_koordinaten[0], v_feld_koordinaten[1], v_feld[0], v_feld[1], pivot="middle", scale=10)
ax1.set_xlabel("x in [m]")
ax1.set_ylabel("y in [m]")


# Dichtefeld
ax2.set_title("Dichtefeld")
ax2.pcolormesh(d_feld, cmap=cm.gray, vmin=d_min, vmax=d_max) #, shading="gouraud"
plt.xticks([])
plt.yticks([])
diff_min_max = str(np.round(np.amax(d_feld) - np.amin(d_feld), 3))
ax2.set_xlabel("x in [m] \n Größte Differenz: " + diff_min_max + r" $kg/m^2$")
ax2.set_ylabel("y in [m]")





# Felder animieren
def animate(i, n, dt, dx, viskosität, v_feld_koordinaten, v_feld, d_feld, d_min, d_max):
    plt.cla()
    v_feld = diff_v(n, dt, viskosität, v_feld, d_feld)
    d_feld = diff_d(n, dt, dx, viskosität, d_feld)
    diff_min_max = str(np.round(np.amax(d_feld) - np.amin(d_feld), 3))
    ax2.set_xlabel("x in [m] \n Größte Differenz: " + diff_min_max + r" $kg/m^2$" + "\n" + str(np.round((dt * i),1)) + " s")
    ax1.quiver(v_feld_koordinaten[0], v_feld_koordinaten[1], v_feld[0], v_feld[1], pivot="middle")
    ax2.pcolormesh(d_feld, cmap=cm.gray, vmin=d_min, vmax=d_max)
    

    
 
# Um die Animation zu nutzen "ani" nicht mehr auskommentieren
#ani = animation.FuncAnimation(plt.gcf(), animate, fargs=(n, dt, dx, viskosität, v_feld_koordinaten, v_feld, d_feld, d_min, d_max), interval=dt*1000)


plt.show()

