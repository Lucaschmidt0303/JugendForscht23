import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
import time
import numba as nb
from numba import prange
import xlsxwriter
from matplotlib.patches import FancyArrowPatch

# i=Zeilen, j=Spalten

def init(n, l):
    # v_x Ausgangsbedingungen
    v_x = np.ones(shape=(n+1, n), dtype=np.float64)
    
    for i in range(len(v_x[0])):
        v_x[n-1][i] = 1
        v_x[n][i] = 1
          
     
    
    # v_y Ausgangsbedingungen
    v_y = np.zeros(shape=(n, n+1), dtype=np.float64)   
    
    
    # p Ausgangsbedingungen
    p = np.ones(shape=(n+1, n+1), dtype=np.float64)
    
    # Divergenz
    div = np.zeros(shape=(n+1, n+1), dtype=np.float64)
    
    return v_x, v_y, p, div
    
    
@nb.njit(parallel=True)
def v_x_berechnen(v_x, v_y, p, Reynold, n, dt, dx):
    v_x_neu = np.zeros_like(v_x)
    for i in prange(1, n):
        for j in prange(1, n-1):
            v_x_neu[i][j] = v_x[i][j] - (dt/dx)*((v_x[i][j+1]**2-v_x[i][j-1]**2)/2 + 0.25*((v_x[i][j]+v_x[i+1][j])*(v_y[i][j]+v_y[i][j+1])-(v_x[i][j]+v_x[i-1][j])*(v_y[i-1][j+1]+v_y[i-1][j])) + (p[i][j+1]-p[i][j]) - (1/(Reynold*dx)) * (v_x[i][j+1]+v_x[i][j-1]+v_x[i-1][j]+v_x[i+1][j]-4*v_x[i][j]))
    
    # Randbedingungen
    for i in prange(1, n):
        v_x_neu[i][0] = 0 #links
        v_x_neu[i][n-1] = 0 #rechts
    
    for i in prange(n):
        v_x_neu[n][i] = 2 - v_x_neu[n-1][i] #oben
        v_x_neu[0][i] = - v_x_neu[1][i] #unten
        
        
    return v_x_neu

@nb.njit(parallel=True)
def v_y_berechnen(v_x, v_y, p, Reynold, n, dt, dx):
    v_y_neu = np.empty_like(v_y)
    for i in prange(1, n-1):
        for j in prange(1, n):
            v_y_neu[i][j] = v_y[i][j] - (dt/dx)*( 0.25 * ((v_y[i][j]+v_y[i][j+1])*(v_x[i+1][j]+v_x[i][j])-(v_y[i][j]+v_y[i][j-1])*(v_x[i+1][j-1]+v_x[i][j-1])) + ((v_y[i+1][j]**2-v_y[i-1][j]**2)/2) + (p[i+1][j]-p[i][j]) - (1/(Reynold*dx))*(v_y[i][j+1]+v_y[i][j-1]+v_y[i-1][j]+v_y[i+1][j]-4*v_y[i][j]))       
            
    # Randbedingungen
    for i in prange(1, n-1):
        v_y_neu[i][0] = - v_y_neu[i][1] #links
        v_y_neu[i][n] = - v_y_neu[i][n-1] #rechts
    
    for i in prange(0, n+1):
        v_y_neu[n-1][i] = 0 #oben
        v_y_neu[0][i] = 0 #unten
        
    return v_y_neu
    
@nb.njit(parallel=True)
def p_berechnen(v_x, v_y, p, Reynold, n, dt, delta):
    p_neu = np.ones_like(p)
    for i in prange(1, n):
        for j in prange(1, n):
            p_neu[i][j] = p[i][j] - dt * delta * ((v_x[i][j]-v_x[i][j-1]+v_y[i][j]-v_y[i-1][j])/dx)
    
    # Randbedingungen
    for i in prange(n+1):
        p_neu[i][0] = p_neu[i][1]
        p_neu[i][n] = p_neu[i][n-1]
        
    for i in prange(1, n):
        p_neu[0][i] = p_neu[1][i]
        p_neu[n][i] = p_neu[n-1][i]
    
    return p_neu

@nb.njit(parallel=True)
def error_berechnen(v_x, v_y, div, divergenz):
    divergenz = 0
    for i in prange(1, n):
        for j in prange(1, n):
            div[i][j] = (v_x[i][j]-v_x[i][j-1]+v_y[i][j]-v_y[i-1][j])/dx
            divergenz += np.absolute(div[i][j])
    
    return divergenz, div

def toGrid(v_x, v_y, p, n):
    v_x_grid = np.empty(shape=(n,n))
    v_y_grid = np.empty(shape=(n,n))
    p_grid = np.empty(shape=(n,n))
    
    for i in range(n):
        for j in range(n):
            v_x_grid[i][j] = 0.5 * (v_x[i][j]+v_x[i+1][j])
            v_y_grid[i][j] = 0.5 * (v_y[i][j]+v_y[i][j+1])
            p_grid[i][j] = 0.25 * (p[i][j]+p[i][j+1]+p[i+1][j]+p[i+1][j+1])
    
    return v_x_grid, v_y_grid, p_grid

# Variablen
n = 64 # Standard Grid
l = 1
Reynold = 100 # Reynoldsnumber
dx = l/n
dt = 0.0001
delta = 5 # künstliche kompressibilitätsdelta
divergenz = 1000000
divergenz_grenze = 0.01 # Error bei der Divergenz -> falls es unter diesen Wert fällt ist ist die Simulation beendet
v_feld_koordinaten = np.meshgrid(np.linspace(0, n, n), np.linspace(0, n, n))

# Initialisierung der Felder
v_x, v_y, p, div = init(n, l)

# Anzahl der Frames
frames = 200


# Schleife zur Berechnung der Felder
i = 0
t = 0


while(divergenz/n**2 > divergenz_grenze):
    v_x_n = v_x_berechnen(v_x, v_y, p, Reynold, n, dt, dx)
    v_y_n = v_y_berechnen(v_x, v_y, p, Reynold, n, dt, dx)
    p_n = p_berechnen(v_x_n, v_y_n, p, Reynold, n, dt, delta)
    v_x = v_x_n
    v_y = v_y_n
    p = p_n
    divergenz, div = error_berechnen(v_x, v_y, div, divergenz)
    if(i%2500 == 0):
        print(divergenz/n**2)
        
    
    i+=1 

# Werte für das "standard" Gitter berechnen
v_x_final, v_y_final, p_final = toGrid(v_x, v_y, p, n)


# Felder darstellen
fig, ax = plt.subplots(2,2, figsize=(10, 10))


# Geschwindigkeitsfeld
ax[0,0].set_title("Geschwindigkeitsfeld")
ax[0,0].streamplot(v_feld_koordinaten[0],v_feld_koordinaten[1], u=v_x_final, v=v_y_final, density=4)
#v_feld = ax[0,0].quiver(v_x_final, v_y_final, pivot="middle")
#ax[0,0].set_ylim(ax[0,0].get_ylim()[1], ax[0,0].get_ylim()[0])
ax[0,0].set_xlabel("x in [m]")
ax[0,0].set_ylabel("y in [m]")

# v_x
ax[0,1].set_title("v_x")
data_v_x = ax[0,1].pcolormesh(v_x_final, cmap=cm.plasma) #cm.get_cmap("plasma",5)
cb_v_x = plt.colorbar(data_v_x, ax=ax[0,1])

# v_y
ax[1,1].set_title("v_y")
data_v_y = ax[1,1].pcolormesh(v_y_final, cmap=cm.plasma)
cb_v_y = plt.colorbar(data_v_y, ax=ax[1,1])


#p
ax[1,0].set_title("P")
data_p = ax[1,0].pcolormesh(p_final, cmap=cm.plasma)
cb_p = plt.colorbar(data_p, ax=ax[1,0])

'''
q = ax[0,0].quiver(v_feld_koordinaten[0], v_feld_koordinaten[1], v_x_final, v_y_final, pivot="middle")
a = ax[0,1].pcolormesh(v_x_final, cmap=cm.plasma)
b = ax[1,1].pcolormesh(v_y_final, cmap=cm.plasma)
c = ax[1,0].pcolormesh(p_final, cmap=cm.plasma)
zeit = ax[0,0].text(70,-10, 100*dt)
'''
'''
# Ergebnisse in eine excel datei schreiben datei schreiben
workbook = xlsxwriter.Workbook("Ergebnisse_10000.xlsx")
worksheet = workbook.add_worksheet()

# Überschriften
worksheet.write(2,1,"y")
worksheet.write(2,2,"x = 0.05")
worksheet.write(2,3,"x = 0.1")
worksheet.write(2,4,"x = 0.5")
worksheet.write(2,5,"x = 0.9")
worksheet.write(2,6,"x = 0.95")

#Daten
for i in range(n):
    worksheet.write(3+i, 1, i/(n-1))
    worksheet.write_number(3+i, 2, np.round(v_x_final[i][5],4))
    worksheet.write_number(3+i, 3, np.round(v_x_final[i][10],4))
    worksheet.write_number(3+i, 4, np.round((v_x_final[i][int((n/2)+0.5)] + v_x_final[i][int((n/2)-0.5)])/2,4))
    worksheet.write_number(3+i, 5, np.round(v_x_final[i][90],4))
    worksheet.write_number(3+i, 6, np.round(v_x_final[i][95],4))

workbook.close()

'''

'''
# Felder animieren
def animate(i, n, dt, dx, Reynold, ax, delta, div, divergenz):
    global v_x
    global v_y
    global p
    
    for s in range(150):
        v_x_n = v_x_berechnen(v_x, v_y, p, Reynold, n, dt, dx)
        v_y_n = v_y_berechnen(v_x, v_y, p, Reynold, n, dt, dx)
        p_n = p_berechnen(v_x_n, v_y_n, p, Reynold, n, dt, delta)
        v_x = v_x_n
        v_y = v_y_n
        p = p_n
        divergenz, div = error_berechnen(v_x, v_y, div, divergenz)
    v_x_final, v_y_final, p_final = toGrid(v_x, v_y, p, n)
    q.set_UVC(v_x_final, v_y_final)
    #ax[0,0].collections = []
    #for artist in ax[0,0].get_children():
        #if isinstance(artist, FancyArrowPatch):
            #artist.remove()
    #ax[0,0].streamplot(v_feld_koordinaten[0],v_feld_koordinaten[1], u=v_x_final, v=v_y_final, density=4)
    ax[0,1].pcolormesh(v_x_final, cmap=cm.plasma)
    ax[1,1].pcolormesh(v_y_final, cmap=cm.plasma)
    ax[1,0].pcolormesh(p_final, cmap=cm.plasma)
    zeit.set_text(round(dt*150*i, 3))
    #print("hallo")
    #d_feld = diff_d(n, dt, dx, viskosität, d_feld)
    #diff_min_max = str(np.round(np.amax(d_feld) - np.amin(d_feld), 3))
    #ax2.set_xlabel("x in [m] \n Größte Differenz: " + diff_min_max + r" $kg/m^2$" + "\n" + str(i) + " s")
    #ax1.quiver(v_feld_koordinaten[0], v_feld_koordinaten[1], v_x_final, v_y_final, pivot="middle")
    #ax2.pcolormesh(v_feld[0], cmap=cm.gray, vmin=d_min, vmax=d_max)
    #print(str(i/frames*100) + "%")

    
 
# Um die Animation zu nutzen "ani" nicht mehr auskommentieren
ani = animation.FuncAnimation(plt.gcf(), animate, fargs=(n, dt, dx, Reynold, ax, delta, div, divergenz), interval=16, blit=False)

#video = animation.FFMpegWriter(fps=60)
#ani.save("test3.mp4", writer = video)

'''
plt.show()