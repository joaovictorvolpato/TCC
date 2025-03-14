from pulp import *
import numpy as np
import math
import os

path = "/home/joaovolp/TCC/"

fuel_co2_cost = 1.4
driver_cost = 0.0022
gravity = 9.81
disel_efficency = 0.9
surface_area_m2 = 3.912
air_density = 1.2041
aerodynamics_drag_coef = 0.7
rolling_resistance = 0.01
k = 0.2 #friction factor
N = 33 #engine speed
V = 5 #engine displacement
lambda_ = 1/(44*737) # lambda = Espilon/heating value * conversion factor
ntf = 0.4
n = 0.9
gamma = 1/(1000 * ntf * n)


a = 0 # a for accelaration -> we assume constant velocity
alpha = a + gravity * math.sin(0) + gravity * rolling_resistance * math.cos(0)
beta = 0.5 * aerodynamics_drag_coef * surface_area_m2 * air_density


class City:
    def __init__(self, num : int, name : str, demand : int, readyTime : int, dueTime : int, serviceTime : int):
        self.id = int(num)
        self.name = name
        self.demand = int(demand)
        self.readyTime = int(readyTime)
        self.dueTime = int(dueTime)
        self.serviceTime = int(serviceTime)

    def __str__(self):
        return f"City {self.id}: {self.name} (Demand: {self.demand}, Ready: {self.readyTime}, Due: {self.dueTime}, Service: {self.serviceTime})"

class Data:
    def __init__(self, filename : str):
        self.filename = filename
        self.size = int
        self.max_load = int
        self.weight = int
        self.maxSpeed = int
        self.minSpeed = int
        self.distances = {}
        self.cities = []
        self.velocities = []

    def truncate_float(self,float_number):
        multiplier = 10 ** 2
        return int(float_number * multiplier) / multiplier
        
    def FetchInstance(self):
        with open(f"{path}{self.filename}","r") as file:
            content = file.readlines()
            lines = [line.strip() for line in content if line.strip()]
            self.size = int(lines[0])
            self.curbweight, _, _ = lines[1].strip().partition('\t')
            self.curbweight = int(self.curbweight)
            _, _, self.max_load = lines[1].strip().partition('\t')
            self.max_load = int(self.max_load)
            self.minSpeed, _, _ = lines[2].strip().partition('\t')
            self.minSpeed = int(self.minSpeed)/3.6
            _, _, self.maxSpeed = lines[2].strip().partition('\t')
            self.maxSpeed = int(self.maxSpeed)/3.6

            discrete_points = 10
            step = (int(self.maxSpeed) - int(self.minSpeed)) / (discrete_points - 1)
            
            self.velocities = list(self.minSpeed + x * step for x in range(discrete_points))

            self.velocities = list(map(self.truncate_float, self.velocities))

            del lines[0]
            del lines[0]
            del lines[0]

            for i in range(self.size+1):
                #print(i)
                distlist = [x for x in lines[i].split('\t')]
                #print(distlist)
                for j in range(len(distlist)):
                    self.distances[(i,j)] = int(distlist[j])
            
            for i in range(self.size+1):
                del lines[0]

            for citinfo in lines:
                #print(citinfo)
                info = citinfo.split('\t')
                info = list(filter(lambda x: x != '', info))
                name,_,_ = info[1].strip().partition(' ')
                _,_,demand = info[1].strip().partition(' ')
                city = City(num=info[0], name=name, demand=demand, readyTime=info[2], dueTime=info[3], serviceTime=info[4])
                self.cities.append(city)

            print("max_Load:",self.max_load)

            print("Curbweight:",self.curbweight)


def build_model(instance: str) -> LpProblem:
    data = Data(instance)
    data.FetchInstance()

    prp = LpProblem("Pollution_Rounting_Problem", LpMinimize)

    cities_demands = [x.demand for x in data.cities]
    cities_arrival = [x.readyTime for x in data.cities]
    cities_dueTime = [x.dueTime for x in data.cities]
    cities_service = [x.serviceTime for x in data.cities]

    #arches = list(filter(lambda x: len(x) == 2, [tuple(c) for c in allcombinations(cities_id,2)]))

    #variáveis de decisão
    x_ij = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1 ,cat="Integer") for j in range(data.size + 1)] for i in range(data.size + 1)]
    z_ij = [[[pulp.LpVariable(f"z_{r}_{i}_{j}", lowBound=0, upBound=1 ,cat="Integer") for j in range(data.size + 1)] for i in range(data.size + 1)] for r in range(len(data.velocities))]
    f_ij = [[pulp.LpVariable(f"f_{i}_{j}", cat="Continuous", lowBound=0) for j in range(data.size + 1)] for i in range(data.size + 1)]
    y_i = [pulp.LpVariable(f"y_{i}", cat="Continuous",lowBound=0) for i in range(data.size+1)]
    s_j = [pulp.LpVariable(f"s_{j}", cat="Continuous", lowBound=0) for j in range(data.size + 1)]

    l_j = []
    for x in range(data.size + 1):
        l_j.append(cities_dueTime[x] + cities_service[x] + data.distances[x,0]/data.minSpeed)

    #obejective function

    #Lj -> worst case end of time window of node j + service time on j + time on j -> 0 on lowest velocity possible

    pt1 = lpSum(fuel_co2_cost * lambda_* gamma * alpha * data.distances[i,j] * data.curbweight * x_ij[i][j] for i in range(data.size + 1) for j in range(data.size + 1))

    pt2 = lpSum(fuel_co2_cost * lambda_* gamma * alpha * f_ij[i][j] * data.distances[i,j] for i in range(data.size + 1) for j in range(data.size + 1))

    #vel = lpSum(data.velocities[r]**2 * z_ij[r][i][j] for r in range(len(data.velocities)))

    pt3 = lpSum(fuel_co2_cost * lambda_* gamma  * beta * data.distances[i,j] * lpSum(data.velocities[r]**2 * z_ij[r][i][j] for r in range(len(data.velocities))) for j in range(data.size + 1) for i in range(data.size + 1) )

    pt4 = lpSum(driver_cost * s_j[j] for j in range(data.size + 1) )

    pt5 = lpSum(fuel_co2_cost * k * N * V * lambda_ * data.distances[i,j] * lpSum(z_ij[r][i][j] * 1 / data.velocities[r]  for r in range(len(data.velocities))) for j in range(data.size + 1) for i in range(data.size + 1))

    prp += pt1 + pt2 + pt3 + pt4 + pt5

    #restrições 

    #prp += (
    #    lpSum(y_i[j] + cities_service[j] + (data.distances[j,0]/data.velocities[r] *  z_ij[r][i][0]) for i in range(data.size + 1) for j in range(data.size + 1) for r in range(len(data.velocities)) == s_j[j] for j in range(data.size + 1))
    #)

    prp += (
        lpSum(x_ij[0][j] for j in range(data.size + 1)) <= data.size #4.8
    )

    for i in range(1,data.size+1): #4.9
        prp += lpSum(x_ij[i][j] for j in range(data.size+1)) == 1

    for j in range(1,data.size+1): #4.10
        prp += lpSum(x_ij[i][j] for i in range(data.size+1)) == 1

    for i in range(1, data.size+1): #4.11
        prp += (
            lpSum(f_ij[j][i] for j in range(data.size + 1)) - lpSum(f_ij[i][j] for j in range(data.size + 1)) == cities_demands[i]
        )

    for i in range(data.size + 1): # 4.12
        for j in range(data.size + 1):
            prp += (
                cities_demands[j]*x_ij[i][j] <= f_ij[i][j]
            )

    for i in range(data.size + 1): # 4.12
        for j in range(data.size + 1):
            prp += (
                f_ij[i][j] <= (data.max_load - cities_demands[i])*x_ij[i][j]
            )

    for i in range(1,data.size + 1): #4.14
        prp += (
            cities_arrival[i] <= y_i[i]
        )

    for i in range(1,data.size + 1): #4.14
        prp += (
            y_i[i] <= cities_dueTime[i]
        )

    for i in range(data.size + 1): #4.13
        for j in range(1,data.size + 1):
            if i != j:
                Mij = max(0, cities_dueTime[i] + cities_service[i] + data.distances[i,j]/data.minSpeed - cities_arrival[j])
                prp += (
                    y_i[i] - y_i[j] + cities_service[i] + lpSum((data.distances[i,j]/data.velocities[r])*z_ij[r][i][j] for r in range(len(data.velocities))) <= Mij * (1 - x_ij[i][j])
                )

    for j in range(1,data.size + 1):
        prp += (
            y_i[j] + cities_service[j] - s_j[j]  + lpSum((data.distances[j,0]/data.velocities[r])*z_ij[r][j][0] for r in range(len(data.velocities))) <= l_j[j] * (1 - x_ij[j][0])
        )

    for i in range(data.size + 1): # 4.16
        for j in range(data.size + 1):
            if i != j:
                prp += lpSum(z_ij[r][i][j] for r in range(len(data.velocities))) == x_ij[i][j]

    for i in range(1, data.size + 1): #4.20
        for j in range(1, data.size + 1):
            prp += lpSum(x_ij[i][j] + x_ij[j][i]) <= 1

    for i in range(1, data.size + 1):
        prp += y_i[i] - lpSum(max(0, cities_arrival[j] - cities_arrival[i] + cities_service[j] + data.distances[j,i]/data.velocities[r])*z_ij[r][j][i] for r in range(len(data.velocities)) for j in range(data.size + 1)) >= cities_arrival[i]
    
    for i in range(1, data.size + 1):
        prp += y_i[i] + lpSum(max(0,cities_dueTime[i] - cities_dueTime[j] + cities_service[i] + data.distances[i,j]/data.velocities[r])*z_ij[r][i][j] for r in range(len(data.velocities)) for j in range(data.size + 1)) <= cities_dueTime[i]
    return prp

model = build_model("UK10_01.txt")

model.solve()

        



