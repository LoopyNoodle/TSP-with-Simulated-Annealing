import numpy as np
import matplotlib.pyplot as plt

N_cities = 20 # number of cities
city_coord = {} # a dictionary that will contain the coordinates of each of the cities
dist_list = [] # we will be keeping a record distances at every step

# Generating positions of cities to visit
for i in range(N_cities):
    city_coord[i] = (np.random.randint(0, 20), np.random.randint(0, 20))

initial_path = city_coord.copy() # a copy of the initial path for comparison

def euc_dist(x_1, x_2, y_1, y_2):
    '''
    This function calculates the distance between any two points (x_i, y_i)
    '''
    return np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)

def total_dist(city_coord):
    '''
    This function returns the total distance covered by the walker while goimg through each of the cities
    '''
    distance = 0
    for i in range(0, len(city_coord)):
        next_i = (i + 1) % len(city_coord) # to handle wrap-around (last city to the first city)
        distance += euc_dist(city_coord[i][0], city_coord[i][1], city_coord[next_i][0], city_coord[next_i][1])
    return distance

def swap(city_coord):
    '''
    This function first generates random indices between the range 1 (we are not considering the starting point) and the last index of city_coord. Then, for the x
    and y coordinates, it chooses a random index from those set of indices, and using those, we can swap the values of the dictionary.
    '''
    cities_new = city_coord.copy()
    index = range(1, len(city_coord) - 1)

    i = np.random.randint(index[0], index[-1])
    j = np.random.randint(index[0], index[-1])

    while i == j:
        j = np.random.randint(index[0], index[-1])
        
    cities_new[i], cities_new[j] = cities_new[j], cities_new[i]

    return cities_new
    
def cooling(temp, gamma):
    '''
    After every iteration, we are decreasing the temperature (in our case some way to quantify the randomness)
    '''
    return temp * gamma
    
def checking(temp, new_f, old_f):
    '''
    If the distance is less than the shortest distance recorded so far, the new path is accepted with a probability e^(-(new distance - old distance)/temperature).
    Returns True if accepted, else False.
    '''
    prob = min(1, np.exp(-(new_f - old_f)/temp))
    if prob > np.random.uniform(0, 1):
        return True
    else:
        return False

def main(temp, gamma, iters, plot_int = 200):
    '''
    This part combines all the previous functions together to generate the best path. Parameters:
    temp: The 'Temperature' of the system is essentially a measure of randomness with withich changes are made to the path. It is a control parameter.
    gamma: The rate at which the system is 'cooling', or settling to a global minimum (best path in our case)
    iters: The number of iterations
    plot_int: The time interval for the animation part
    '''
    global best_tour, final_coord, best_dist
    cur_tour = city_coord.copy()
    cur_dist = total_dist(cur_tour)

    best_tour = cur_tour.copy()
    best_dist = cur_dist

    fig, ax = plt.subplots()
    plt.ion() # interactive mode

    for i in range(iters):
        dist_list.append(cur_dist)
        prop_tour = swap(cur_tour)
        prop_dist = total_dist(prop_tour)

        if prop_dist < cur_dist:
            cur_tour = prop_tour
            cur_dist = prop_dist
        else:
            if checking(temp, prop_dist, cur_dist) == True:
                cur_tour = prop_tour
                cur_dist = prop_dist

        if cur_dist < best_dist:
            best_tour = cur_tour.copy()
            best_dist = cur_dist

        temp = cooling(temp, gamma)

        # Animating paths
        if i % plot_int == 0 or i == iters - 1:
            ax.clear()
            final_coord = list(cur_tour.values())
            x_coord = [i[0] for i in final_coord]
            y_coord = [i[1] for i in final_coord]

            ax.plot(x_coord + [x_coord[0]], y_coord + [y_coord[0]], 'o-',c = 'teal', label = f'Iter: {i}\nDist: {round(cur_dist, 3)}')
            # ax.scatter(x_coord, y_coord, c = 'midnightblue')
            ax.set_title('TSP Paths with Simulated Annealing')
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.legend()
            ax.grid(True)
            plt.pause(0.001)

        plt.ioff()

def best_path(best_tour, best_dist):
    '''
    This function plots the final best path and how the total distance decreases with the iterations
    '''
    final_coord = list(best_tour.values())
    x_coord = [i[0] for i in final_coord]
    y_coord = [i[1] for i in final_coord]

    plt.figure()
    plt.plot(x_coord + [x_coord[0]], y_coord + [y_coord[0]], 'o-',c = 'teal', label = f'Best Distance: {round(best_dist, 3)}')
    plt.title('Best Tour')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(dist_list, c = 'midnightblue')
    plt.title(r'Total Distance $vs$ Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    plt.grid(True)
    plt.show()

iters = 10000 # number of iterations
gamma = 0.99 # rate of cooling
temp = 100 # initial temperature

main(temp, gamma, iters)
best_path(best_tour, best_dist)
print(f'\n Initial Path for {N_cities} Cities: ', initial_path, '\n')
print(f'Best Path for {N_cities} Cities: ', best_tour)