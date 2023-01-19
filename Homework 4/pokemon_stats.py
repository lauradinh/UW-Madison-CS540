import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename) as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)
        for row in data:
            row['Attack'] = int(row['Attack'])
            row['Sp. Atk'] = int(row['Sp. Atk'])
            row['Speed'] = int(row['Speed'])
            row['Defense'] = int(row['Defense'])
            row['Sp. Def'] = int(row['Sp. Def'])
            row['HP'] = int(row['HP'])
            del row['Generation']
            del row['Legendary']
        updated_data = data[:20]
    return updated_data

def calculate_x_y(stats):
    x = stats.get('Attack') + stats.get('Sp. Atk') + stats.get('Speed')
    y = stats.get('Defense') + stats.get('Sp. Def') + stats.get('HP')
    return (x, y)

def math_distance(x,y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def hac(dataset):
    # removes all NaNs or infs from dataset
    for i in dataset:
        for j in i:
            if not math.isfinite(j) or not math.isfinite(j):
                dataset.remove(i)

    # creates matrix
    rows, cols = (len(dataset) - 1, 4)
    z = [[None for i in range(cols)] for j in range(rows)]

    # distance matrix
    rows, cols = (len(dataset), len(dataset))
    dist_array = [[0] * cols] * rows
    dist_array = np.array(dist_array, dtype=float)
    for i, element1 in enumerate(dataset):
        for j, element2 in enumerate(dataset):
            dist_array[i][j] = math_distance(element1, element2)
            # turns tuples into a list
    for i, element in enumerate(dataset):
        set = []
        set.append(element)
        dataset[i] = set

    for row in range(len(z)):
        index_of_i = None
        cluster_of_i = None
        index_of_j = None
        cluster_of_j = None
        distance = None
        for i in range(len(dist_array)):
            for j, dist in enumerate(dist_array[i]):
                if dist == 0:
                    break

                if dist == -1:
                    continue
                duplicate = False
                for element in dataset:
                    if dataset[i][0] in element and dataset[j][0] in element:
                        duplicate = True
                        break
                if duplicate:
                    continue

                elif distance == None or dist < distance:
                    # finding cluster and their names
                    for index1, element1 in reversed(list(enumerate(dataset))):
                        if dataset[i][0] in element1:
                            cluster_of_i = element1
                            index_of_i = index1
                            break
                    for index2, element2 in reversed(list(enumerate(dataset))):
                        if dataset[j][0] in element2:
                            cluster_of_j = element2
                            index_of_j = index2
                            break
                    distance = dist

        if (index_of_j < index_of_i):
            z[row][0] = index_of_j
            z[row][1] = index_of_i
        else:
            z[row][0] = index_of_i
            z[row][1] = index_of_j
        z[row][2] = distance
        set = cluster_of_j + cluster_of_i
        dataset.append(set)
        z[row][3] = len(dataset[len(dataset) - 1])
        finish = False
        for i in range(len(dist_array)):
            for j, dist in enumerate(dist_array[i]):
                if dist == distance:
                    dist_array[j][i] = -1
                    finish = True
                    break
            if finish:
                break
    return (np.asarray(z))

def random_x_y(m):
    random_list = []
    for i in range(m):
        x = random.randint(1, 359)
        y = random.randint(1, 359)
        random_list.append((x,y))
    return random_list

def imshow_hac(dataset):
    orig_dataset = []
    for element in dataset:
        orig_dataset.append(element)
    z = hac(dataset)
    x,y = zip(*orig_dataset)
    plt.scatter(x,y)
    for row in z:
        min = None
        x = None
        y = None
        for pt1 in dataset[int(row[0])]:
            for pt2 in dataset[int(row[1])]:
                if min == None or math_distance(pt1,pt2) < min:
                    min = math_distance(pt1,pt2)
                    x = pt1
                    y = pt2
        x_values = [x[0], y[0]]
        y_values = [x[1], y[1]]
        plt.plot(x_values,y_values)
        plt.pause(0.1)
    plt.draw()

if __name__ == '__main__':
    data = load_data('Pokemon.csv')
    pokemon_list = []
    for element in data:
        pokemon_list.append(calculate_x_y(element))
    imshow_hac(pokemon_list)
    
