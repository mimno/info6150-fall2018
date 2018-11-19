import csv
from collections import Counter

airport_names = {}
flight_counter = Counter()

with open("airports.txt") as infile:
    reader = csv.reader(infile)
    
    for row in reader:
        airport_names[row[4]] = row[1]

with open("routes.txt") as infile:
    reader = csv.reader(infile)
    
    for row in reader:
        if not row[2] in airport_names:
            continue
        if not row[4] in airport_names:
            continue
        
        if row[2] > row[4]:
            key = "{}\t{}".format(airport_names[row[2]], airport_names[row[4]])
        else:
            key = "{}\t{}".format(airport_names[row[4]], airport_names[row[2]])
        flight_counter[key] += 1

for route in flight_counter.keys():
    print("{}\t{}".format(route, flight_counter[route]))