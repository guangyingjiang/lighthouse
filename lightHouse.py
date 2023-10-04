# Part 1 is about fetching data from OpenStreetMap and visualizing it using the Pydeck library.
import osmnx as ox
import pydeck as pdk
import pandas as pd

ox.config(use_cache=True, log_console=True)
pd.options.mode.chained_assignment = None

class findLightHouse:
    def __init__(self, north, south, east, west):
        self.north = north
        self.south = south
        self.east = east
        self.west = west

    """
    get lighthouse locations from OpenStreetMap

    Returns:
        pandas.DataFrame: lighthouse locations
    """
    def getLighthouseLocations(self):
        # specify map feature as light house
        tags = {"man_made":"lighthouse"}
        # download lighthouse locations from OpenStreetMap
        lighthouse_gdf = ox.features.features_from_bbox(self.north, self.south, self.east, self.west, tags)
        data = {'lon': lighthouse_gdf.geometry.centroid.x, 'lat': lighthouse_gdf.geometry.centroid.y}
        lighthouse_locations = pd.DataFrame(data=data)
        return lighthouse_locations

    """
    plot lighthouse locations with pydeck

    Args:
        lighthouse_data: lighthouse locations in pandas.DataFrame
        pydeck_layers: list of pydeck layers

    Returns:
        list of pydeck layers: pydeck_layers
    """
    def plotLighthouseLocations(self, lighthouse_data, pydeck_layers):
        # use a lighthouse icon to represent lighthouse locations in the map
        ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/9/9b/Lighthouse_icon.svg"
        icon_data = {
            "url": ICON_URL,
            "width": 200,
            "height": 200,
            "anchorY": 200
        }

        # add icon info to lighthouse data for pydeck layer
        lighthouse_data["icon_data"] = None
        for i in lighthouse_data.index:
            lighthouse_data["icon_data"][i] = icon_data

        # Create a Pydeck map around the center of the defined region
        view_state = pdk.ViewState(
            latitude=lighthouse_data.lat.mean(),
            longitude=lighthouse_data.lon.mean(),
            zoom=8.5
        )

        # create pydeck layer from lighthouse data
        lighthouse_layer = pdk.Layer(
            type="IconLayer",
            data=lighthouse_data,
            get_icon="icon_data",
            get_size=1,
            size_scale=15,
            get_position=["lon", "lat"],
            pickable=True,
        )
        pydeck_layers.append(lighthouse_layer)

        # visualize the result
        r = pdk.Deck(layers=[lighthouse_layer], initial_view_state=view_state)
        r.to_html("lighthouse.html")
        return pydeck_layers


# In order to identify the minimal set of lighthouses required for optimal navigation, an unweighted graph can be constructed. Because we the goal is to minimize the number of nodes along a path.
# In this graph, each lighthouse is represented as a vertex, and a direct straight-line path connects lighthouses, forming the edges. Branching factor is determined based on other lighthouses located within the flight range.
# Utilizing this graph, Breadth-First Search (BFS) algorithm is employed to identify the shortest path between two lighthouses.
# In the context of our project, "shortest path" signifies reaching a target lighthouse while traversing the minimum number of docks along the way.

import math


"""
calculate distance between two coordinates

Args:
    start: dictionary of starting lighthouse location in {"lon":longitude, "lat":latitude} format
    end: dictionary of ending lighthouse location in {"lon":longitude, "lat":latitude} format

Returns:
    float: distance between starting lighthouse and ending lighthouse
"""
def calculateDistance(start, end):
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [start["lon"], start["lat"], end["lon"], end["lat"]])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radius of the Earth in kilometers (mean value)
    earth_radius = 6371  # Kilometers

    # Calculate the distance
    distance = earth_radius * c
    return distance

class PickDockLocations:
    def __init__(self, lighthouse_locations):
        battery_life = 30 # min
        flight_speed = 12 # m/s
        self.range = flight_speed * 60 * battery_life / 1000.0 # kilometers
        self.lighthouse_locations = lighthouse_locations

    """
    Breadth First Search on the lighthouse graph

    Args:
        start: dictionary of starting lighthouse location in {"lon":longitude, "lat":latitude} format
        goal: dictionary of ending lighthouse location in {"lon":longitude, "lat":latitude} format

    Returns:
        list of list: list of docks in [longitude, latitude] format
    """
    def BFS(self, start, goal):
        queue = []
        closed_set = set()
        parent = {}

        closed_set.add(str(start))
        queue.append(start)

        while queue:
            current_node = queue[0]
            queue.pop(0)

            if current_node == goal:
                return self.reconstructPath(parent, start, goal) # return found path

            for neighbor_node in self.getNeighbors(current_node):
                if str(neighbor_node) not in closed_set:
                    closed_set.add(str(neighbor_node))
                    parent[str(neighbor_node)] = current_node
                    queue.append(neighbor_node)

        return None  # No path found
 
    """
    help function to give the path from the starting dock to the end dock

    Args:
        parent: list of dictionary of parent of a node
        start: dictionary of starting lighthouse location in {"lon":longitude, "lat":latitude} format
        goal: dictionary of ending lighthouse location in {"lon":longitude, "lat":latitude} format

    Returns:
        list of list: list of docks in [longitude, latitude] format
    """
    def reconstructPath(self, parent, start, end):
        path = []
        while end != start:
            path.append([end["lon"], end["lat"]])
            end = parent[str(end)]
        path.append([start["lon"], start["lat"]])
        return list(reversed(path))

    """
    find neighboring lighthouse

    Args:
        current_point: dictionary of ending lighthouse location in {"lon":longitude, "lat":latitude} format

    Returns:
        list of dictionary: list of neighbors in {"lon":longitude, "lat":latitude} format
    """
    def getNeighbors(self, current_point):
        neighbors = []
        for i in self.lighthouse_locations.index:
            point = {"lat": self.lighthouse_locations["lat"][i], "lon": self.lighthouse_locations["lon"][i]}
            if point != current_point:  # skip its own
                distance = calculateDistance(point, current_point)
                if distance < self.range:   # find neighboring lighthouse within flight range
                    neighbors.append(point)
        return neighbors

    """
    calculate total distance from the starting dock to the end dock

    Args:
        docks: list of docks in [longitude, latitude] format

    Returns:
        float: total_distance in km
    """
    def calculateTotalDistance(self, docks):
        total_distance = 0
        last_point = None
        for dock in docks:
            point = {"lat": dock[1], "lon": dock[0]}
            if last_point:
                total_distance = total_distance + calculateDistance(point, last_point)
            last_point = point

        return total_distance


    """
    plot lighthouse locations with pydeck

    Args:
        path: list of docks in [longitude, latitude] format
        pydeck_layers: list of pydeck layers

    Returns:
        list of pydeck layers: pydeck_layers
    """
    def plotPath(self, path, pydeck_layers):
        path_data = {"path": [path]}
        df = pd.DataFrame(path_data)

        path_layer = pdk.Layer(
            "PathLayer",
            data=df,
            get_path="path",
            get_width=100,
            get_color=[255, 0, 0, 255],
            pickable=True,
            auto_highlight=True,
        )
        pydeck_layers.append(path_layer)

        view_state = pdk.ViewState(
            latitude=self.lighthouse_locations.lat.mean(),
            longitude=self.lighthouse_locations.lon.mean(),
            zoom=8.5
        )

        r = pdk.Deck(
            layers=pydeck_layers,
            initial_view_state=view_state,
        )

        r.to_html("path.html")
        return pydeck_layers


# the objective is to minimize the time required to fill all the docks. A schedule for dispatching drones and calculate the time it takes for each drone to reach its destination needs to be calculated.

# To simplify the problem, some assumptions are made:
# 1. The drone's battery usage and charging occur linearly over time.
# 2. All drones are fully charged before being dispatched.

# Given these assumptions, The ideal scenario entails the following steps:
# 1. The first drone flies to each dock, charges, and then proceeds to the next dock until it reaches the last one.
# 2. Subsequent drones are launched, and they are scheduled to arrive at the next dock just as the previous drone departs, ensuring no downtime at the docks after the first drone's visit.

# For the first drone, its task involves flying and charging continuously until it reaches the last dock.
# In contrast, for the subsequent drones, they need to wait at each dock to ensure they are fully charged and the next dock becomes available when they arrive.
# Upon calculating the time required for each drone to reach its designated dock after first drone is launched, the largest one will dictate the total time required to dispatch drones to every dock.

class SendDronesToDocks:
    def __init__(self, docks):
        self.battery_life = 30 # min
        self.charge_time = 45 # min
        self.flight_speed = 12 * 60 / 1000.0 # km/min
        self.docks = docks

    """
    calculate total time to deploy drones in all docks

    Returns:
        float: total time in mins
    """
    def calculateTotalTime(self):
        # calculate travel time between each dock
        travel_time = []
        last_point = None
        for waypoint in self.docks:
            point = {"lat": waypoint[1], "lon": waypoint[0]}
            if last_point:
                travel_time.append(calculateDistance(point, last_point) / self.flight_speed)
            last_point = point

        # calculate time to charge battery for the first drone at each dock
        charge_time = []
        charge_time.append(0)
        for time in travel_time:
            charge_time.append(time / self.battery_life * self.charge_time)

        # calculate time to reach its desination dock for each drone
        time_at_dock = []
        total_travel_time = []
        for i in range(len(travel_time)):
            if i == 0:
                # first drone just need to fly and charge until reaching last dock
                time_at_dock = charge_time
                # no need to include charging time in the last dock
                time_at_dock[-1] = 0
                # total travel time to last dock for the first drone is sum of traveling time and charging time
                time_to_destination = sum(charge_time) + sum(travel_time)
                total_travel_time.append(time_to_destination)
            else:
                time_to_destination = 0
                time_to_wait_in_dock = []
                # following drones just need to reach the dock before last occupied dock
                for t in range(len(travel_time) - i):
                    # following drones need to wait in current dock for certain amount of time, so next dock is available when the drone reaches the dock
                    if t == 0:
                        # before leaving first dock, following drones need to wait in the first dock for sum of previous drones's wait time in first and second dock
                        time_to_wait_in_dock.append(time_at_dock[0] + time_at_dock[1])
                    else:
                        # after leaving first dock, following drones need to wait in each dock until it's fully charged and next dock will be available
                        time_to_wait_in_dock.append(max(charge_time[t], time_at_dock[t+1]))
                    time_to_destination += (time_to_wait_in_dock[-1] + travel_time[t])
                time_at_dock = time_to_wait_in_dock
                total_travel_time.append(time_to_destination)

        return max(total_travel_time)

# In the context of search and rescue missions, ensuring adequate coverage is important. The coverage plot below provides a visual representation of the maximum area covered by the selected docks.
# However, the actual covered area can vary due to the time required for search and rescue operations. As the mission time increases, the effective coverage area may shrink.

# Additionally, an equally vital consideration is the response time. A heatmap could potentially illustrate how quickly a search and rescue drone can reach a specific location.
# This information could be helpful to strategically assign resources areas where swift response is essential, ensure that critical areas receive prompt attention.
# Regrettably, due to time constraints, I am unable to delve into the optimization of response times in this analysis.

class Coverage:
    def __init__(self, lighthouse_locations, docks):
        self.battery_life = 30 # min
        self.flight_speed = 12 * 60 / 1000.0 # km/min
        self.lighthouse_locations = lighthouse_locations
        self.docks = docks

    """
    plot lighthouse locations with pydeck

    Args:
        pydeck_layers: list of pydeck layers
        mission_time: search and rescue operation time in mins
    """
    def plotCoverage(self, pydeck_layers, mission_time=0):
        radius = self.flight_speed * (self.battery_life - mission_time) / 2.0 * 1000.0
        docks_data = {"docks": self.docks, "radius": [radius]*len(self.docks)}
        df = pd.DataFrame(docks_data)

        coverage_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            line_width_min_pixels=1,
            get_position="docks",
            get_radius="radius",
            get_fill_color=[255, 140, 0],
            get_line_color=[0, 0, 0],
        )

        view_state = pdk.ViewState(
            latitude=self.lighthouse_locations.lat.mean(),
            longitude=self.lighthouse_locations.lon.mean(),
            zoom=8.5
        )

        r = pdk.Deck(
            layers=[coverage_layer] + pydeck_layers,
            initial_view_state=view_state,
        )

        r.to_html("coverage.html")


def main():
    # Part 1: Find lighthouses near Stockholm
    # Define the bounding box for the region near Stockholm
    north = 59.4  # Northernmost latitude
    south = 58.5  # Southernmost latitude
    east = 18.8   # Easternmost longitude
    west = 17.8   # Westernmost longitude

    lighthouse = findLightHouse(north, south, east, west)
    lighthouse_locations = lighthouse.getLighthouseLocations();
    pydeck_layers = []
    pydeck_layers = lighthouse.plotLighthouseLocations(lighthouse_locations, pydeck_layers)

    # Part 2: Find lighthouses to install docks
    starting_point = {"lat": 59.3207673, "lon": 18.1549076} # lighthouse in Stockholm
    goal_point = {"lat": 58.7301639, "lon": 17.8764771} # lighthouse in Torö
    path_planning = PickDockLocations(lighthouse_locations)
    docks = path_planning.BFS(starting_point, goal_point)

    # print the suggested set of lighthouses and the total distance
    if docks:
        print(f'Path found with a total distance of {path_planning.calculateTotalDistance(docks)} km')
        print('Locations of selected lighthouse are at:')
        for dock in docks:
            lighthouse = {"lat": dock[1], "lon": dock[0]}
            print(f'{lighthouse}')
    else:
        print("No docks found.")

    # plot the route
    if docks:
        pydeck_layers = path_planning.plotPath(docks, pydeck_layers)


    # Part 3: Get drones from Stockholm to the docks!
    drones_to_docks = SendDronesToDocks(docks)
    # your estimate of the duration required to fill the docks
    print(f'Total time to fill drones in every dock: {drones_to_docks.calculateTotalTime()} min')


    # Part 4: Analyze and visualize any other interesting aspect of this proposal.

    coverage = Coverage(lighthouse_locations, docks)
    # Coverage with 0 mins search and rescue operation
    mission_time = 0
    coverage.plotCoverage(pydeck_layers, mission_time)

if __name__ == "__main__":
    main()
