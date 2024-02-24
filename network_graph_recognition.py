# The MIT License (MIT)

# Copyright (C) 2024 Charly Schmidt alias Picorims <picorims.contact@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import cv2 as cv
import numpy as np
import numpy.ma as ma
import networkx as nx
import sys
import matplotlib.pyplot as plt
import os
import json
import re

def is_white(pixel):
    return pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255

def is_black(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0

def is_gray(pixel):
    return pixel[0] == pixel[1] and pixel[1] == pixel[2] and not is_white(pixel) and not is_black(pixel)

def same_color(pixel1, pixel2):
    return pixel1[0] == pixel2[0] and pixel1[1] == pixel2[1] and pixel1[2] == pixel2[2]

def exists(pixel, rows, cols):
    return pixel[0] >= 0 and pixel[0] < rows and pixel[1] >= 0 and pixel[1] < cols

def has_tuple(tutple, list):
    for t in list:
        if t[0] == tutple[0] and t[1] == tutple[1]:
            return True
    return False

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def bgr_to_rgb(bgr):
    return (bgr[2], bgr[1], bgr[0])

def get_color_speed_from_bgr(bgr, config):
    hex = rgb_to_hex(bgr_to_rgb(bgr))
    if hex == "#000000":
        return 1
    elif not (hex in config["networks_speed"]):
        print(f"Color {hex} not found in config file, defaulting to 1")
        return 1
    else:
        return config["networks_speed"][rgb_to_hex(bgr_to_rgb(bgr))]
    
def setup_out_folder():
    if not os.path.exists("out"):
        os.makedirs("out")

def make_serializable(obj):
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        print(f"Unserializable object: {obj}, type: {type(obj)}, name: {obj.__class__.__name__}")

def thickerDiagonal(img):
    # To parallelize / optimize with numpy, see thickerDiagonalNumpy
    rows,cols,_ = img.shape
    for i in range(0, rows-1):
        print("%d%%\r" % (i/rows*100), end="") # progress percentage
        for j in range(0, cols-1):
            if (is_white(img[i,j]) and is_white(img[i+1,j+1]) and not is_white(img[i,j+1]) and not is_white(img[i+1,j])):
                img[i,j] = img[i+1,j]
            if (is_white(img[i,j+1]) and is_white(img[i+1,j]) and not is_white(img[i,j]) and not is_white(img[i+1,j+1])):
                img[i+1,j] = img[i,j]
    return img

def whitePixelsMask(img):
    return (img == [255, 255, 255]).all(axis=2) # logical and on the 3 channels, which is at depth 2 from 0



def thickerDiagonalNumpy(img, name):
    # To parallelize / optimize with numpy
    # boolean array
    print("white pixels mask...")
    white_pixels = whitePixelsMask(img)

    # Create shifted versions of the array for comparison
    print("shifter masks...")
    shifted_right = np.roll(white_pixels, -1, axis=1)
    shifted_down = np.roll(white_pixels, -1, axis=0)
    shifted_right_down = np.roll(shifted_right, -1, axis=0)

    # Create a mask for the condition in the original code
    print("conditions...")
    # top left must be white, bottom right must be white, top right and bottom left must not.
    # The edited pixel is the one at the top left (1st condition of the line on which bitwise operations are applied)
    condition1 = white_pixels & shifted_right_down & ~shifted_right & ~shifted_down

    # top right must be white, bottom left must be white, top left and bottom right must not.
    # The edited pixel is the one at the top right (1st condition of the line on which bitwise operations are applied)
    condition2 = shifted_right & shifted_down & ~white_pixels & ~shifted_right_down


    # Apply the condition to the image
    print("masking...")
    img[condition1] = img[np.roll(condition1, 1, axis=1)] # copy the color of the pixel to its right
    img[np.roll(condition2, 1, axis=1)] = img[condition2] # copy the color of the pixel to its left
    cv.imwrite(f"out/processing_of_{name}.png", img)

    # DEBUG
    # img_cond_1 = img.copy()
    # img_cond_1[condition1] = np.full(np.shape(img_cond_1[condition1]), [0, 255, 0])
    # # img_cond_1 is based on top left corner, so we need to roll the condition to the right to edit the top right corner
    # img_cond_1[np.roll(condition2, 1, axis=1)] = np.full(np.shape(img_cond_1[condition2]), [0, 0, 255])

    # cv.imwrite("out/condition1.png", img_cond_1)
    # cv.imwrite("out/thicker_diagonal.png", img)
    # END DEBUG
    return img

if __name__ == "__main__":
    # Load the image
    img = cv.imread(sys.argv[1]) # BGR

    # name
    name = os.path.basename(sys.argv[1])

    shall_show = sys.argv[2] == "-d" if len(sys.argv) > 2 else False

    # Load the config
    print("Loading the config...")
    json_name = re.sub(r"\.[^\.]*$", ".json", sys.argv[1])
    with open(json_name, "r") as f:
        config = json.load(f)
    
    print(config["networks_speed"])


    # Preprocess the image by making diagonal lines thicker
    # example:
    # 1 0 0                   becomes 1 1 0
    # 0 1 0                           0 1 1
    # 0 0 1                           0 0 1
    print("Preprocessing the image...")
    img = thickerDiagonalNumpy(img, name)
    # img = thickerDiagonal(img)

    # Find pixels to process
    print("caching pixels to process...")
    rows,cols,_ = img.shape
    pixels_to_process = []

    pixels_to_explore = img[~whitePixelsMask(img)]
    indexes_to_explore = np.argwhere(~whitePixelsMask(img))

    # NON OPTIMIZED VERSION
    # for i in range(rows):
    #     print("%d%%\r" % (i/rows*100), end="") # progress bar
    #     for j in range(cols):
    #         k = img[i,j]
    #         if not is_white(k) and not is_gray(k):
    #             pixels_to_process.append((i,j))

    i = 0
    total = len(pixels_to_explore)
    for p in pixels_to_explore:
        print("%d%%\r" % (i/total*100), end="") # progress percentage
        if not is_gray(p):
            pixels_to_process.append(tuple(indexes_to_explore[i]))
        i += 1

    # Build the graph
    graph = nx.Graph()
    node_to_coord = {}
    coord_to_node = np.ones((rows, cols), dtype=int) * -1
    count = 0

    def add_node(pixel, node_value):
        graph.add_node(node_value)
        graph.nodes[node_value]["is_black"] = is_black(img[pixel[0], pixel[1]]) # black pass-through must not be deleted, so we need to keep the info
        graph.nodes[node_value]["x"] = pixel[1]
        graph.nodes[node_value]["y"] = pixel[0]
        node_to_coord[node_value] = pixel
        coord_to_node[pixel[0], pixel[1]] = node_value

    # link all neighbour pixels
    print("create base graph...")
    total = len(pixels_to_process)
    while (len(pixels_to_process) > 0):
        print(f"{len(pixels_to_process)}/{total} pixels left to process ({(len(pixels_to_process)/total * 100):.2f}%)", end="\r")

        # process a batch of connected pixels
        pixels_to_analyze = [pixels_to_process[0]] # better structure ?
        add_node(pixels_to_process[0], count)
        count += 1

        cnt = 0
        while (len(pixels_to_analyze) > 0):

            if (cnt % 1000 == 0):
                print(f"\t\t\t\t\t\t\tcache: {len(pixels_to_analyze)}_____________\t computed in current iteration: {cnt}______________", end="\r")
            cnt += 1

            pixel = pixels_to_analyze[0]
            pixels_to_analyze.remove(pixel)
            
            # do not process twice
            if (not has_tuple(pixel, pixels_to_process)):
                continue
            pixels_to_process.remove(pixel)

            # check neighbours
            sides = [((pixel[0], pixel[1] - 1), 'd'), ((pixel[0] - 1, pixel[1]), 'l'), ((pixel[0], pixel[1] + 1), 'u'), ((pixel[0] + 1, pixel[1]), 'r')]
            for side in sides:
                p = side[0]
                if (exists(p, rows, cols) and not is_white(img[p[0], p[1]])):
                    final_p = p
                    distance = 1

                    # Should pass through ?
                    if (is_gray(img[p[0], p[1]])):
                        direction = side[1]

                        i = 1 # from gray just found
                        # direction
                        add = -1 if direction == 'd' else 1
                        # disable the other axis
                        ver = 1 if direction == 'd' or direction == 'u' else 0
                        hor = 1 if direction == 'l' or direction == 'r' else 0
                        # skip through all gray pixels in the same direction
                        while (exists((pixel[0] + i*add*hor, pixel[1] + i*add*ver), rows, cols) and is_gray(img[pixel[0] + i*add*hor, pixel[1] + i*add*ver])):
                            i += 1
                            distance += 1

                        final_p = (pixel[0] + i*add*hor, pixel[1] + i*add*ver)

                    # should connect ?
                    if (exists(final_p, rows, cols) and not is_gray(img[final_p[0], final_p[1]])):
                        if (is_black(img[final_p[0], final_p[1]]) or same_color(img[final_p[0], final_p[1]], img[pixel[0], pixel[1]])):
                            # Yes
                            if (coord_to_node[final_p[0], final_p[1]] == -1): # not yet in the graph
                                add_node((final_p[0], final_p[1]), count)
                                count += 1
                            
                            p_node = coord_to_node[final_p[0], final_p[1]]
                            u, v = int(coord_to_node[pixel[0], pixel[1]]), int(p_node)
                            graph.add_edge(u, v, weight=distance)
                            graph.edges[u, v]["speed"] = get_color_speed_from_bgr(img[pixel[0], pixel[1]], config)
                            graph.edges[u, v]["color"] = rgb_to_hex(bgr_to_rgb(img[pixel[0], pixel[1]]))
                            pixels_to_analyze.append(final_p) # IDENTICAL_A_END
    print("\nGraph created")                


    # replace pass-through nodes with edges
    print("simplify graph part 1...")
    keep_exploring = True
    while (keep_exploring):
        keep_exploring = False
        nodes = list(graph.nodes)
        for i in range(len(nodes)-1, -1, -1):
            node = nodes[i]
            neighbours = list(graph.neighbors(node))
            if (len(neighbours) == 2 and not graph.nodes[node].get("is_black", False)):
                n1, n2 = neighbours
                n1, n2 = int(n1), int(n2)
                graph.add_edge(n1, n2, weight=graph.edges[n1, node]["weight"] + graph.edges[node, n2]["weight"])
                graph.edges[n1, n2]["speed"] = graph.edges[n1, node]["speed"]
                graph.edges[n1, n2]["color"] = graph.edges[n1, node]["color"]
                graph.remove_node(node)
                keep_exploring = True # May have created new pass-through nodes
    
    # merge nodes with a distance of MAX
    print("simplify graph part 2...")
    MAX = 2 # min 2 to handle squares and bridges/tunnels
    keep_exploring = True
    while (keep_exploring):
        keep_exploring = False
        edges = list(graph.edges)
        for i in range(len(edges)-1, -1, -1):
            u, v = edges[i]
            if (graph.has_edge(u, v) and graph.edges[u, v]["weight"] <= MAX):
                # merge v into u
                for n in graph.neighbors(v):
                    if (n != u):
                        weight = graph.edges[v, n]["weight"]
                        u, n = int(u), int(n)
                        graph.add_edge(u, n, weight= weight if weight <= MAX else weight + 1) #prevents falsification of existing mergeable edges into false bigger values
                        graph.edges[u, n]["speed"] = graph.edges[v, n]["speed"]
                        graph.edges[u, n]["color"] = graph.edges[v, n]["color"]

                graph.remove_node(v)
                keep_exploring = True # May still have nodes to merge
            

    # Draw the graphs
    print("Exporting...")
    fig, axes = plt.subplots()
    positions = {}
    for node in graph.nodes:
        coord = node_to_coord[node]
        x = coord[1]
        y = coord[0]
        positions[node] = np.array([x, y])

    plt.imshow(img[...,::-1]) # BGR to RGB
    
    nx.draw(graph, positions, alpha=0.5, node_size=20)
    nx.draw_networkx_edge_labels(graph, positions, edge_labels={(u, v): d["weight"] for u, v, d in graph.edges(data=True)}, font_size=8)

    if shall_show: plt.show()



    # save
    setup_out_folder()

    # save the preview
    fig.savefig(f"out/output_of_{name}.png")

    # Save the graph
    with open(f"out/dist_graph_of_{name}.json", "w") as f:
        graph_data = nx.node_link_data(graph)
        # print(graph_data)
        json.dump(graph_data, f, default=make_serializable)

    # Save the speed graph and image
    with open(f"out/speed_graph_of_{name}.json", "w") as f:
        speed_graph = graph.copy()
        for u, v, d in speed_graph.edges(data=True):
            # print(u, v, d)
            d["weight"] = d["weight"] / d["speed"] # speed = distance / time => time = distance / speed
        graph_data = nx.node_link_data(speed_graph)
        # print(graph_data)
        json.dump(graph_data, f, default=make_serializable)

        # Save the image for speed
        fig, axes = plt.subplots()

        plt.imshow(img[...,::-1]) # BGR to RGB
    
        nx.draw(speed_graph, positions, alpha=0.5, node_size=20)
        nx.draw_networkx_edge_labels(speed_graph, positions, edge_labels={(u, v): d["weight"] for u, v, d in speed_graph.edges(data=True)}, font_size=8)

        if shall_show: plt.show()

        fig.savefig(f"out/speed_output_of_{name}.png")



    # # Show the image
    # cv.imshow("Graph", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
