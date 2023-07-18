import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from convcolors import rgb_to_lab
from extcolors import difference
from colormap import rgb2hex


def extract_colors_from_path(path: str, tolerance=12):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480),
                     interpolation=cv2.INTER_AREA)
    pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = pixels.reshape(-1, pixels.shape[-1])
    colors = _count_colors(pixels)
    colors = _compress(colors, tolerance)
    return colors


def _count_colors(pixels):
    unique_elements, element_counts = np.unique(pixels, axis=0, return_counts=True)
    return unique_elements, element_counts

 
def _compress(colors, tolerance):
    elements_rgb, counts = colors

    if tolerance <= 0:
        return elements_rgb

    elements_lab = np.array([rgb_to_lab(rgb) for rgb in elements_rgb])
    compressed = np.zeros(len(elements_rgb), dtype=bool)
    iterator = list(map(list, zip(elements_rgb, elements_lab, counts, compressed)))
    sorted_colors = sorted(iterator, reverse=True, key=lambda x: x[2])

    i = 0
    while i < len(sorted_colors):
        larger = sorted_colors[i][1]
        larger_compressed = sorted_colors[i][3]
        
        if not larger_compressed:
            j = i + 1 

            while j < len(sorted_colors):
                smaller = sorted_colors[j][1]
                smaller_compressed = sorted_colors[j][3]

                if not smaller_compressed and difference.cie76(larger, smaller) <= tolerance:
                    sorted_colors[i][2] += sorted_colors[j][2]
                    sorted_colors[j][3] = True
                
                j += 1
        i += 1

    colors = [color for color in sorted(sorted_colors, reverse=True,
                                        key=lambda x: x[2]) if not color[3]]
    return colors
    

def _print_image(img):
    data_frame = pd.DataFrame(img,
                              columns=['color_code', 'lab', 'occurence', 'compressed'])

    list_color = list(data_frame['color_code'])
    list_color = [rgb2hex(int(i[0]), int(i[1]), int(i[2])) for i in list_color]

    list_precent = list(data_frame['occurence'])
    list_precent = [int(i) for i in list(data_frame['occurence'])]
    text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%'
            for c, p in zip(list_color, list_precent)]

    fig, ax = plt.subplots(figsize=(90,90) ,dpi=10)
    wedges, text = ax.pie(list_precent, labels=text_c, labeldistance=1.05,
                        colors=list_color, textprops={'fontsize': 120, 'color':'black'})

    plt.setp(wedges, width=0.3)
    plt.setp(wedges, width=0.36)
    ax.set_aspect("equal")
    fig.set_facecolor("yellow")
    plt.show()


img = extract_colors_from_path("page_images/donkey.png", tolerance=15)
_print_image(img)
