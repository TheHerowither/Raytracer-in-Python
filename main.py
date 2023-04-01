import numpy as np
from threading import Thread
import os, time, json, random, ffmpeg, cv2, glob
import matplotlib.pyplot as plt
from tkinter import messagebox as mb
from PIL import Image, ImageTk, ImageFilter
from numba import jit, cuda


def normalize(vector):
    return vector / np.linalg.norm(vector)
def Average(lst):
    #print(lst)
    return sum(lst) / len(lst)
def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    
    #print(delta)
    try:
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
    except ValueError:
        return 0
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def ray_scatter(vector, hit_point, scatter):
    rays = []
    rays.append(reflected(vector, hit_point))

    for point in range(scatter):
        positive = bool(random.getrandbits(1))
        if positive:
            rays.append(reflected(vector+(point/2), hit_point))
        else:
            rays.append(reflected(vector-(point/2), hit_point))
    
    return rays

width = 1080
height = 720
preview_width = 50
preview_height = 50
preview_aspect_ratio = preview_width/preview_height

max_depth = 3





camera = np.array([0, 0, 5])


light = { 'center': np.array([5, 5, 5]), "radius": .1, 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': .55, 'ambient': np.array([0, 0, 0]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 1 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.2 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.2 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]


object_names = [
    '{"Sphere":1}',
    '{"Sphere":2}',
    '{"Sphere":3}',
    '{"Sphere":4}'
    ]



import tkinter as tk
from tkinter import ttk

app = tk.Tk()
app.geometry("1080x820")
app.resizable(False, False)
app.title("Raytracer in Python; Made by The_Herowither")

variable = tk.StringVar(app)
variable.set(object_names[0]) # default value

w = tk.OptionMenu(app, variable, *object_names)

render_info = tk.Label(app, text = "")
rendered_image = tk.Label(image = None)
preview = tk.Label(image = None)

light_rays = 2
#@jit
frames = 1

def render(imname = "image", vid = None, frame = 0, **kwargs):
    aspect_ratio = width/height
    #preview_width, preview_height = round(width/10), round(height/10)
    if imname != "tmp":
        screen = (-1, 1 / aspect_ratio, 1, -1 / aspect_ratio)
    else:
        screen = (-1, 1 / preview_aspect_ratio, 1, -1 / preview_aspect_ratio)
    im_start_time = time.time()
    image = np.zeros((height, width, 3))
    w, h = width, height
    if imname == "tmp":
        w, h = preview_width, preview_height
    for i, y in enumerate(np.linspace(screen[1], screen[3], h)):
        start_time = time.time()
        pixel_render_times = []
        for j, x in enumerate(np.linspace(screen[0], screen[2], w)):
            pixel_start_time = time.time()
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)

            color = np.zeros((3))
            reflection = 1
            is_shadowed = False

            for k in range(max_depth):
                # check for intersections
                nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
                if nearest_object is None:
                    
                    break

                intersection = origin + min_distance * direction
                normal_to_surface = normalize(intersection - nearest_object['center'])
                shifted_point = intersection + 1e-5 * normal_to_surface
                intersection_to_light = normalize(light['center'] - shifted_point)


                _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                intersection_to_light_distance = np.linalg.norm(light['center'] - intersection)
                is_shadowed = min_distance < intersection_to_light_distance
                #light_rays = ray_scatter(direction, intersection, 10)
                #for ray in light_rays:
                    
                #print(ray_scatter(direction, intersection, 10))

                if is_shadowed:
                    #color = (.5,.5,1)
                    break

                illumination = np.zeros((3))

                # ambiant
                illumination += nearest_object['ambient'] * light['ambient']

                # diffuse
                illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)

                # specular
                intersection_to_camera = normalize(camera - intersection)
                H = normalize(intersection_to_light + intersection_to_camera)
                illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

                # reflection
                color += reflection * illumination
                reflection *= nearest_object['reflection']
                #if imname != "tmp":
                #    print(color.tolist())
                
                origin = shifted_point
                direction = reflected(direction, normal_to_surface)

            #if imname != "tmp":
            #    print(np.round(color, 1).tolist())
            #if (np.around(color, 1).tolist() == [0.0,0.0,0.0] and not is_shadowed):
            #    color = (.9,.9,1)
            image[i, j] = np.clip(color, 0, 1)
            pixel_render_times.append(time.time()-pixel_start_time)
            
        plt.imsave(f'{imname}.png', image)
        im = Image.open(f'{imname}.png')
        
        if imname != "tmp":
            im = im.resize((round(400*aspect_ratio),400), Image.Resampling.NEAREST)
            #im = im.filter(ImageFilter.BoxBlur(((width+height)/4)/100))
            im = ImageTk.PhotoImage(im)
            rendered_image.configure(image = im)
            rendered_image.image = im
        else:
            im = im.resize((round(100*preview_aspect_ratio),100), Image.Resampling.NEAREST)
            im = ImageTk.PhotoImage(im)
            preview.configure(image = im)
            preview.image = im
        #os.system("cls")
        rtime = time.time() - start_time
        #print("Rendering image\n",
        #      "#################",
        #      f"Width: {width}",
        #      f"Height: {height}",
        #      f"Average pixel render time: {round(Average(pixel_render_times), 5)}",
        #      "#################\n",
        #      "Rendered columns:", sep = "\n")
        #print("%d/%d" % (i + 1, height))
        #print("Column render time:", round(rtime, 5), "seconds", end = "\n")
        #print("Estimated time remaining:", round(rtime*(height-(i+1)), 2))
        if imname != "tmp":
            render_info.config(text = f"Rendering image\n\nRendered columns:\n{i + 1}/{height}\n\nWidth: {width}\nHeight: {height}\n\nAverage render time per pixel: {round(Average(pixel_render_times), 5)}\nLast column render time: {round(rtime, 5)} seconds\nEstimated time remaining: {round(rtime*(height-(i+1)), 2)}\n\nTime spent: {round(time.time()-im_start_time,5)}")

    plt.imsave(f'{imname}.png', image)
    
    
    if (imname != "tmp" and vid != None):
        render_info.config(text = f"Rendering image\n\nRendered columns:\n{i + 1}/{height}\n\nWidth: {width}\nHeight: {height}\n\nAverage render time per pixel: {round(Average(pixel_render_times), 5)}\nLast column render time: {round(rtime, 5)} seconds\nEstimated time remaining: {round(rtime*(height-(i+1)), 2)}\n\nTotal time spent: {round(time.time()-im_start_time, 5)} seconds")
        #mb.showinfo("Render complete", f"The render was completed in {round(time.time()-im_start_time, 5)} seconds")
    if vid != None:
        #frame = cv2.imread(".\\image.png")
        #vid.write(frame)
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        os.system("copy image.png tmp")
        os.system(f"rename tmp\\image.png '{frame}.png'")
        print("Appended frame to video")
        
    
    
    

#current_frame = 0




def start_render():
    global width, height
    vid = cv2.VideoWriter('output_video.jpg',cv2.VideoWriter_fourcc(*'DIVX'), frames/2, (width, height))
    for i in range(frames):
        print(f"Started rendering frame: {i}")
        render_thread = Thread(target = render, args = ["image", vid, i])
        width = round(int(width_input.get("1.0", "end-1c")))
        height = round(int(height_input.get("1.0", "end-1c")))

        camera[0] = i+1
        camera[2] = frames+1
        
        render_thread.start()
        render_thread.join()

    for filename in glob.glob(f'{os.getcwd}/tmp/*.png'):
        img = cv2.imread(filename)
        vid.write(img)
    #cv2.destroyAllWindows()
    vid.release()
    print("Video released")
th = Thread(target = start_render)
def preview_refresh():
    global camera
    render_thread2 = Thread(target = render, args = ["tmp"])
    try:
        tmpcam = camera_input.get("1.0", "end-1c").split(",")
        cam = []
        for i in tmpcam:
            cam.append(int(i))
        camera = np.array(cam)
    except ValueError:
        pass
    width = 20
    height = 20

    render_thread2.start()

obj_settings = []
def edit_objects():
    #print("Edting object number:", json.loads(variable.get())["Sphere"])
    index = json.loads(variable.get())["Sphere"]-1
    string = []
    obj_settings = []
    for i in objects[index].keys():
        string.append(i+": ")#+str(objects[index][i]))
        obj_settings.append(str(objects[index][i]))
    object_info.config(text = "\n".join(string))
    #print("\n".join(string))


edit_objects_btn = tk.Button(app, text = "Edit object", command = edit_objects)
start_render_btn = tk.Button(app, text = "Start rendering", command = th.start)

width_input = tk.Text(app, width = 10, height = 1)
height_input = tk.Text(app, width = 10, height = 1)

camera_input = tk.Text(app, width = 10, height = 1)
camera_text = tk.Label(text = "Camera:")
cam = []
for i in camera.tolist():
    cam.append(str(i))
cam = ",".join(cam)
camera_input.insert(tk.END, f"{cam}")
submit_btn = tk.Button(app, text = "Update", command = preview_refresh)

width_lbl = tk.Label(text = "Width:")
height_lbl = tk.Label(text = "Height:")

object_info = tk.Label(text = None, justify = "left")

width_lbl.place(anchor = tk.CENTER, relx = .45, rely = .02)
width_input.place(anchor = tk.CENTER, relx = .45, rely = .05)
height_lbl.place(anchor = tk.CENTER, relx = .55, rely = .02)
height_input.place(anchor = tk.CENTER, relx = .55, rely = .05)

camera_text.pack(side = "left")
camera_input.pack(side = "left")
submit_btn.pack(side = "left")


start_render_btn.place(anchor = tk.CENTER, relx = .5, rely = .1)
render_info.place(anchor = tk.CENTER, relx = .5, rely = .25)
rendered_image.place(anchor = tk.CENTER, relx = .5, rely = .7)
preview.pack(side = "right")
w.place(anchor = tk.SE, relx = .1, rely = .04)
edit_objects_btn.place(anchor = tk.SE, relx = .065, rely = .07)
object_info.place(anchor = tk.SE, relx = .056, rely = .21)

preview_refresh()
app.mainloop()

