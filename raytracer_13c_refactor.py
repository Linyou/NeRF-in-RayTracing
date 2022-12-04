import numpy as np
from numpy.core.arrayprint import format_float_scientific
import random
import taichi as ti

ti.init(arch=ti.cuda)  # Try to run on GPU

infinity = np.Inf
pi = np.pi


@ti.pyfunc
def field1D_to_vector(v):
    u = v[None]
    return ti.Vector([u[0], u[1], u[2]])

@ti.pyfunc
def angle_cosine(a, b):
    return a.dot(b)/(a.norm() * b.norm())


@ti.func
def clamp(x, min, max):
    if x < min:
        x = min
    if x > max:
        x = max
    return x


@ti.pyfunc
def degrees_to_radians(d):
    return d * pi / 180.0


@ti.pyfunc
def radians_to_degrees(r):
    return r * 180.0 / pi


@ti.func
def near_zero(v):
    s = 1e-8
    v_abs = ti.abs(v)
    return (v_abs[0] < s) and (v_abs[1] < s) and (v_abs[2] < s)

@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = ti.min((-uv).dot(n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta*n)
    r_out_parallel = -ti.sqrt(ti.abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def write_color(out, i, j, pixel_color, samples_per_pixel, cnt):

    radience[i, j] += pixel_color

    out[i, j] = ti.sqrt(radience[i, j] / samples_per_pixel / ti.cast(cnt, float)) 


@ti.func
def get_random_vector3_length():

    p = ti.Vector([
        ti.random(ti.f32), 
        ti.random(ti.f32), 
        ti.random(ti.f32)
    ]) * 2.0 - ti.Vector([1.0,1.0,1.0])
    length_squared = p.dot(p)
    
    return p, length_squared

@ti.func
def get_random_vector2_length():

    p = ti.Vector([
        ti.random(ti.f32), 
        ti.random(ti.f32)
    ]) * 2.0 - ti.Vector([1.0,1.0])
    length_squared = p.dot(p)
    
    return p, length_squared


# @ti.func
# def random_in_unit_disk():

#     p, length_squared = get_random_vector2_length()

#     while length_squared >= 1.0:
#         p, length_squared = get_random_vector2_length()

#     return p

@ti.func
def random_in_unit_disk():
    theta = 2.0 * pi * ti.random()
    return ti.Vector([ti.cos(theta), ti.sin(theta)])

# @ti.func
# def random_in_unit_sphere():

#     p, length_squared = get_random_vector3_length()

#     while length_squared >= 1.0:
#         p, length_squared = get_random_vector3_length()

#     return p

@ti.func
def random_in_unit_sphere(): # Here is the optimization
    theta = 2.0 * pi * ti.random()
    phi = ti.acos((2.0 * ti.random()) - 1.0)
    r = ti.pow(ti.random(), 1.0/3.0)
    return ti.Vector([r * ti.sin(phi) * ti.cos(theta), r * ti.sin(phi) * ti.sin(theta), r * ti.cos(phi)])


@ti.func
def random_in_hemisphere(normal):

    in_unit_sphere = random_in_unit_sphere()
    out = -in_unit_sphere
    if in_unit_sphere.dot(normal) > 0.0:
        out = in_unit_sphere

    return out


@ti.data_oriented
class materials:
    def __init__(self, s, a, e):
        self.kind = s
        self.albedo = a
        self.extra = e #Fuzz/Refraction_Rate

@ti.func
def scatter(r: ti.template(), hit_r: ti.template(), attenuation: ti.template()):

    kind = hit_r.material_kind
    albedo = hit_r.material_albedo
    extra = hit_r.material_extra

    did_scatter = False
    if kind == 0: # lambertian

        random_unit = hit_r.normal + random_in_unit_sphere()

        if near_zero(random_unit):
            random_unit = hit_r.normal 

        r.origin = hit_r.p
        r.direction = random_unit

        attenuation = albedo

        did_scatter =  True

    elif kind == 1: # metal

        reflected = reflect(r.direction.normalized(), hit_r.normal)

        r.origin = hit_r.p
        r.direction = reflected + extra*random_in_unit_sphere()

        attenuation = albedo

        did_scatter = reflected.dot(hit_r.normal) > 0

    elif kind == 2: # dielectric

        attenuation = ti.Vector([1.0, 1.0, 1.0])
        
        refraction_ratio = extra
        if hit_r.front_face == 1.0:
            refraction_ratio = 1.0/extra

        unit_direction = r.direction.normalized()
        cos_theta = ti.min((-unit_direction).dot(hit_r.normal), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta*cos_theta)

        cannot_refract = refraction_ratio * sin_theta > 1.0

        direction = ti.Vector([1.0, 1.0, 1.0])
        did_reflectance = reflectance(cos_theta, refraction_ratio) > ti.random(ti.f32)
        # did_reflectance = False
        if cannot_refract or did_reflectance:
            direction = reflect(unit_direction, hit_r.normal)
        else:
            direction = refract(unit_direction, hit_r.normal, refraction_ratio)

        r.origin = hit_r.p
        r.direction = direction

        did_scatter =  True
    
    else:
        r.origin = hit_r.p
        r.direction = hit_r.normal
        attenuation = albedo
        did_scatter = True

    return did_scatter

@ti.func
def reflectance(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0_new = r0 * r0
    return r0_new + (1-r0_new)*ti.pow((1- cosine), 5)


@ti.func
def new_record():

    float1 = 0.5
    float2 = 0.5
    float3 = 0.5
    float4 = 0.5
    vector1 = ti.Vector([1.0, 1.0, 1.0])
    vector2 = ti.Vector([1.0, 1.0, 1.0])
    vector3 = ti.Vector([1.0, 1.0, 1.0])
    record = HitRecord_(
        float1, float2, float3, float4, 
        vector1, vector2, vector3
    )

    return record


# Hit record
@ti.data_oriented
class HitRecord_:
    def __init__(self, float1, float2, float3, float4, vector1, vector2, vector3):
        self.p = vector1
        self.normal = vector2
        self.t = float1
        self.front_face = float2
        self.material_kind = float3
        self.material_albedo = vector3
        self.material_extra = float4

    @ti.func
    def set_face_normal(self, r, outward_normal):
        if r.direction.dot(outward_normal) < 0:
            self.front_face = 1.0
        else:
            self.front_face = -1.0

        if self.front_face > 0:
            self.normal = outward_normal
        else:
            self.normal = -outward_normal

    

# Ray
@ti.data_oriented
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + self.direction * t


# Hittable object
# Shpere
@ti.data_oriented
class Shpere:
    def __init__(self, descriptor):
        vec3f = ti.types.vector(3, ti.f32)
        shpere_field = ti.types.struct(
            center=vec3f, 
            raduis=ti.f32, 
            material_kind=ti.i32, 
            material_albedo=vec3f, 
            material_extra=ti.f32
        )

        self.size = len(descriptor)
        self.content = shpere_field.field(shape=(self.size,))

        for i, obj in enumerate(descriptor):
            self.content[i].center = ti.Vector(obj["center"])
            self.content[i].raduis = obj["raduis"]
            self.content[i].material_kind = obj["material"].kind
            self.content[i].material_albedo = obj["material"].albedo
            self.content[i].material_extra = obj["material"].extra
    
    @ti.func
    def hit(self, index, r: ti.template(), hit_r: ti.template(), t_min, t_max):
        # assume the object is hit by ray
        hitting = True

        ray_origin = r.origin
        ray_direction = r.direction 

        oc = ray_origin - self.content[index].center
        a = ray_direction.dot(ray_direction)
        half_b = oc.dot(ray_direction)
        c = oc.dot(oc) - self.content[index].raduis*self.content[index].raduis
        discriminant = half_b*half_b - a*c
        sqrtd = ti.sqrt(discriminant)

        if discriminant < 0:
            hitting = False

        # Find the nearest root that lies in the acceptable range.
        root = (-half_b - sqrtd) / a
        if (root < t_min or t_max < root):
            root = (-half_b + sqrtd) / a
            if (root < t_min or t_max < root):
                hitting = False
        
        if hitting == True:
            hit_r.t = root
            hit_r.p = r.at(root)
            outward_normal = (hit_r.p - self.content[index].center) / self.content[index].raduis
            hit_r.set_face_normal(r, outward_normal)
            hit_r.material_kind = self.content[index].material_kind
            hit_r.material_albedo = self.content[index].material_albedo
            hit_r.material_extra = self.content[index].material_extra

        return hitting

@ti.data_oriented
class ShapeIndexer:
    def __init__(self, shape_descriptors):
        
        size_of_each = [len(x) for x in shape_descriptors]
        total_size = sum(size_of_each)

        shape = (total_size, 2)
        shape_index = np.zeros(shape, dtype=np.int32)

        shape_len_arr = [sum(size_of_each[:i+1]) for i in range(len(size_of_each))]

        cat_index = 0
        each_index = 0
        for i in range(total_size):
            
            for j in range(len(shape_len_arr)):
                if i == shape_len_arr[j]:
                    each_index = 0
                    cat_index += 1

            shape_index[i, 0] = cat_index
            shape_index[i, 1] = each_index

            
            each_index+=1

        print(shape_index)

        self.shape_index = ti.Vector.field(2, ti.i32, shape=(total_size, ))
        self.shape_index.from_numpy(shape_index)
        print(self.shape_index)


# Hittable list
@ti.data_oriented
class HittableList:
    def __init__(self):
        self.shape = []

    @ti.func
    def hit(self, r, hit_r, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        # for shpere
        # for index in ti.static(range(shpere.size)):
        for index in range(shpere.size):
            if shpere.hit(index, r, hit_r, t_min, closest_so_far):
                hit_anything = True
                closest_so_far = hit_r.t

        return hit_anything

    def add(self, descriptor):
        for obj in descriptor:
            self.shape.append(obj)

    def clear(self):
        self.shpere_num = 0


# Camera
class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist, image_width):

        theta = np.deg2rad(vfov)
        h = np.tan(theta/2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        w = (lookfrom - lookat).normalized()
        u = vup.cross(w).normalized()
        v = w.cross(u)

        image_height = int(image_width / aspect_ratio) 


        self.viewport_height = ti.field(float, shape = ())
        self.viewport_width = ti.field(float, shape = ())
        self.horizontal = ti.Vector.field(3, float, shape = ())
        self.vertical = ti.Vector.field(3, float, shape = ())
        self.origin = ti.Vector.field(3, float, shape = ())
        self.lower_left_corner = ti.Vector.field(3, float, shape = ())
        self.focal_length = ti.field(float, shape = ())
        self.lens_radius = ti.field(float, shape = ())
        self.focus_dist = ti.field(float, shape = ())

        self.w = ti.Vector.field(3, float, shape = ())
        self.u = ti.Vector.field(3, float, shape = ())
        self.v = ti.Vector.field(3, float, shape = ())

        self.lookfrom = lookfrom
        self.lookat = lookat
        self.aspect_ratio = aspect_ratio
        self.vup = vup

        self.w[None] = w
        self.u[None] = u
        self.v[None] = v

        self.focus_dist[None] = focus_dist

        origin = lookfrom

        horizontal = ti.Vector([u[0], u[1], u[2]]) * viewport_width * focus_dist
        vertical = ti.Vector([v[0], v[1], v[2]]) * viewport_height * focus_dist  

        self.origin[None] = origin
        self.horizontal[None] = horizontal
        self.vertical[None] = vertical

        self.lower_left_corner[None] = origin - horizontal/2 - vertical/2 - w * focus_dist

        self.lens_radius[None] = aperture / 2

        self.res = (image_width, image_height)
        self.samples_per_pixel = 1

    def update_camera(self, local_movement, oriented_movement, vfov, focus_dist, look_at_change):

        w = field1D_to_vector(self.w)
        u = field1D_to_vector(self.u)
        v = field1D_to_vector(self.v)

        lookfrom_vector = self.lookfrom - local_movement[1] * w + local_movement[0] * u + oriented_movement * u
        lookat_vector = self.lookat + local_movement[0] * u +look_at_change

        theta = degrees_to_radians(vfov)
        h = ti.tan(theta/2)
        viewport_height = 2.0 * h
        viewport_width = self.aspect_ratio * viewport_height

        w = (lookfrom_vector - lookat_vector).normalized()
        u = self.vup.cross(w).normalized()
        v = w.cross(u)

        self.w[None] = w
        self.u[None] = u
        self.v[None] = v

        self.focus_dist[None] = focus_dist

        origin = lookfrom_vector


        horizontal = ti.Vector([u[0], u[1], u[2]]) * viewport_width * self.focus_dist[None]
        vertical = ti.Vector([v[0], v[1], v[2]]) * viewport_height * self.focus_dist[None]
        lower_left_corner = origin - horizontal/2 - vertical/2 - w * self.focus_dist[None]

        return origin, horizontal, vertical, lower_left_corner

    @ti.func
    def set_ray(self, s, t):

        lens_radius, u, v, _origin, lower_left_corner, horizontal, vertical = ti.static(
            self.lens_radius, self.u, self.v, self.origin, 
            self.lower_left_corner, self.horizontal, self.vertical)

        rd = lens_radius[None] * random_in_unit_disk()
        offset = u[None] *  rd[0] + v[None] * rd[1]

        origin = _origin[None] + offset
        direction = lower_left_corner[None] + s * horizontal[None] + t * vertical[None] - _origin[None] - offset

        return Ray(origin, direction)



max_depth = ti.field(int, shape = ())
max_depth[None] = 10
vfov = 20.0
R = np.cos(pi/4)
aspect_ratio = 16.0 / 9.0
image_width = 1280
lookfrom_list = [13.0,2.0,3.0]
lookat_list = [0.0,0.0,0.0]
aperture = 0.1
dist_to_focus = 10.0


pause = False
local_movement = [0.0, 0.0, 0.0]
new_local_movement = [0.0, 0.0, 0.0]
look_at_change = [0.0, 0.0, 0.0]
new_look_at_change = [0.0, 0.0, 0.0]
movement_speed = 0.0
oriented_movement = 0.0
new_oriented_movement = 0.0
cnt = 0
changing_factor = False

camera = Camera(
    ti.Vector(lookfrom_list), 
    ti.Vector(lookat_list), 
    ti.Vector([0.0,1.0,0.0]), 
    vfov, 
    aspect_ratio,
    aperture, 
    dist_to_focus,
    image_width
)


# material
m_lambertian = 0
m_metal = 1
m_dielectric = 2


world = HittableList()

shpere_descriptor = []
ground_material = materials(m_lambertian, ti.Vector([0.5, 0.5, 0.5]), 0.0)
shpere_descriptor.append({"shape_name": "shpere", "center": [ 0.0, -1000.0, -1.0], "raduis":  1000, "material": ground_material})

for a in range(-5, 5):
    for b in range(-5, 5):
        choose_mat = random.random()
        a + 0.9*random.random(), 0.2, b + 0.9*random.random()
        center = [a + 0.9*random.random(), 0.2, b + 0.9*random.random()]
        temp_center_sub = ti.Vector(center) - ti.Vector([4, 0.2, 0])

        if ti.sqrt(temp_center_sub.dot(temp_center_sub)) > 0.9:
            if choose_mat < 0.6:
                albedo = ti.Vector([random.random(), random.random(), random.random()]) * ti.Vector([random.random(), random.random(), random.random()])
                temp_material = materials(m_lambertian, albedo, 0.0)
                shpere_descriptor.append({"shape_name": "shpere", "center": center, "raduis":  0.2, "material": temp_material})
            elif choose_mat < 0.95:
                albedo = ti.Vector([random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1)]) 
                fuzz = random.uniform(0, 0.5)
                temp_material = materials(m_metal, albedo, fuzz)
                shpere_descriptor.append({"shape_name": "shpere", "center": center, "raduis":  0.2, "material": temp_material})
            else:
                temp_material = materials(m_dielectric, ti.Vector([0.0,0.0,0.0]), 1.5)
                shpere_descriptor.append({"shape_name": "shpere", "center": center, "raduis":  0.2, "material": temp_material})


material1 = materials(m_dielectric, ti.Vector([0.0,0.0,0.0]), 1.5)
shpere_descriptor.append({"shape_name": "shpere", "center": [0.0, 1.0, 0.0], "raduis":  1.0, "material": material1})

material2 = materials(m_lambertian, ti.Vector([0.4, 0.2, 0.1]), 0.0)
material_light = materials(3, ti.Vector([10.0, 10.0, 10.0]), 1.0)
shpere_descriptor.append({"shape_name": "shpere", "center": [-4.0, 1.0, 0.0], "raduis":  1.0, "material": material2})

material3 = materials(m_metal, ti.Vector([0.7, 0.6, 0.5]), 0.0)
shpere_descriptor.append({"shape_name": "shpere", "center": [4.0, 1.0, 0.0], "raduis":  1.0, "material": material3})

shpere = Shpere(shpere_descriptor)
world.add(shpere_descriptor)


image_width = camera.res[0]
image_height = camera.res[1]
samples_per_pixel = camera.samples_per_pixel

# Render
radience = ti.Vector.field(3, dtype = float, shape=camera.res)
final_pixels = ti.Vector.field(3, dtype=float, shape=camera.res)

@ti.kernel
def reset_camera(lookfrom_vector: ti.template(), horizontal: ti.template(), vertical: ti.template(), lower_left_corner: ti.template()):

    camera.origin[None] = lookfrom_vector
    camera.horizontal[None] = horizontal
    camera.vertical[None] = vertical
    camera.lower_left_corner[None] = lower_left_corner

cur_attenuation_temp = ti.Vector.field(3, shape=(image_width, image_height), dtype=ti.f32)
temp_direction = ti.Vector.field(3, shape=(image_width, image_height), dtype=ti.f32)

@ti.kernel
def render_trace(cnt : int):
    for i, j in ti.ndrange((0, image_width), (0, image_height)):

        # pixel_color = ti.Vector([0.0, 0.0, 0.0])

        u = (i + ti.random(ti.f32)) / (image_width - 1)
        v = (j + ti.random(ti.f32)) / (image_height-1)

        # ray
        r = camera.set_ray(u, v)

        # color = ti.Vector([0.0,0.0,0.0])
        cur_attenuation = ti.Vector([1.0, 1.0, 1.0])
        d = 0

        record = new_record()

        while d < max_depth[None]:
            d += 1
            hitting = world.hit(r, record, 0.001, infinity)
            
            if hitting:

                attenuation = ti.Vector([1.0, 1.0, 1.0])
                
                if scatter(r, record, attenuation):
                    cur_attenuation *= attenuation
                else:
                    cur_attenuation_temp[i, j] = ti.Vector([0.0,0.0,0.0])
                    # color = ti.Vector([0.0,0.0,0.0])
                    break
            else:
                cur_attenuation_temp[i, j] = cur_attenuation
                temp_direction[i, j] = r.direction.normalized()
                # unit_direction = r.direction.normalized()
                # t = 0.5*(unit_direction[1] + 1.0)
                # color = ((1.0-t)*ti.Vector([1.0, 1.0, 1.0]) + t*ti.Vector([0.5, 0.7, 1.0])) * cur_attenuation
                break

        if d == max_depth[None]:
            cur_attenuation_temp[i, j] = ti.Vector([0.0,0.0,0.0])
            color = ti.Vector([0.0,0.0,0.0])

    #     pixel_color += color

    # radience[i, j] += pixel_color
    # final_pixels[i, j] = ti.sqrt(radience[i, j] / samples_per_pixel / ti.cast(cnt, float)) 

@ti.kernel
def render_record(cnt : int):
    for i, j in ti.ndrange((0, image_width), (0, image_height)):

        unit_direction = temp_direction[i,j]
        cur_attenuation = cur_attenuation_temp[i,j]
        t = 0.5*(unit_direction[1] + 1.0)
        color = ((1.0-t)*ti.Vector([1.0, 1.0, 1.0]) + t*ti.Vector([0.5, 0.7, 1.0])) * cur_attenuation

        radience[i, j] += color
        final_pixels[i, j] = ti.sqrt(radience[i, j] / samples_per_pixel / ti.cast(cnt, float)) 


window = ti.ui.Window("Taichi RayTracer", camera.res, vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()
while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key in [ti.ui.ESCAPE]:
            exit()
    
    with gui.sub_window("Configuration", 0.05, 0.05, 0.3, 0.4) as w:
        new_vfov = w.slider_float("vfov", vfov, 0.0, 100.0)
        if abs(new_vfov - vfov) > 0:
            vfov = new_vfov
            changing_factor = True

        new_max_depth = int(w.slider_float("max_depth", max_depth[None], 0, 100))
        if abs(new_max_depth - max_depth[None]) > 0:
            max_depth[None] = new_max_depth
            changing_factor = True

        new_oriented_movement = w.slider_float("oriented_movement", oriented_movement, -10.0, 10.0)
        if abs(new_oriented_movement - oriented_movement) > movement_speed:
            oriented_movement = new_oriented_movement
            changing_factor = True

        new_dist_to_focus = w.slider_float("dist_to_focus", dist_to_focus, 0.0, 30.0)
        if abs(new_dist_to_focus - dist_to_focus) > movement_speed:
            dist_to_focus = new_dist_to_focus
            changing_factor = True

        w.text("local_movement:")

        new_local_movement[0] = w.slider_float("localx", local_movement[0], -10.0, 10.0)
        if abs(new_local_movement[0] - local_movement[0]) > movement_speed:
            local_movement[0] = new_local_movement[0] 
            changing_factor = True

        new_local_movement[1] = w.slider_float("localy", local_movement[1], -10.0, 10.0)
        if abs(new_local_movement[1] - local_movement[1]) > movement_speed:
            local_movement[1] = new_local_movement[1]
            changing_factor = True

        w.text("look_at_change:")
        look_at_change
        new_look_at_change[0] = w.slider_float("lookAtx", look_at_change[0], -10.0, 10.0)
        if abs(new_look_at_change[0] - look_at_change[0]) > movement_speed:
            look_at_change[0] = new_look_at_change[0] 
            changing_factor = True

        new_look_at_change[1] = w.slider_float("lookAty", look_at_change[1], -10.0, 10.0)
        if abs(new_look_at_change[1] - look_at_change[1]) > movement_speed:
            look_at_change[1] = new_look_at_change[1]
            changing_factor = True

        new_look_at_change[2] = w.slider_float("lookAtz", look_at_change[2], -10.0, 10.0)
        if abs(new_look_at_change[2] - look_at_change[2]) > movement_speed:
            look_at_change[2] = new_look_at_change[2]
            changing_factor = True

    if changing_factor:
        changing_factor = False
        cnt = 0
        radience.fill(0)
        origin, horizontal, vertical, lower_left_corner = camera.update_camera(
            ti.Vector(local_movement), oriented_movement, vfov, dist_to_focus, ti.Vector(look_at_change)
        )
        reset_camera(origin, horizontal, vertical, lower_left_corner)

    cnt += 1
    render_trace(cnt)
    render_record(cnt)
    # print(final_pixels)
    canvas.set_image(final_pixels)
    window.show()
