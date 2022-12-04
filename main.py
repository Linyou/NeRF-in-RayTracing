import taichi as ti
from vector import *
import ray
from time import time
from hittable import World, Sphere, Cube
from camera import Camera
from material import *
import math
import random
from taichi.math import vec3
import tqdm
import numpy as np
from taichi_ngp import NGP_fw, PRETRAINED_MODEL_URL
import os
import wget


# switch to cpu if needed
ti.init(arch=ti.cuda)


@ti.func
def get_background(dir):
    ''' Returns the background color for a given direction vector '''
    unit_direction = dir.normalized()
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * WHITE + t * BLUE


if __name__ == '__main__':
    # image data
    aspect_ratio = 3.0 / 2.0
    image_width = 1200
    image_height = int(image_width / aspect_ratio)
    rays = ray.Rays(image_width, image_height)
    pixels = ti.Vector.field(3, dtype=float)
    final_pixels = ti.Vector.field(3, dtype=float)
    attenuation_temp = ti.Vector.field(3, dtype=float)
    dir_temp = ti.Vector.field(3, dtype=float)
    sample_count = ti.field(dtype=ti.i32)
    needs_sample = ti.field(dtype=ti.i32)
    ti.root.dense(ti.ij,
                  (image_width, image_height)).place(
                    pixels, sample_count,
                    needs_sample, final_pixels,
                    attenuation_temp, dir_temp
                )

    samples_per_pixel = 512
    max_depth = 16

    # materials
    mat_ground = Lambert([0.5, 0.5, 0.5])
    mat2 = Lambert([0.4, 0.2, 0.2])
    mat1 = Dielectric(1.5)
    mat3 = Metal([0.7, 0.6, 0.5], 0.0)


    # ray cube info buffer
    cube_ray_org = ti.Vector.field(3, dtype=ti.f32, shape=(image_width*image_height*10))
    cube_ray_dir = ti.Vector.field(3, dtype=ti.f32, shape=(image_width*image_height*10))
    cube_ray_id = ti.Vector.field(2, dtype=ti.i32, shape=(image_width*image_height*10))
    # cube_ray_t1t2 = ti.Vector.field(2, dtype=ti.f32, shape=(image_width*image_height*10))



    # load density field
    grid_size = 128
    model_path = '../taichi-ngp-renderer/npy_models/lego.npy'
    print('Loading model from {}'.format(model_path))
    model = np.load(model_path, allow_pickle=True).item()
    print(model.keys())

    density_grid = ti.field(ti.uint8, shape=(grid_size**3//8))
    density_grid.from_numpy(model['model.density_bitfield'])

    init_grid = ti.field(ti.i32, shape=(128**3, ))

    @ti.func
    def __expand_bits(v):
        v = (v * ti.uint32(0x00010001)) & ti.uint32(0xFF0000FF)
        v = (v * ti.uint32(0x00000101)) & ti.uint32(0x0F00F00F)
        v = (v * ti.uint32(0x00000011)) & ti.uint32(0xC30C30C3)
        v = (v * ti.uint32(0x00000005)) & ti.uint32(0x49249249)
        return v

    @ti.func
    def __morton3D(xyz):
        xyz = __expand_bits(xyz)
        return xyz[0] | (xyz[1] << 1) | (xyz[2] << 2)

    @ti.kernel
    def init_m_transforms() -> ti.i32:
        bit_sum = 0
        ti.loop_config(serialize=True)
        for i,j,k in ti.ndrange(128, 128, 128):
            n = i*128*128 + k * 128 + j

            nxyz = vec3(i, j, k)
            idx = __morton3D(ti.cast(nxyz, ti.u32))
            occ = density_grid[ti.u32(idx//8)] & (1 << ti.u32(idx%8))
            if occ:
                bit_sum+=1
                init_grid[n] = 1

        return bit_sum

    bit_sum = init_m_transforms()

    final_grid = ti.Vector.field(3, ti.f32, shape=(bit_sum, ))

    @ti.kernel
    def resemable(center: ti.template()):
        count = 0
        ti.loop_config(serialize=True)
        for i,j,k in ti.ndrange(128, 128, 128):
            n = i*128*128 + k * 128 + j
            if init_grid[n]:
                index = ti.atomic_add(count, 1)
                final_grid[index] = ((vec3((127-j), k, (127-i)) + 0.5) / grid_size)*2 - 1 + center

    resemable(vec3(4.0, 1.0, 0.0))
            
    # world
    R = math.cos(math.pi / 4.0)
    world = World()
    world.add(Sphere([0.0, -1000, 0], 1000.0, mat_ground))

    static_point = Point(4.0, 0.2, 0.0)
    print("gen cubes")
    # for a in tqdm.tqdm(range(-5, 5)):
    #     for b in range(-5, 5):
    #         choose_mat = random.random()
    #         center = Point(a + 0.9 * random.random(), 0.1,
    #                        b + 0.9 * random.random())

    #         if (center - static_point).norm() > 0.9:
    #             if choose_mat < 0.8:
    #                 # diffuse
    #                 mat = Lambert(
    #                     Color(random.random(), random.random(),
    #                           random.random())**2)
    #             elif choose_mat < 0.95:
    #                 # metal
    #                 mat = Metal(
    #                     Color(random.random(), random.random(),
    #                           random.random()) * 0.5 + 0.5,
    #                     random.random() * 0.5)
    #             else:
    #                 mat = Dielectric(1.5)

    #         world.add(Cube(center, 0.1, mat))

    world.add(Sphere([0.0, 1.0, 0.0], 1.0, mat3))
    world.add(Sphere([-4.0, 1.0, 0.0], 1.0, mat2))
    # world.add(Sphere([4.0, 1.0, 0.0], 1.0, mat3))

    for i in tqdm.tqdm(range(bit_sum)):
        list_array = final_grid[i].to_list()
        # print(list_array)
        world.add(Cube(list_array, 1/128, mat2))

    
    world.commit()

    # camera
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    up = Vector(0.0, 1.0, 0.0)
    focus_dist = 10.0
    aperture = 0.1
    cam = Camera(vfrom, at, up, 20.0, aspect_ratio, aperture, focus_dist)

    start_attenuation = Vector(1.0, 1.0, 1.0)
    initial = True

    max_depth = 20

    @ti.kernel
    def render_complete():
        for x, y in pixels:
            pdf = attenuation_temp[x, y]
            ray_dir = dir_temp[x, y]
            pixels[x, y] += pdf * ray_dir

    @ti.kernel
    def finish(cnt: int):
        for x, y in pixels:
            pdf = attenuation_temp[x, y]
            ray_dir = dir_temp[x, y]
            pixels[x, y] += pdf * ray_dir
            final_pixels[x, y] = ti.sqrt(pixels[x, y] / cnt)

    @ti.kernel
    def render() -> ti.i32:
        ''' Loops over pixels
            for each pixel:
                generate ray if needed
                intersect scene with ray
                if miss or last bounce sample backgound
            return pixels that hit max samples
        '''
        # num_completed = 0
        save_index = 0
        for x, y in pixels:
            # if sample_count[x, y] == samples_per_pixel:
            #     continue

            # gen sample
            ray_org = Point(0.0, 0.0, 0.0)
            ray_dir = Vector(0.0, 0.0, 0.0)
            # depth = max_depth
            pdf = start_attenuation

            u = (x + ti.random()) / (image_width - 1)
            v = (y + ti.random()) / (image_height - 1)
            ray_org, ray_dir = cam.get_ray(u, v)
            # rays.set(x, y, ray_org, ray_dir, depth, pdf)

            d = 0
            while True:
                d += 1
                # intersect
                hit, p, n, front_facing, index = world.hit_all(ray_org, ray_dir)
                if hit:
                    reflected, out_origin, out_direction, attenuation = world.materials.scatter(
                        index, ray_dir, p, n, front_facing)

                    if reflected:
                        # pdf *= attenuation
                        if world.object_type[index] == 1:
                            cube_info_index = ti.atomic_add(save_index, 1)
                            ray_d_factor = 0.5
                            if d > 1:
                                ray_d_factor = 0.6
                            cube_info_index = ti.atomic_add(save_index, 1)
                            # convert to ngp coordinate
                            #((vec3((127-j), k, (127-i)) + 0.5) / grid_size)*2 - 1 + center
                            pre_ray_o = (ray_org - vec3(4.0, 1.0, 0.0)) * 0.5
                            pre_ray_d = ray_dir * ray_d_factor

                            cube_ray_org[cube_info_index] = vec3(pre_ray_o[2], -pre_ray_o[0], pre_ray_o[1])
                            cube_ray_dir[cube_info_index] = vec3(pre_ray_d[2], -pre_ray_d[0], pre_ray_d[1])
                            cube_ray_id[cube_info_index][0] = x
                            cube_ray_id[cube_info_index][1] = y

                        else:
                            pdf *= attenuation
                    else:
                        attenuation_temp[x, y] = vec3(0.0)
                        break

                    ray_org = out_origin
                    ray_dir = out_direction
                    
                if not hit or d == max_depth:
                    attenuation_temp[x, y] = pdf
                    dir_temp[x, y] = get_background(ray_dir)
                    # pixels[x, y] += pdf * get_background(ray_dir)
                    break

        return save_index

    @ti.kernel
    def comp_pdf_from_ngp(color: ti.template(), opc: ti.template(), count: int):
        for i in ti.ndrange(count):
            x = cube_ray_id[i][0]
            y = cube_ray_id[i][1]
            color_temp = color[i]
            color_temp += vec3(1.0)*(1-opc[i])
            attenuation_temp[x, y] *= color_temp


    scale = 0.5
    ngp = NGP_fw(
        scale=scale, 
        cascades=max(1+int(np.ceil(np.log2(2*scale))), 1),   
        grid_size=128, 
        base_res=16, 
        log2_T=19, 
        n_rays=800*800,
        level=16, 
        exp_step_factor=0,
    )

    model_dir = './npy_models/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    npy_file = os.path.join(model_dir, 'lego.npy')
    if not os.path.exists(npy_file):
        print(f"No lego model found, downloading ...")
        url = PRETRAINED_MODEL_URL.format('lego')
        wget.download(url, out=npy_file)
    ngp.load_model(npy_file)

    ngp.hash_table_init()

    window = ti.ui.Window("Taichi RayTracing", (image_width, image_height), vsync=False)
    canvas = window.get_canvas()
    gui = window.get_gui()
    d = 0
    while window.running:
        d += 1
        # wavefront_initial()
        cube_hit = render()
        ngp.N_rays = cube_hit
        # print("save ray")
        # ray_dict = {'ray_o': cube_ray_org.to_numpy(), 'ray_d': cube_ray_dir.to_numpy()}
        # np.save('ray_dict.npy', ray_dict)
        # assert False
        # print(f"ngp render rays: {cube_hit}")
        # assert False
        ngp.render(max_samples=100, T_threshold=1e-4, ray_o=cube_ray_org, ray_dir=cube_ray_dir)
        # print("ngp render done")
        comp_pdf_from_ngp(ngp.rgb, ngp.opacity, cube_hit)
        # render_complete()
        finish(d)
        canvas.set_image(final_pixels)
        window.show()

