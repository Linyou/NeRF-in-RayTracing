import taichi as ti
from taichi_glsl.vector import reflect
from vector import *
import numpy as np


@ti.func
def reflectance(cosine, idx):
    r0 = ((1.0 - idx) / (1.0 + idx))**2
    return r0 + (1.0 - r0) * ((1.0 - cosine)**5)


@ti.func
def reflect(v, n):
    return v - 2.0 * v.dot(n) * n


@ti.func
def refract(v, n, etai_over_etat):
    cos_theta = min(-v.dot(n), 1.0)
    r_out_perp = etai_over_etat * (v + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.norm_sqr())) * n
    return r_out_perp + r_out_parallel


class _material:
    def scatter(self, in_direction, p, n):
        pass

@ti.func
def near_zero(v):
    s = 1e-8
    v_abs = ti.abs(v)
    return (v_abs[0] < s) and (v_abs[1] < s) and (v_abs[2] < s)

class Lambert(_material):
    def __init__(self, color):
        self.color = color
        self.index = 0
        self.roughness = 0.0
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color):
        out_direction = n + random_in_hemisphere(n)
        if near_zero(out_direction):
            out_direction = n
        attenuation = color
        return True, p, out_direction, attenuation


class Metal(_material):
    def __init__(self, color, roughness):
        self.color = color
        self.index = 1
        self.roughness = min(roughness, 1.0)
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color, roughness):
        out_direction = reflect(in_direction.normalized(),
                                n) + roughness * random_in_unit_sphere()
        attenuation = color
        reflected = out_direction.dot(n) > 0.0
        return reflected, p, out_direction, attenuation


class Dielectric(_material):
    def __init__(self, ior):
        self.color = Color(1.0, 1.0, 1.0)
        self.index = 2
        self.roughness = 0.0
        self.ior = ior

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color, ior, front_facing):
        refraction_ratio = 1.0 / ior if front_facing else ior
        unit_dir = in_direction.normalized()
        cos_theta = min(-unit_dir.dot(n), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

        out_direction = Vector(0.0, 0.0, 0.0)
        cannot_refract = refraction_ratio * sin_theta > 1.0
        if cannot_refract or reflectance(cos_theta,
                                         refraction_ratio) > ti.random():
            out_direction = reflect(unit_dir, n)
        else:
            out_direction = refract(unit_dir, n, refraction_ratio)
        attenuation = color

        return True, p, out_direction, attenuation


@ti.data_oriented
class Materials:
    ''' List of materials for a scene.'''
    def __init__(self, n):
        self.roughness = ti.field(ti.f32)
        self.colors = ti.Vector.field(3, dtype=ti.f32)
        self.mat_index = ti.field(ti.u32)
        self.ior = ti.field(ti.f32)
        ti.root.dense(ti.i, n).place(self.roughness, self.colors,
                                     self.mat_index, self.ior)

        self.data_file = './lego_materials.npy'

    def set(self, i, material):
        self.colors[i] = material.color
        self.mat_index[i] = material.index
        self.roughness[i] = material.roughness
        self.ior[i] = material.ior

    def save(self):
        colors = self.colors.to_numpy()
        mat_index = self.mat_index.to_numpy()
        roughness = self.roughness.to_numpy()
        ior = self.ior.to_numpy()

        save_dict = {
            'colors': colors,
            'mat_index': mat_index,
            'roughness': roughness,
            'ior': ior
        }
        print("save materials")
        np.save(self.data_file , save_dict)

    def load(self):
        save_dict = np.load(self.data_file, allow_pickle=True).item()
        colors = save_dict['colors']
        mat_index = save_dict['mat_index']
        roughness = save_dict['roughness']
        ior = save_dict['ior']

        self.colors.from_numpy(colors)
        self.mat_index.from_numpy(mat_index)
        self.roughness.from_numpy(roughness)
        self.ior.from_numpy(ior)
        print("load materials")

    @ti.func
    def scatter(self, i, ray_direction, p, n, front_facing):
        ''' Get the scattered ray that hits a material '''
        mat_index = self.mat_index[i]
        color = self.colors[i]
        roughness = self.roughness[i]
        ior = self.ior[i]
        reflected = True
        out_origin = Point(0.0, 0.0, 0.0)
        out_direction = Vector(0.0, 0.0, 0.0)
        attenuation = Color(0.0, 0.0, 0.0)

        if mat_index == 0:
            reflected, out_origin, out_direction, attenuation = Lambert.scatter(
                ray_direction, p, n, color)
        elif mat_index == 1:
            reflected, out_origin, out_direction, attenuation = Metal.scatter(
                ray_direction, p, n, color, roughness)
        else:
            reflected, out_origin, out_direction, attenuation = Dielectric.scatter(
                ray_direction, p, n, color, ior, front_facing)
        return reflected, out_origin, out_direction, attenuation
