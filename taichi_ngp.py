from multiprocessing import shared_memory
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import taichi as ti
from taichi.math import uvec3, vec3, vec2
import wget
import cv2
import platform

from typing import Tuple

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

arch = ti.cuda if ti._lib.core.with_cuda() else ti.vulkan

if platform.system() == 'Darwin':
    block_dim = 64
else:
    block_dim = 128

sigma_sm_preload = int(128/block_dim)*24
rgb_sm_preload = int(128/block_dim)*50
data_type = ti.f16
np_type = np.float16
tf_vec3 = ti.types.vector(3, dtype=data_type)
tf_vec8 = ti.types.vector(8, dtype=data_type)
tf_vec32 = ti.types.vector(32, dtype=data_type)
tf_vec1 = ti.types.vector(1, dtype=data_type)
tf_vec2 = ti.types.vector(2, dtype=data_type)
tf_mat1x3 = ti.types.matrix(1, 3, dtype=data_type)
tf_index_temp = ti.types.vector(8, dtype=ti.i32)

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01

SQRT3 = 1.7320508075688772
SQRT3_MAX_SAMPLES = SQRT3/1024
SQRT3_2 = 1.7320508075688772*2
PRETRAINED_MODEL_URL = 'https://github.com/Linyou/taichi-ngp-renderer/releases/download/v0.1-models/{}.npy'

#<----------------- hash table util code ----------------->
@ti.func
def calc_dt(t, exp_step_factor, grid_size, scale):
    return data_type(ti.math.clamp(t*exp_step_factor, SQRT3_MAX_SAMPLES, SQRT3_2*scale/grid_size))

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

@ti.func
def fast_hash(pos_grid_local):
    result = ti.uint32(0)
    primes = uvec3(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861))
    for i in ti.static(range(3)):
        result ^= ti.uint32(pos_grid_local[i]) * primes[i]
    return result

@ti.func
def under_hash(pos_grid_local, resolution):
    result = ti.uint32(0)
    stride = ti.uint32(1)
    for i in ti.static(range(3)):
        result += ti.uint32(pos_grid_local[i] * stride)
        stride *= resolution
    return result

@ti.func
def grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size):
    hash_result = ti.uint32(0)
    if indicator == 1:
        hash_result = under_hash(pos_grid_local, resolution)
    else:
        hash_result = fast_hash(pos_grid_local)

    return hash_result % map_size


#<----------------- hash table util code ----------------->

@ti.func
def random_in_unit_disk():
    theta = 2.0 * np.pi * ti.random()
    return ti.Vector([ti.sin(theta), ti.cos(theta)])

@ti.func
def random_normal():
    x = ti.random() * 2. - 1.
    y = ti.random() * 2. - 1.
    return tf_vec2(x, y)

@ti.func
def dir_encode_func(dir_):
    input = tf_vec32(0.0)
    dir = dir_/dir_.norm()
    x = dir[0]; y = dir[1]; z = dir[2]
    xy= x*y; xz= x*z; yz= y*z; x2= x*x; y2= y*y; z2= z*z
    
    temp = 0.28209479177387814
    input[0] = data_type(temp); input[1] = data_type(-0.48860251190291987*y); input[2] = data_type(0.48860251190291987*z)
    input[3] = data_type(-0.48860251190291987*x); input[4] = data_type(1.0925484305920792*xy); input[5] = data_type(-1.0925484305920792*yz)
    input[6] = data_type(0.94617469575755997*z2 - 0.31539156525251999); input[7] = data_type(-1.0925484305920792*xz)
    input[8] = data_type(0.54627421529603959*x2 - 0.54627421529603959*y2); input[9] = data_type(0.59004358992664352*y*(-3.0*x2 + y2))
    input[10] = data_type(2.8906114426405538*xy*z); input[11] = data_type(0.45704579946446572*y*(1.0 - 5.0*z2))
    input[12] = data_type(0.3731763325901154*z*(5.0*z2 - 3.0)); input[13] = data_type(0.45704579946446572*x*(1.0 - 5.0*z2))
    input[14] = data_type(1.4453057213202769*z*(x2 - y2)); input[15] = data_type(0.59004358992664352*x*(-x2 + 3.0*y2))

    return input

@ti.data_oriented
class NGP_fw:
    def __init__(self, scale, cascades, grid_size, base_res, log2_T, n_rays, level, exp_step_factor, center, xyz_min, xyz_max):
        self.N_rays = n_rays
        self.grid_size = grid_size
        self.exp_step_factor = exp_step_factor
        self.scale = scale

        # rays intersection parameters
        # t1, t2 need to be initialized to -1.0
        self.hits_t = ti.Vector.field(n=2, dtype=data_type, shape=(self.N_rays))
        self.hits_t.fill(-1.0)
        self.center = center
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.half_size = (self.xyz_max - self.xyz_min) / 2

        self.rays_o = ti.Vector.field(n=3, dtype=data_type, shape=(self.N_rays))
        self.rays_d = ti.Vector.field(n=3, dtype=data_type, shape=(self.N_rays))

        # density_bitfield is used for point sampling
        self.density_bitfield = ti.field(ti.uint8, shape=(cascades*grid_size**3//8))

        # count the number of rays that still alive
        self.counter = ti.field(ti.i32, shape=())
        self.counter[None] = self.N_rays
        # current alive buffer index
        self.current_index = ti.field(ti.i32, shape=())
        self.current_index[None] = 0

        # how many samples that need to run the model
        self.model_launch = ti.field(ti.i32, shape=())

        # buffer for the alive rays
        self.alive_indices = ti.field(ti.i32, shape=(2*self.N_rays,))

        # padd the thread to the factor of block size (thread per block)
        self.padd_block_network = ti.field(ti.i32, shape=())
        self.padd_block_composite = ti.field(ti.i32, shape=())

        # hash table variables
        self.min_samples = 1 if exp_step_factor==0 else 4
        self.per_level_scales = 1.3195079565048218 # hard coded, otherwise it will be have lower percision
        self.base_res = base_res
        self.max_params = 2**log2_T
        self.level = level
        # hash table fields
        self.offsets = ti.field(ti.i32, shape=(16,))
        self.hash_map_sizes = ti.field(ti.uint32, shape=(16,))
        self.hash_map_indicator = ti.field(ti.i32, shape=(16,))

        # model parameters
        layer1_base = 32 * 64
        layer2_base = layer1_base + 64 * 64
        self.hash_embedding= ti.field(dtype=data_type, shape=(11445040,))
        self.sigma_weights= ti.field(dtype=data_type, shape=(layer1_base + 64*16,))
        self.rgb_weights= ti.field(dtype=data_type, shape=(layer2_base+64*8,))

        # buffers that used for points sampling 
        self.max_samples_per_rays = 1
        self.max_samples_shape = self.N_rays * self.max_samples_per_rays

        self.xyzs = ti.Vector.field(3, dtype=data_type, shape=(self.max_samples_shape,))
        self.dirs = ti.Vector.field(3, dtype=data_type, shape=(self.max_samples_shape,))
        self.deltas = ti.field(data_type, shape=(self.max_samples_shape,))
        self.ts = ti.field(data_type, shape=(self.max_samples_shape,))

        # buffers that store the info of sampled points
        self.run_model_ind = ti.field(ti.int32, shape=(self.max_samples_shape,))
        self.N_eff_samples = ti.field(ti.int32, shape=(self.N_rays,))

        # intermediate buffers for network
        self.xyzs_embedding = ti.field(data_type, shape=(self.max_samples_shape, 32))
        self.final_embedding = ti.field(data_type, shape=(self.max_samples_shape, 16))
        self.out_3 = ti.field(data_type, shape=(self.max_samples_shape, 3))
        self.out_1 = ti.field(data_type, shape=(self.max_samples_shape,))
        self.temp_hit = ti.field(ti.i32, shape=(self.max_samples_shape,))

        # results buffers
        self.opacity = ti.field(ti.f32, shape=(self.N_rays,))
        self.depth = ti.field(ti.f32, shape=(self.N_rays))
        self.rgb = ti.Vector.field(3, dtype=ti.f32, shape=(self.N_rays,))


    def hash_table_init(self):
        print(f'GridEncoding: base resolution: {self.base_res}, log scale per level:{self.per_level_scales:.5f} feature numbers per level: {2} maximum parameters per level: {self.max_params} level: {self.level}')
        offset = 0
        for i in range(self.level):
            resolution = int(np.ceil(self.base_res * np.exp(i*np.log(self.per_level_scales)) - 1.0)) + 1
            params_in_level = resolution ** 3
            params_in_level = int(resolution ** 3) if params_in_level % 8 == 0 else int((params_in_level + 8 - 1) / 8) * 8
            params_in_level = min(self.max_params, params_in_level)
            self.offsets[i] = offset
            self.hash_map_sizes[i] = params_in_level
            self.hash_map_indicator[i] = 1 if resolution ** 3 <= params_in_level else 0
            offset += params_in_level

    def load_model(self, model_path):
        print('Loading model from {}'.format(model_path))
        model = np.load(model_path, allow_pickle=True).item()
        # model = torch.load(model_path, map_location='cpu')['state_dict']
        self.hash_embedding.from_numpy(model['model.xyz_encoder.params'].astype(np_type))
        self.sigma_weights.from_numpy(model['model.xyz_sigmas.params'].astype(np_type))
        self.rgb_weights.from_numpy(model['model.rgb_net.params'].astype(np_type))

        self.density_bitfield.from_numpy(model['model.density_bitfield'])


    @ti.kernel
    def reset(self):
        self.depth.fill(0.0)
        self.opacity.fill(0.0)
        self.counter[None] = self.N_rays
        for i, j in ti.ndrange(self.N_rays, 2):
            self.alive_indices[i*2+j] = i    

    @ti.func
    def _ray_aabb_intersec(self, ray_o, ray_d):
        inv_d = 1.0 / ray_d

        t_min = (self.center-self.half_size-ray_o)*inv_d
        t_max = (self.center+self.half_size-ray_o)*inv_d

        _t1 = ti.min(t_min, t_max)
        _t2 = ti.max(t_min, t_max)
        t1 = _t1.max()
        t2 = _t2.min()

        return tf_vec2(t1, t2)


    @ti.kernel
    def ray_intersect(self, ray_o: ti.template(), ray_d: ti.template()):
        for i in range(self.N_rays): 
            ray_d_ = ray_d[i]
            ray_o_ = ray_o[i]
            
            t1t2 = self._ray_aabb_intersec(ray_o_, ray_d_)

            if t1t2[1] > 0.0:
                self.hits_t[i][0] = data_type(ti.max(t1t2[0], NEAR_DISTANCE))
                self.hits_t[i][1] = t1t2[1]  

            self.rays_o[i] = ray_o_
            self.rays_d[i] = ray_d_

    @ti.kernel
    def raymarching_test_kernel(self, N_samples: int):

        self.run_model_ind.fill(0)
        for n in ti.ndrange(self.counter[None]):
            c_index = self.current_index[None]
            r = self.alive_indices[n*2+c_index]
            grid_size3 = self.grid_size**3
            grid_size_inv = 1.0/self.grid_size

            ray_o = self.rays_o[r]
            ray_d = self.rays_d[r]
            t1t2 = self.hits_t[r]

            d_inv = 1.0/ray_d

            t = t1t2[0]
            t2 = t1t2[1]

            s = 0

            start_idx = n * N_samples

            while (0<=t) & (t<t2) & (s<N_samples):
                # xyz = ray_o + t*ray_d
                xyz = ray_o + t*ray_d
                dt = calc_dt(t, self.exp_step_factor, self.grid_size, self.scale)
                # mip = ti.max(mip_from_pos(xyz, cascades),
                #             mip_from_dt(dt, grid_size, cascades))


                mip_bound = 0.5
                mip_bound_inv = 1/mip_bound

                nxyz = ti.math.clamp(0.5*(xyz*mip_bound_inv+1)*self.grid_size, 0.0, self.grid_size-1.0)
                # nxyz = ti.ceil(nxyz)

                idx =  __morton3D(ti.cast(nxyz, ti.u32))
                # occ = density_grid_taichi[idx] > 5.912066756501768
                occ = self.density_bitfield[ti.u32(idx//8)] & (1 << ti.u32(idx%8))

                if occ:
                    sn = start_idx + s
                    for p in ti.static(range(3)):
                        self.xyzs[sn][p] = xyz[p]
                        self.dirs[sn][p] = ray_d[p]
                    self.run_model_ind[sn] = 1
                    self.ts[sn] = t
                    self.deltas[sn] = dt
                    t += dt
                    self.hits_t[r][0] = t
                    s += 1

                else:
                    txyz = (((nxyz+0.5+0.5*ti.math.sign(ray_d))*grid_size_inv*2-1)*mip_bound-xyz)*d_inv

                    t_target = t + ti.max(0, txyz.min())
                    t += calc_dt(t, self.exp_step_factor, self.grid_size, self.scale)
                    while t < t_target:
                        t += calc_dt(t, self.exp_step_factor, self.grid_size, self.scale)

            self.N_eff_samples[n] = s
            if s == 0:
                self.alive_indices[n*2+c_index] = -1

    @ti.kernel
    def rearange_index(self, B: ti.i32):
        self.model_launch[None] = 0
        
        for i in ti.ndrange(B):
            if self.run_model_ind[i]:
                index = ti.atomic_add(self.model_launch[None], 1)
                self.temp_hit[index] = i

        self.model_launch[None] += 1
        self.padd_block_network[None] = ((self.model_launch[None]+ block_dim - 1)// block_dim) *block_dim
        # self.padd_block_composite[None] = ((self.counter[None]+ 128 - 1)// 128) *128

    @ti.kernel
    def hash_encode(self):
        # get hash table embedding
        ti.loop_config(block_dim=16)
        for sn, level in ti.ndrange(self.model_launch[None], 16):
            # normalize to [0, 1], before is [-0.5, 0.5]
            xyz = self.xyzs[self.temp_hit[sn]] + 0.5
            offset = self.offsets[level] * 2
            indicator = self.hash_map_indicator[level]
            map_size = self.hash_map_sizes[level]

            init_val0 = tf_vec1(0.0)
            init_val1 = tf_vec1(1.0)
            local_feature_0 = init_val0[0]
            local_feature_1 = init_val0[0]

            index_temp = tf_index_temp(0)
            w_temp = tf_vec8(0.0)
            hash_temp_1 = tf_vec8(0.0)
            hash_temp_2 = tf_vec8(0.0)

            scale = self.base_res * ti.exp(level*ti.log(self.per_level_scales)) - 1.0
            resolution = ti.cast(ti.ceil(scale), ti.uint32) + 1

            pos = xyz * scale + 0.5
            pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)
            pos -= pos_grid_uint
            # pos_grid_uint = ti.cast(pos_grid, ti.uint32)

            for idx in ti.static(range(8)):
                # idx_uint = ti.cast(idx, ti.uint32)
                w = init_val1[0]
                pos_grid_local = uvec3(0)

                for d in ti.static(range(3)):
                    if (idx & (1 << d)) == 0:
                        pos_grid_local[d] = pos_grid_uint[d]
                        w *= data_type(1 - pos[d])
                    else:
                        pos_grid_local[d] = pos_grid_uint[d] + 1
                        w *= data_type(pos[d])

                index = ti.int32(grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size))
                index_temp[idx] = offset+index*2
                w_temp[idx] = w

            for idx in ti.static(range(8)):
                hash_temp_1[idx] = self.hash_embedding[index_temp[idx]]
                hash_temp_2[idx] = self.hash_embedding[index_temp[idx]+1]

            for idx in ti.static(range(8)):
                local_feature_0 += data_type(w_temp[idx] * hash_temp_1[idx])
                local_feature_1 += data_type(w_temp[idx] * hash_temp_2[idx])

            self.xyzs_embedding[sn, level*2] = local_feature_0
            self.xyzs_embedding[sn, level*2+1] = local_feature_1

    @ti.kernel
    def sigma_layer(self):
        ti.loop_config(block_dim=block_dim)
        for sn in ti.ndrange(self.padd_block_network[None]):
            tid = sn % block_dim
            did_launch_num = self.model_launch[None]
            init_val = tf_vec1(0.0)
            input = ti.simt.block.SharedArray((32, block_dim), data_type)
            weight = ti.simt.block.SharedArray((64*32+64*16,), data_type)
            hid1 = ti.simt.block.SharedArray((64, block_dim), data_type)
            hid2 = ti.simt.block.SharedArray((16, block_dim), data_type)
            for i in ti.static(range(sigma_sm_preload)):
                k = tid*sigma_sm_preload+i
                weight[k] = self.sigma_weights[k]
            ti.simt.block.sync()

            if sn < did_launch_num:
                
                for i in ti.static(range(32)):
                    input[i, tid] = self.xyzs_embedding[sn, i]

                for i in range(64):
                    temp = init_val[0]
                    for j in ti.static(range(32)):
                        temp += input[j, tid] * weight[i*32+j]

                    hid1[i, tid] = temp
                ti.simt.block.sync()
                
                for i in range(16):
                    temp = init_val[0]
                    for j in ti.static(range(64)):
                        temp += data_type(ti.max(0.0, hid1[j, tid])) * weight[64*32+i*64+j]
                    hid2[i, tid] = temp
                ti.simt.block.sync()

                self.out_1[self.temp_hit[sn]] = data_type(ti.exp(hid2[0, tid]))
                for i in ti.static(range(16)):
                    self.final_embedding[sn, i] = hid2[i, tid]
                
                ti.simt.block.sync()

    @ti.kernel
    def rgb_layer(self):
        ti.loop_config(block_dim=block_dim)
        for sn in ti.ndrange(self.padd_block_network[None]):
            ray_id = self.temp_hit[sn]
            tid = sn % block_dim
            did_launch_num = self.model_launch[None]
            init_val = tf_vec1(0.0)
            weight = ti.simt.block.SharedArray((64*32+64*64+64*4,), data_type)
            hid1 = ti.simt.block.SharedArray((64, block_dim), data_type)
            hid2 = ti.simt.block.SharedArray((64, block_dim), data_type)
            for i in ti.static(range(rgb_sm_preload)):
                k = tid*rgb_sm_preload+i
                weight[k] = self.rgb_weights[k]
            ti.simt.block.sync()

            if sn < did_launch_num:
                
                dir_ = self.dirs[ray_id]
                input = dir_encode_func(dir_)

                for i in ti.static(range(16)):
                    input[16+i] = self.final_embedding[sn, i]

                for i in range(64):
                    temp = init_val[0]
                    for j in ti.static(range(32)):
                        temp += input[j] * weight[i*32+j]

                    hid1[i, tid] = temp
                ti.simt.block.sync()

                for i in range(64):
                    temp = init_val[0]
                    for j in ti.static(range(64)):
                        temp += data_type(ti.max(0.0, hid1[j, tid])) * weight[64*32+i*64+j]

                    hid2[i, tid] = temp
                ti.simt.block.sync()

                for i in ti.static(range(3)):
                    temp = init_val[0]
                    for j in ti.static(range(64)):
                        temp += data_type(ti.max(0.0, hid2[j, tid])) * weight[64*32+64*64+i*64+j]

                    hid1[i, tid] = temp
                ti.simt.block.sync()

                for i in ti.static(range(3)):
                    self.out_3[self.temp_hit[sn], i] = data_type(1 / (1 + ti.exp(-hid1[i, tid])))
                ti.simt.block.sync()

    @ti.kernel
    def composite_test(self, max_samples: ti.i32, T_threshold: data_type):
        for n in ti.ndrange(self.counter[None]):
            N_samples = self.N_eff_samples[n]
            if N_samples != 0:
                c_index = self.current_index[None]
                r = self.alive_indices[n*2+c_index]

                T = data_type(1.0 - self.opacity[r])

                start_idx = n * max_samples

                rgb_temp = tf_vec3(0.0)
                depth_temp = tf_vec1(0.0)
                opacity_temp = tf_vec1(0.0)
                out_3_temp = tf_vec3(0.0)

                for s in range(N_samples):
                    sn = start_idx + s
                    a = data_type(1.0 - ti.exp(-self.out_1[sn]*self.deltas[sn]))
                    w = a * T

                    for i in ti.static(range(3)):
                        out_3_temp[i] = self.out_3[sn, i]

                    rgb_temp += w * out_3_temp
                    depth_temp[0] += w * self.ts[sn]
                    opacity_temp[0] += w

                    T *= data_type(1.0 - a)

                    if T <= T_threshold:
                        self.alive_indices[n*2+c_index] = -1
                        break


                self.rgb[r] += rgb_temp
                self.depth[r] += depth_temp[0]
                self.opacity[r] += opacity_temp[0]

    @ti.kernel
    def re_order(self, B: ti.i32):

        self.counter[None] = 0
        c_index = self.current_index[None]
        n_index = (c_index + 1) % 2
        self.current_index[None] = n_index

        for i in ti.ndrange(B):
            alive_temp = self.alive_indices[i*2+c_index]
            if alive_temp >= 0:
                index = ti.atomic_add(self.counter[None], 1)
                self.alive_indices[index*2+n_index] = alive_temp


    def write_image(self):
        rgb_np = self.rgb.to_numpy().reshape(self.res[0], self.res[1], 3)
        depth_np = self.depth.to_numpy().reshape(self.res[0], self.res[1])
        plt.imsave('taichi_ngp.png', (rgb_np*255).astype(np.uint8))
        plt.imsave('taichi_ngp_depth.png', depth2img(depth_np))

    @ti.kernel
    def set_final_color(self):
        for i in ti.ndrange(self.N_rays):
            rgb_bg = vec3(1.0)
            self.rgb[i] += rgb_bg*(1-self.opacity[i])

    def render(self, max_samples, T_threshold, ray_o, ray_dir, use_dof=False, dist_to_focus=0.8, len_dis=0.0) -> Tuple[float, int, int]:
        samples = 0
        self.reset()
        self.ray_intersect(ray_o, ray_dir)
        print("done ray intersect")

        while samples < max_samples:
            N_alive = self.counter[None]
            if N_alive == 0: break

            # how many more samples the number of samples add for each ray
            N_samples = max(min(self.N_rays//N_alive, 64), self.min_samples)
            samples += N_samples
            launch_model_total = N_alive * N_samples

            print(f"samples: {samples}, N_alive: {N_alive}, N_samples: {N_samples}")
            self.raymarching_test_kernel(N_samples)
            print("done raymarching")
            self.rearange_index(launch_model_total)
            print("done rearange index")
            # self.dir_encode()
            self.hash_encode()
            print("done hash encode")
            self.sigma_layer()
            print("done sigma layer")
            self.rgb_layer()
            print("done rgb layer")
            # self.FullyFusedMLP()
            self.composite_test(N_samples, T_threshold)
            print("done composite test")
            self.re_order(N_alive)
            print("done re order")

        self.set_final_color()

        return samples, N_alive, N_samples

    def render_frame(self, frame_id):
        t = time.time()
        samples, N_alive, N_samples = self.render(max_samples=100, T_threshold=1e-4)
        self.write_image()

        print(f"samples: {samples}, N_alive: {N_alive}, N_samples: {N_samples}")
        print(f'Render time: {1000*(time.time()-t):.2f} ms')



def main(args):
    NGP_fw.taichi_init(args.print_profile)
    res = args.res
    scale = 0.5
    ngp = NGP_fw(
        scale=scale, 
        cascades=max(1+int(np.ceil(np.log2(2*scale))), 1),   
        grid_size=128, 
        base_res=16, 
        log2_T=19, 
        res=[res, res], 
        level=16, 
        exp_step_factor=0
    )
    if args.model_path:
        ngp.load_model(args.model_path)
    else:
        model_dir = './npy_models/'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        npy_file = os.path.join(model_dir, args.scene+'.npy')
        if not os.path.exists(npy_file):
            print(f"No {args.scene} model found, downloading ...")
            url = PRETRAINED_MODEL_URL.format(args.scene)
            wget.download(url, out=npy_file)
        ngp.load_model(npy_file)

    ngp.hash_table_init()

    ngp.render_frame(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=800)
    parser.add_argument('--scene', type=str, default='lego',
                        choices=['ship', 'mic', 'materials', 'lego', 'hotdog', 'ficus', 'drums', 'chair'],)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gui', action='store_true', default=False)
    parser.add_argument('--print_profile', action='store_true', default=False)
    args = parser.parse_args()
    main(args)