# NeRF-in-RayTracing

- Team: TaiDenG
- Project: NeRF in ray tracing

## Description

Neural radiance field (**NeRF**) is a state-of-the-art novel view synthesis method, which is capable of rendering photorealistic scenes by only training with collective images and camera pose.

However, **NeRF** is using an implicit modeling style that inputs a position with viewing direction and outputs the color and density of that position, which can not be used as a general object file that contains geometric vertices.

As a result, the traditional pipeline of a path tracer is not able to do ray tracing for a learned **NeRF** model. But, there is a workaround that you can extract the geometric and color from **NeRF** using methods like marching cubes, so it can be compatible with a common renderer.

Unfortunately, the extraction is not perfect, the extracted model will have lower quality than the original **NeRF**.

This project aims to integrate the **NeRF** model into a ray tracer so that the **NeRF** can be directly rendered and participate in ray tracing without previously extracting the geometric and color.
