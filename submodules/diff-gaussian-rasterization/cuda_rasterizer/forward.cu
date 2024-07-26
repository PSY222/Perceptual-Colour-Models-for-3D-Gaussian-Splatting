/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <algorithm>
#include <cmath>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_count(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int* __restrict__ gaussian_count,
	float* __restrict__ important_score)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			gaussian_count[collected_id[j]]++; // add count 
			important_score[collected_id[j]] += con_o.w; // opacity


			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::count_gaussian(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	int* gaussians_count,
	float* important_score,
	float* out_color)
{
	renderCUDA_count<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		gaussians_count,
		important_score);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_bw_score(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ blending_weight_score)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			blending_weight_score[collected_id[j]] += alpha * T; // blending weight


			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::bw_score_gaussian(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* blending_weight_score,
	float* out_color)
{
	renderCUDA_bw_score<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		blending_weight_score);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_mw_score(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ max_weight_score)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			max_weight_score[collected_id[j]] = max(max_weight_score[collected_id[j]], alpha * T); // max weight


			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::mw_score_gaussian(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* max_weight_score,
	float* out_color)
{
	renderCUDA_mw_score<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		max_weight_score);
}


// Added NEW ssim calculation function

template <uint32_t CHANNELS>
__device__ float calculate_ssim(const float* gt_color, const float* prim_color) {
    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;

    // Calculate mean
    float mu_gt = 0.0f;
    float mu_prim = 0.0f;
    for (uint32_t ch = 0; ch < CHANNELS; ch++) {
        mu_gt += gt_color[ch];
        mu_prim += prim_color[ch];
    }
    mu_gt /= CHANNELS;
    mu_prim /= CHANNELS;

    // Calculate variance and covariance
    float sigma_gt_sq = 0.0f;
    float sigma_prim_sq = 0.0f;
    float sigma_gt_prim = 0.0f;
    for (uint32_t ch = 0; ch < CHANNELS; ch++) {
        sigma_gt_sq += (gt_color[ch] - mu_gt) * (gt_color[ch] - mu_gt);
        sigma_prim_sq += (prim_color[ch] - mu_prim) * (prim_color[ch] - mu_prim);
        sigma_gt_prim += (gt_color[ch] - mu_gt) * (prim_color[ch] - mu_prim);
    }
    sigma_gt_sq /= CHANNELS;
    sigma_prim_sq /= CHANNELS;
    sigma_gt_prim /= CHANNELS;

    // Calculate SSIM
    float ssim_numerator = (2.0f * mu_gt * mu_prim + C1) * (2.0f * sigma_gt_prim + C2);
    float ssim_denominator = (mu_gt * mu_gt + mu_prim * mu_prim + C1) * (sigma_gt_sq + sigma_prim_sq + C2);
    return ssim_numerator / ssim_denominator;
}

template <uint32_t CHANNELS>
__device__ float calculate_ms_ssim(const float* gt_color, const float* prim_color, int height, int width) {
    const int num_scales = 2; // Number of scales in the MS-SSIM calculation
    const float weights[num_scales] = {0.0448f, 0.2856f}; // Weight factors for each scale
    const float sigma = 1.5f; // Sigma value for Gaussian filter

    // Temporary arrays for intermediate results
    float mu_x[num_scales], mu_y[num_scales];
    float sigma_x_sq[num_scales], sigma_y_sq[num_scales], sigma_xy[num_scales];

    // Initialize arrays
    for (int s = 0; s < num_scales; s++) {
        mu_x[s] = 0.0f;
        mu_y[s] = 0.0f;
        sigma_x_sq[s] = 0.0f;
        sigma_y_sq[s] = 0.0f;
        sigma_xy[s] = 0.0f;
    }

    // Compute SSIM at each scale
    float ms_ssim = 1.0f;
    for (int s = 0; s < num_scales; s++) {
        int current_height = height / (1 << s); // Calculate current height at scale s
        int current_width = width / (1 << s); // Calculate current width at scale s

        // Create Gaussian kernel for filtering
        // Apply Gaussian filter to gt_color and prim_color (optional)

        // Calculate SSIM at the current scale
        float ssim = calculate_ssim<CHANNELS>(gt_color, prim_color);

        // Update intermediate results for MS-SSIM
        for (uint32_t ch = 0; ch < CHANNELS; ch++) {
            mu_x[s] += gt_color[ch] / (current_height * current_width);
            mu_y[s] += prim_color[ch] / (current_height * current_width);
        }

        ms_ssim *= max(ssim, 1e-10f); // Avoid potential numerical issues with very small SSIM values
    }

    // Combine SSIM values from different scales using weights
    for (int s = 0; s < num_scales - 1; s++) {
        ms_ssim *= powf(max(mu_x[s] * mu_y[s], 1e-10f), weights[s]);
    }

    return ms_ssim;
}

// Helper function to convert RGB to XYZ
__device__ void rgb_to_xyz(const float* rgb, float* xyz) {
    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];

    // Linearize sRGB values
    r = r > 0.04045 ? powf((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = g > 0.04045 ? powf((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = b > 0.04045 ? powf((b + 0.055) / 1.055, 2.4) : b / 12.92;

    // Convert to XYZ using the sRGB D65 conversion formula
    xyz[0] = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    xyz[1] = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    xyz[2] = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
}


//New color model conversion, calculation added
// Helper function to clamp values
__device__ float clamp(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

// Helper function to convert xyz to lab
__device__ void xyz_to_lab(const float* xyz, float* lab) {
    // Reference white point for D65 illuminant
    const float Xn =  95.047f; // D65
    const float Yn = 100.000f;
    const float Zn = 108.883f;

    float x = xyz[0] / Xn;
    float y = xyz[1] / Yn;
    float z = xyz[2] / Zn;

    // Apply the f(t) function
    auto f = [](float t) -> float {
        return t > 0.008856 ? powf(t, 1.0f / 3.0f) : (t * 903.3f + 16.0f) / 116.0f;
    };

    float fx = f(x);
    float fy = f(y);
    float fz = f(z);

    lab[0] = 116.0f * fy - 16.0f; // L*
    lab[1] = 500.0f * (fx - fy);   // a*
    lab[2] = 200.0f * (fy - fz);    // b*
}

__device__ void rgb_to_lab(const float* rgb, float* lab) {
    float xyz[3];
    rgb_to_xyz(rgb, xyz);
    xyz_to_lab(xyz, lab);
}

__device__ void rgb_to_hsv(const float* rgb, float* hsv) {
    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];

    float cmax = fmaxf(r, fmaxf(g, b));
    float cmin = fminf(r, fminf(g, b));
    float delta = cmax - cmin;

    // Hue calculation
    float h = 0.0f;
    if (delta > 0.0f) {
        if (cmax == r) {
            h = fmodf((g - b) / delta, 6.0f);
        } else if (cmax == g) {
            h = (b - r) / delta + 2.0f;
        } else if (cmax == b) {
            h = (r - g) / delta + 4.0f;
        }
        h *= 60.0f;
        if (h < 0.0f) h += 360.0f;
    }

    // Saturation calculation
    float s = (cmax > 0.0f) ? (delta / cmax) : 0.0f;

    // Value calculation
    float v = cmax;

    hsv[0] = h;   // Hue
    hsv[1] = s;   // Saturation
    hsv[2] = v;   // Value
}


// Helper function to convert XYZ to LUV
__device__ void xyz_to_luv(const float* xyz, float* luv) {
    float X = xyz[0];
    float Y = xyz[1];
    float Z = xyz[2];

    // Reference white point (D65)
    const float Xr = 0.95047f;
    const float Yr = 1.00000f;
    const float Zr = 1.08883f;

    // Calculate u' and v'
    float u_prime = 4 * X / (X + 15 * Y + 3 * Z);
    float v_prime = 9 * Y / (X + 15 * Y + 3 * Z);

    float ur_prime = 4 * Xr / (Xr + 15 * Yr + 3 * Zr);
    float vr_prime = 9 * Yr / (Xr + 15 * Yr + 3 * Zr);

    // Calculate L*
    float L = Y / Yr;
    L = L > 0.008856 ? powf(L, 1.0 / 3.0) * 116.0f - 16.0f : 903.3f * L;

    // Calculate u* and v*
    float u = 13 * L * (u_prime - ur_prime);
    float v = 13 * L * (v_prime - vr_prime);

    // Store L*, u*, v*
    luv[0] = L;
    luv[1] = u;
    luv[2] = v;
}

__device__ float ciede2000(const float* lab1, const float* lab2) {
    // Constants
    const float K_L = 1.0f;
    const float K_C = 1.0f;
    const float K_H = 1.0f;
    
    // LAB1 and LAB2
    float L1 = lab1[0];
    float a1 = lab1[1];
    float b1 = lab1[2];
    float L2 = lab2[0];
    float a2 = lab2[1];
    float b2 = lab2[2];

    // Calculate differences
    float dL = L2 - L1;
    float dC = sqrtf(a2 * a2 + b2 * b2) - sqrtf(a1 * a1 + b1 * b1);
    float dH = sqrtf((a2 - a1) * (a2 - a1) + (b2 - b1) * (b2 - b1) - dC * dC);

    // Calculate average values
    float L_mean = (L1 + L2) / 2.0f;
    float C1 = sqrtf(a1 * a1 + b1 * b1);
    float C2 = sqrtf(a2 * a2 + b2 * b2);
    float C_mean = (C1 + C2) / 2.0f;
    float h1 = atan2f(b1, a1);
    float h2 = atan2f(b2, a2);
    float H_mean = (h1 + h2) / 2.0f;

    // Calculate weights
    float T = 1.0f - 0.17f * cosf(H_mean - 30.0f) + 0.24f * cosf(2.0f * H_mean) + 0.32f * cosf(3.0f * H_mean + 6.0f) - 0.20f * cosf(4.0f * H_mean - 63.0f);
    float SL = 1.0f + (0.015f * (L_mean - 50.0f) * (L_mean - 50.0f)) / sqrtf(20.0f + (L_mean - 50.0f) * (L_mean - 50.0f));
    float SC = 1.0f + 0.045f * C_mean;
    float SH = 1.0f + 0.015f * C_mean * T;
    float delta_H_prime = h2 - h1;

    // Ensure hue difference is in range [0, 360)
    if (fabsf(delta_H_prime) > 180.0f) {
        delta_H_prime = (delta_H_prime > 0.0f) ? delta_H_prime - 360.0f : delta_H_prime + 360.0f;
    }

    float delta_E = sqrtf(
        (dL / (K_L * SL)) * (dL / (K_L * SL)) +
        (dC / (K_C * SC)) * (dC / (K_C * SC)) +
        (delta_H_prime / (K_H * SH)) * (delta_H_prime / (K_H * SH))
    );

    return delta_E;
}

__device__ float calculate_HSV_diff(const float* gt_color, const float* prim_color) {
    float gt_hsv[3], prim_hsv[3];
    rgb_to_hsv(gt_color, gt_hsv);
    rgb_to_hsv(prim_color, prim_hsv);

    // Calculate hue difference with wrap-around
    float delta_H = fabsf(gt_hsv[0] - prim_hsv[0]);
    delta_H = fminf(delta_H, 360.0f - delta_H); // Use fminf for clarity

    // Calculate saturation and value differences
    float delta_S = fabsf(gt_hsv[1] - prim_hsv[1]);
    float delta_V = fabsf(gt_hsv[2] - prim_hsv[2]);

    // Return total HSV difference
    return delta_H + delta_S + delta_V;
}

__device__ float calculate_LAB_diff(const float* gt_color, const float* prim_color) {
    float gt_lab[3], prim_lab[3];
    rgb_to_lab(gt_color, gt_lab);
    rgb_to_lab(prim_color, prim_lab);
    return ciede2000(gt_lab, prim_lab);
}

// Function to calculate the absolute difference between LUV colors
__device__ float calculate_LUV_diff(const float* gt_color, const float* prim_color) {
    float gt_xyz[3], prim_xyz[3];
    float gt_luv[3], prim_luv[3];
    rgb_to_xyz(gt_color, gt_xyz);
    rgb_to_xyz(prim_color, prim_xyz);
    xyz_to_luv(gt_xyz, gt_luv);
    xyz_to_luv(prim_xyz, prim_luv);
    float diff = fabs(gt_luv[0] - prim_luv[0]) + fabs(gt_luv[1] - prim_luv[1]) + fabs(gt_luv[2] - prim_luv[2]);
    return diff;
}


// Function IDs are defined using bitmasking. For example, `safeguard_gs_score_function=0x24`, which is SafeguardGS' choice, outputs `L1_color_error * alpha * transmittance`.
// First 2 bytes:
//   0x00. score = 1
//   0x01. score = opacity
//   0x02. score = alpha
//   0x03. score = opacity * transmittance
//   0x04. score = alpha * transmittance
//   0x05. score = dist error
//   0x06. score = dist error * opacity
//   0x07. score = dist error * alpha
//   0x08. score = dist error * opacity * transmittance
//   0x09. score = dist error * alpha * transmittance
// Last 2 bytes:
//   0x10. score = color error (Cosine similarity)
//   0x20. score = color error (Manhattan distance)
//   0x30. score = exp color error (Manhattan distance)
// Output: score of [0, 1] for a Gaussian primitive with respect to a ray
template<uint32_t C>
__device__ float compute_score(
	int func_id,
	float opacity,
	float alpha,
	float T,
	float2* pix_dist,
	double p_dist_activation_coef = 1.0,
	double c_dist_activation_coef = 1.0,
	float* gt_color = nullptr,
	float* prim_color = nullptr)
{
	float activated_dist_err = 0.0f;
	float activated_color_dist_err = 0.0f;
	float color_cos_sim = 0.0f;
	float color_dist_err = 0.0f;
	float score = 0.0f;
    float ssim = 0.0f;
    float abs_ms_ssim = 0.0f;
    float color_diff = 0.0f;

	switch (func_id & 0x0f)
	{
		case 0x00:
			score = 1; break;
		case 0x01:
			score = opacity; break;
		case 0x02:
			score = alpha; break;
		case 0x03:
			score = opacity * T; break;
		case 0x04:
			score = alpha * T; break;
		case 0x05:
			activated_dist_err = exp(-1.0f * (p_dist_activation_coef * sqrt(pix_dist->x * pix_dist->x + pix_dist->y * pix_dist->y)));
			score = activated_dist_err; break;
		case 0x06:
			activated_dist_err = exp(-1.0f * (p_dist_activation_coef * sqrt(pix_dist->x * pix_dist->x + pix_dist->y * pix_dist->y)));
			score = activated_dist_err * opacity; break;
		case 0x07:
			activated_dist_err = exp(-1.0f * (p_dist_activation_coef * sqrt(pix_dist->x * pix_dist->x + pix_dist->y * pix_dist->y)));
			score = activated_dist_err * alpha; break;
		case 0x08:
			activated_dist_err = exp(-1.0f * (p_dist_activation_coef * sqrt(pix_dist->x * pix_dist->x + pix_dist->y * pix_dist->y)));
			score = activated_dist_err * opacity * T; break;
		case 0x09:
			activated_dist_err = exp(-1.0f * (p_dist_activation_coef * sqrt(pix_dist->x * pix_dist->x + pix_dist->y * pix_dist->y)));
			score = activated_dist_err * alpha * T; break;
	}

	switch (func_id & 0xf0)
	{
		case 0x00:
			return score;
		case 0x10:
			for (int ch = 0; ch < C; ch++)
				color_cos_sim += gt_color[ch] * prim_color[ch];
			return score * color_cos_sim;
		case 0x20:
			for (int ch = 0; ch < C; ch++)
				color_dist_err += abs(gt_color[ch] - prim_color[ch]);
			activated_color_dist_err = 1 - color_dist_err / C;
			return score * activated_color_dist_err;
		case 0x30:
			for (int ch = 0; ch < C; ch++)
				color_dist_err += abs(gt_color[ch] - prim_color[ch]);
			activated_color_dist_err = exp(-1.0f * c_dist_activation_coef * color_dist_err / C);
			return score * activated_color_dist_err;
        //Added function    
        case 0x40:
            float ssim = 1-calculate_ssim<C>(gt_color, prim_color);
            return score * ssim;
        case 0x50:
            float abs_ms_ssim = abs(calculate_ms_ssim<C>(gt_color, prim_color, 256, 256));
            return score * abs_ms_ssim;
        //NEW COLOR MODEL BASED DIFF
        case 0x60:
			color_diff = calculate_LAB_diff(gt_color,prim_color);
			//return score / (1.0f + logf(1.0f + color_diff));
            return score * exp(-1.0f * color_diff);
        case 0x70:
			color_diff = calculate_LUV_diff(gt_color,prim_color);
			return score * exp(-1.0f * color_diff);
        case 0x80:
            color_diff = calculate_HSV_diff(gt_color,prim_color);
			return score * exp(-1.0f * color_diff);
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_topk(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const int score_function,
	const float p_dist_activation_coef,
	const float c_dist_activation_coef,
	uint64_t* __restrict__ gaussian_keys_unsorted,
	uint32_t* __restrict__ gaussian_values_unsorted)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			uint64_t block_id = block.group_index().y * horizontal_blocks + block.group_index().x;
			// Use value range of [0, 65535].
			// To be valid for ascending order sorting, use 1 - score instead of score.
			float score = compute_score<CHANNELS>(score_function, con_o.w, alpha, T, &d, p_dist_activation_coef, c_dist_activation_coef);
			uint32_t scaled_score = __float2uint_rd((1 - score) * 65535);
			gaussian_keys_unsorted[range.x + progress] = (block_id << 32) | scaled_score;
			gaussian_values_unsorted[range.x + progress] = collected_id[j];
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::topk_gaussian(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	const int score_function,
	const float p_dist_activation_coef,
	const float c_dist_activation_coef,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	float* out_color)
{
	renderCUDA_topk<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		score_function,
		p_dist_activation_coef,
		c_dist_activation_coef,
		gaussian_keys_unsorted,
		gaussian_values_unsorted);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_topk_color(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	const float* __restrict__ image_gt,
	float* __restrict__ out_color,
	const int score_function,
	const float p_dist_activation_coef,
	const float c_dist_activation_coef,
	uint64_t* __restrict__ gaussian_keys_unsorted,
	uint32_t* __restrict__ gaussian_values_unsorted)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float gt_color[CHANNELS] = { 0 };
	float gt_color_len = 0.0f;
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		gt_color[ch] = image_gt[ch * H * W + pix_id];
		gt_color_len += gt_color[ch] * gt_color[ch];
	}
	gt_color_len = sqrt(gt_color_len);
	if (gt_color_len > 0)
		for (int ch = 0; ch < CHANNELS; ch++)
			gt_color[ch] /= gt_color_len;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			float prim_color[CHANNELS] = { 0 };
			float prim_color_len = 0.0f;
			for (int ch = 0; ch < CHANNELS; ch++)
			{
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

				prim_color[ch] = features[collected_id[j] * CHANNELS + ch];
				prim_color_len += prim_color[ch] * prim_color[ch];
			}
			prim_color_len = sqrt(prim_color_len);
			if (prim_color_len > 0)
				for (int ch = 0; ch < CHANNELS; ch++)
					prim_color[ch] /= prim_color_len;

			uint64_t block_id = block.group_index().y * horizontal_blocks + block.group_index().x;
			// Use value range of [0, 65535].
			// To be valid for ascending order sorting, use 1 - score instead of score.
			float score = compute_score<CHANNELS>(score_function, con_o.w, alpha, T, &d, p_dist_activation_coef, c_dist_activation_coef, gt_color, prim_color);
			uint32_t scaled_score = __float2uint_rd((1 - score) * 65535);
			gaussian_keys_unsorted[range.x + progress] = (block_id << 32) | scaled_score;
			gaussian_values_unsorted[range.x + progress] = collected_id[j];
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::topk_color_gaussian(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	const float* image_gt,
	const int score_function,
	const float p_dist_activation_coef,
	const float c_dist_activation_coef,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	float* out_color)
{
	renderCUDA_topk_color<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		image_gt,
		out_color,
		score_function,
		p_dist_activation_coef,
		c_dist_activation_coef,
		gaussian_keys_unsorted,
		gaussian_values_unsorted);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_topk_weight(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	uint64_t* __restrict__ gaussian_keys_unsorted,
	uint32_t* __restrict__ gaussian_values_unsorted)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			uint64_t block_id = block.group_index().y * horizontal_blocks + block.group_index().x;
			// Use value range of [0, 65535].
			// To be valid for ascending order sorting, use 1 - weight instead of weight.
			uint32_t scaled_dT = __float2uint_rd((1 - (alpha * T)) * 65535);
			gaussian_keys_unsorted[range.x + progress] = (block_id << 32) | scaled_dT;
			gaussian_values_unsorted[range.x + progress] = collected_id[j];
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::topk_weight_gaussian(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	float* out_color)
{
	renderCUDA_topk_weight<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		gaussian_keys_unsorted,
		gaussian_values_unsorted);
}
