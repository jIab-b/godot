#pragma once

#include "scene/resources/compositor.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/rendering_device.h"

class DiffusionEffect : public CompositorEffect {
	GDCLASS(DiffusionEffect, CompositorEffect);

protected:
	static void _bind_methods();

public:
    DiffusionEffect();

    // Render callback invoked by the RenderingServer via CompositorEffect
    void _render_callback(int p_effect_callback_type, const RenderData *p_render_data);

	// Method to manually trigger diffusion generation from GDScript
	void trigger_diffusion();

private:
    void _run_lightning_with_depth(const RID &p_depth, const Ref<RenderSceneBuffersRD> &p_rb, int p_width, int p_height, float p_z_near, float p_z_far);
    bool should_trigger_diffusion = false;

    // GPU readback path (module-only) to produce a CPU-readable grayscale depth image.
    // This allows us to avoid engine changes, and later we can swap to a GPU-only backend.
    RID depth_copy_shader;
    RID depth_copy_pipeline;
    RID depth_copy_sampler;
    RID depth_copy_target;      // R8 or RGBA8 color texture with CAN_COPY_FROM_BIT
    RID depth_copy_framebuffer; // FB wrapping the target
    Size2i depth_copy_target_size = Size2i(0, 0);

    // Ensures shader/pipeline/sampler/target exist and match size.
    void _ensure_depth_copy_setup(int p_width, int p_height);
    // Render a full-screen pass that samples p_depth and writes linearized grayscale to target, then read back to out_gray.
    bool _blit_depth_to_cpu_gray(const RID &p_depth, int p_width, int p_height, float p_z_near, float p_z_far, PackedByteArray &r_gray_bytes);
};
