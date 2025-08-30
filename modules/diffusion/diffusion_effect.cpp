#include "diffusion_effect.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/object/class_db.h"
#include "core/io/image.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"

static bool g_diffusion_capture_requested = false;

void DiffusionEffect::_bind_methods() {
}

DiffusionEffect::DiffusionEffect() {
	set_effect_callback_type(CompositorEffect::EFFECT_CALLBACK_TYPE_POST_OPAQUE);
	set_access_resolved_color(true);
	set_access_resolved_depth(true);

	String method = OS::get_singleton()->get_current_rendering_method();
	bool is_forward_plus = (method == "forward_plus");
	if (is_forward_plus) {
		set_needs_motion_vectors(true);
		set_needs_normal_roughness(true);
	}
}

void DiffusionEffect::_render_callback(int p_effect_callback_type, const RenderData *p_render_data,
    RID &p_color, RID &p_depth, RID &p_motion, RID &p_normal_roughness) {
	if (!ProjectSettings::get_singleton()->get_setting("diffusion/enabled", false).operator bool()) {
		return;
	}

	const RenderDataRD *rd = static_cast<const RenderDataRD *>(p_render_data);
	if (!rd) { return; }

	Ref<RenderSceneBuffersRD> rb = rd->render_buffers; 
    if (rb.is_null()) { return; }


    // fetch color + depth, if forward+, fetch motion + normal as well
    String method = OS::get_singleton()->get_current_rendering_method();
	bool is_forward_plus = (method == "forward_plus");

	RID color = rb->get_back_buffer_texture();
	RID depth = rb->get_depth_texture();

	RID motion = RID();
	RID normal_roughness = RID();
	if (is_forward_plus) {
		motion = rb->get_velocity_buffer(false);
		if (rb->has_texture(SNAME("forward_clustered"), SNAME("normal_roughness"))) {
			normal_roughness = rb->get_texture(SNAME("forward_clustered"), SNAME("normal_roughness"));
		}
	}

    p_color = color;
    p_depth = depth;
    p_motion = motion;
    p_normal_roughness = normal_roughness;


    /// camera stuff (can be packaged for downstream use)
	Transform3D cam_xf = rd->scene_data->cam_transform;
	Projection cam_proj = rd->scene_data->cam_projection;
	float z_near = rd->scene_data->z_near;
	float z_far = rd->scene_data->z_far;
	Size2i size = rb->get_internal_size();

	// On-demand depth capture and external pipeline call
	if (g_diffusion_capture_requested) {
		g_diffusion_capture_requested = false;
		_run_lightning_with_depth(p_depth, size.x, size.y, z_near, z_far);
	}
}



void DiffusionEffect::_run_lightning_with_depth(const RID &p_depth, int p_width, int p_height, float p_z_near, float p_z_far) {
	if (!p_depth.is_valid() || p_width <= 0 || p_height <= 0) return;
	PackedByteArray raw = RD::get_singleton()->texture_get_data(p_depth, 0);
	if (raw.size() >= p_width * p_height * (int)sizeof(float)) {
		const float *src = reinterpret_cast<const float *>(raw.ptr());
		PackedByteArray gray;
		gray.resize(p_width * p_height);
		uint8_t *dst = gray.ptrw();
		for (int i = 0; i < p_width * p_height; i++) {
			float z_ndc = src[i];
			float denom = (p_z_far + p_z_near - z_ndc * (p_z_far - p_z_near));
			float z_lin = (2.0f * p_z_near * p_z_far) / (denom == 0.0f ? 1e-6f : denom);
			float nz = (z_lin - p_z_near) / ((p_z_far - p_z_near) == 0.0f ? 1e-6f : (p_z_far - p_z_near));
			nz = CLAMP(nz, 0.0f, 1.0f);
			dst[i] = (uint8_t)(nz * 255.0f);
		}
		Ref<Image> img; img.instantiate();
		img->create(p_width, p_height, false, Image::FORMAT_L8, gray);
		String out_path = ProjectSettings::get_singleton()->globalize_path("res://diff_stuff/out_depth.png");
		img->save_png(out_path);
		String script_path = ProjectSettings::get_singleton()->globalize_path("res://diff_stuff/lightning_gen.py");
		Vector<String> args;
		String os_name = OS::get_singleton()->get_name();
		if (os_name == "Windows") {
			String diff_dir = ProjectSettings::get_singleton()->globalize_path("res://diff_stuff");
			String cmd = String("cd /d \"") + diff_dir + String("\" && python \"") + script_path.get_file() + String("\"");
			args.push_back("/C");
			args.push_back(cmd);
			OS::get_singleton()->execute("cmd", args, nullptr, nullptr, false, false);
		} else {
			String diff_dir = ProjectSettings::get_singleton()->globalize_path("res://diff_stuff");
			String sh = String("cd \"") + diff_dir + String("\" && python3 \"") + script_path.get_file() + String("\"");
			args.push_back("-lc");
			args.push_back(sh);
			OS::get_singleton()->execute("/bin/bash", args, nullptr, nullptr, false, false);
		}
	}
}