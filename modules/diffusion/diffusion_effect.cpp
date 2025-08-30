#include "diffusion_effect.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/object/class_db.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"

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


    /// camera stuff, unused rn
	Transform3D cam_xf = rd->scene_data->cam_transform;
	Projection cam_proj = rd->scene_data->cam_projection;
	float z_near = rd->scene_data->z_near;
	float z_far = rd->scene_data->z_far;
	Size2i size = rb->get_internal_size();


}


