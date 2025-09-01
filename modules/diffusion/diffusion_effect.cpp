#include "diffusion_effect.h"

#include "core/config/project_settings.h"
#include "core/config/engine.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/object/class_db.h"
#include "core/io/image.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/rendering_device.h"

static bool g_diffusion_prev_key4_pressed = false;

void DiffusionEffect::_bind_methods() {
    // Bind manual trigger for GDScript
    ClassDB::bind_method(D_METHOD("trigger_diffusion"), &DiffusionEffect::trigger_diffusion);
    // Bind the render callback so the initial Callable(this, "_render_callback") works
    ClassDB::bind_method(D_METHOD("_render_callback", "effect_callback_type", "render_data"), &DiffusionEffect::_render_callback);
}

DiffusionEffect::DiffusionEffect() {
    // Note: Do NOT call set_effect_callback_type() here.
    // CompositorEffect constructor installs a Callable(this, "_render_callback").
    // Calling set_effect_callback_type() would swap it to the GDVIRTUAL path,
    // which C++ modules can't override. Keep default stage and just request resources.
    set_access_resolved_color(true);
    set_access_resolved_depth(true);

    String method = OS::get_singleton()->get_current_rendering_method();
    bool is_forward_plus = (method == "forward_plus");
    if (is_forward_plus) {
        set_needs_motion_vectors(true);
        set_needs_normal_roughness(true);
    }

    print_line("Diffusion module: DiffusionEffect constructed (rendering method: " + method + ")");
}

void DiffusionEffect::_render_callback(int p_effect_callback_type, const RenderData *p_render_data) {
    
    if (!ProjectSettings::get_singleton()->get_setting("diffusion/enabled", false).operator bool()) {
        // Only log once to avoid spam
        static bool logged_disabled = false;
        if (!logged_disabled) {
            print_line("Diffusion module: disabled in project settings. Enable 'diffusion/enabled' to use.");
            logged_disabled = true;
        }
        return;
    }

    // Check for keypress first before doing any heavy processing
    bool in_editor = Engine::get_singleton()->is_editor_hint();
    if (in_editor) { 
        return; 
    }

    // Use action system instead of direct key check for better reliability
    bool should_generate = should_trigger_diffusion; // Check manual trigger first
    if (should_generate) {
        print_line("Diffusion module: Manual trigger flag detected");
    }
    should_trigger_diffusion = false; // Reset manual trigger
    
    if (!should_generate) {
        if (InputMap::get_singleton()->has_action("diffusion_generate")) {
            bool just = Input::get_singleton()->is_action_just_pressed("diffusion_generate");
            if (just) {
                print_line("Diffusion module: Action 'diffusion_generate' just pressed");
            }
            should_generate = just;
        } else {
            // Fallback to direct key check if action doesn't exist
            bool pressed = Input::get_singleton()->is_key_pressed(Key::KEY_4);
            should_generate = pressed && !g_diffusion_prev_key4_pressed;
            g_diffusion_prev_key4_pressed = pressed;
            if (should_generate) {
                print_line("Diffusion module: Fallback KEY_4 just pressed");
            }
        }
    }
    
    if (!should_generate) {
        return;
    }
    
    // Key pressed, execute diffusion logic
    print_line("Diffusion module: Generation triggered! (callback type=" + itos(p_effect_callback_type) + ")");

    const RenderDataRD *rd = static_cast<const RenderDataRD *>(p_render_data);
    if (!rd) { 
        ERR_PRINT("Diffusion module: Invalid render data");
        return; 
    }

    Ref<RenderSceneBuffersRD> rb = rd->render_buffers; 
    if (rb.is_null()) { 
        ERR_PRINT("Diffusion module: Invalid render buffers");
        return; 
    }

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

    // Execute diffusion generation
    Size2i size = rb->get_internal_size();
    float z_near = rd->scene_data->z_near;
    float z_far = rd->scene_data->z_far;
    
    print_line("Diffusion module: Processing depth buffer - Size: " + String::num(size.x) + "x" + String::num(size.y));
    if (rb->has_texture(RB_SCOPE_BUFFERS, RB_TEX_BACK_DEPTH)) {
        print_line("Diffusion module: Back depth buffer present (CPU-readable)");
    } else {
        print_line("Diffusion module: Back depth buffer NOT present; will attempt direct read");
    }
    _run_lightning_with_depth(depth, rb, size.x, size.y, z_near, z_far);
}

void DiffusionEffect::_run_lightning_with_depth(const RID &p_depth, const Ref<RenderSceneBuffersRD> &p_rb, int p_width, int p_height, float p_z_near, float p_z_far) {
	if (!p_depth.is_valid() || p_width <= 0 || p_height <= 0) {
		ERR_PRINT("Diffusion module: Invalid depth buffer or dimensions");
		return;
	}
	
	print_line("Diffusion module: Extracting depth data...");
    // Prefer a CPU-readable back depth texture if available.
    RID cpu_depth = p_depth;
    Ref<RenderSceneBuffersRD> rb = p_rb;
    if (rb.is_valid()) {
        // Try to use the back depth texture that has CAN_COPY_FROM_BIT set.
        if (rb->has_texture(RB_SCOPE_BUFFERS, RB_TEX_BACK_DEPTH)) {
            cpu_depth = rb->get_texture_slice(RB_SCOPE_BUFFERS, RB_TEX_BACK_DEPTH, 0, 0);
            if (cpu_depth.is_valid()) {
                print_line("Diffusion module: Using back depth texture for CPU read");
            }
        }
    }

    // If our chosen texture is not CPU-readable, bail with a helpful message.
    RD::TextureFormat tf;
    if (cpu_depth.is_valid()) {
        tf = RD::get_singleton()->texture_get_format(cpu_depth);
    }
    PackedByteArray gray;
    if (!cpu_depth.is_valid() || !(tf.usage_bits & RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT)) {
        // Fall back to module-only GPU pass that writes grayscale depth to a CPU-readable color texture.
        print_line("Diffusion module: Back depth not CPU-readable. Using module GPU pass to produce CPU-readable grayscale depth.");
        bool ok = _blit_depth_to_cpu_gray(p_depth, p_width, p_height, p_z_near, p_z_far, gray);
        if (!ok) {
            ERR_PRINT("Diffusion module: GPU blit to CPU-readable grayscale failed");
            return;
        }
    } else {
        // Direct CPU readback of float depth, then linearize to 8-bit grayscale on CPU.
        PackedByteArray raw = RD::get_singleton()->texture_get_data(cpu_depth, 0);
        if (raw.size() < p_width * p_height * (int)sizeof(float)) {
            ERR_PRINT("Diffusion module: Insufficient depth data size. Expected: " + String::num(p_width * p_height * sizeof(float)) + ", got: " + String::num(raw.size()));
            return;
        }

        const float *src = reinterpret_cast<const float *>(raw.ptr());
        gray.resize(p_width * p_height);
        uint8_t *dst = gray.ptrw();
        for (int i = 0; i < p_width * p_height; i++) {
            // Raw depth likely in [0,1]; convert to NDC [-1,1]
            float z_ndc = src[i] * 2.0f - 1.0f;
            float denom = (p_z_far + p_z_near - z_ndc * (p_z_far - p_z_near));
            float z_lin = (2.0f * p_z_near * p_z_far) / (denom == 0.0f ? 1e-6f : denom);
            float nz = (z_lin - p_z_near) / ((p_z_far - p_z_near) == 0.0f ? 1e-6f : (p_z_far - p_z_near));
            nz = CLAMP(nz, 0.0f, 1.0f);
            dst[i] = (uint8_t)(nz * 255.0f);
        }
    }

    Ref<Image> img;
    img.instantiate();
    img->set_data(p_width, p_height, false, Image::FORMAT_L8, gray);
	
	// Save depth image to the diff_stuff directory where lightning_gen.py expects it
	String diff_dir = ProjectSettings::get_singleton()->globalize_path("res://diff_stuff");
	
	// If the path doesn't exist in the project directory, try the source directory
	if (!FileAccess::exists(diff_dir.path_join("lightning_gen.py"))) {
		// Try the source directory instead
		String source_diff_dir = "res://../diff_stuff";
		diff_dir = ProjectSettings::get_singleton()->globalize_path(source_diff_dir);
		print_line("Diffusion module: Using source diff_stuff directory: " + diff_dir);
	}
	
	// Ensure the directory exists
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (!da->dir_exists(diff_dir)) {
		Error error = da->make_dir_recursive(diff_dir);
		if (error != OK) {
			ERR_PRINT("Diffusion module: Failed to create diff_stuff directory: " + diff_dir);
			return;
		}
	}
	
	String out_path = diff_dir.path_join("out_depth.png");
	
	Error save_err = img->save_png(out_path);
	if (save_err != OK) {
		ERR_PRINT("Diffusion module: Failed to save depth image to: " + out_path + " (Error: " + String::num(save_err) + ")");
		return;
	}
	print_line("Diffusion module: Depth image saved to: " + out_path);
	
	// Call the existing lightning_gen.py script
	String script_path = diff_dir.path_join("lightning_gen.py");
	if (!FileAccess::exists(script_path)) {
		ERR_PRINT("Diffusion module: Lightning generation script not found at: " + script_path);
		ERR_PRINT("Diffusion module: Please ensure lightning_gen.py exists in your project's diff_stuff folder");
		return;
	}
	
	List<String> args;
	String os_name = OS::get_singleton()->get_name();
	print_line("Diffusion module: Executing lightning generation script on " + os_name + "...");
	
	if (os_name == "Windows") {
		// Use cmd to execute python script in the diff_stuff directory
		String cmd = String("cd /d \"") + diff_dir + String("\" && python \"") + script_path.get_file() + String("\"");
		args.push_back("/C");
		args.push_back(cmd);
		print_line("Diffusion module: Executing command: " + cmd);
		Error exec_err = OS::get_singleton()->execute("cmd", args, nullptr, nullptr, false, nullptr, false);
		if (exec_err != OK) {
			ERR_PRINT("Diffusion module: Failed to execute lightning generation script (Error: " + String::num(exec_err) + ")");
			// Try alternative approach
			List<String> alt_args;
			alt_args.push_back(script_path);
			exec_err = OS::get_singleton()->execute("python", alt_args, nullptr, nullptr, false, nullptr, false);
			if (exec_err != OK) {
				ERR_PRINT("Diffusion module: Alternative execution also failed (Error: " + String::num(exec_err) + ")");
			} else {
				print_line("Diffusion module: Lightning generation script executed with alternative method");
			}
		} else {
			print_line("Diffusion module: Lightning generation script execution started successfully");
		}
	} else {
		// Use bash for Linux/macOS
		String sh = String("cd \"") + diff_dir + String("\" && python3 \"") + script_path.get_file() + String("\"");
		args.push_back("-lc");
		args.push_back(sh);
		print_line("Diffusion module: Executing command: " + sh);
		Error exec_err = OS::get_singleton()->execute("/bin/bash", args, nullptr, nullptr, false, nullptr, false);
		if (exec_err != OK) {
			ERR_PRINT("Diffusion module: Failed to execute lightning generation script (Error: " + String::num(exec_err) + ")");
			// Try alternative approach
			List<String> alt_args;
			alt_args.push_back(script_path);
			exec_err = OS::get_singleton()->execute("python3", alt_args, nullptr, nullptr, false, nullptr, false);
			if (exec_err != OK) {
				ERR_PRINT("Diffusion module: Alternative execution also failed (Error: " + String::num(exec_err) + ")");
			} else {
				print_line("Diffusion module: Lightning generation script executed with alternative method");
			}
		} else {
			print_line("Diffusion module: Lightning generation script execution started successfully");
		}
	}
}

// --- GPU readback helpers ---

static const char *s_depth_copy_vs = R"GLSL(
#version 450
layout(location=0) out vec2 uv;
void main() {
    const vec2 pts[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    uv = pts[gl_VertexIndex] * 0.5 + 0.5;
    gl_Position = vec4(pts[gl_VertexIndex], 0.0, 1.0);
}
)GLSL";

static const char *s_depth_copy_fs = R"GLSL(
#version 450
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;

layout(set=0,binding=0) uniform texture2D depth_tex;
layout(set=0,binding=1) uniform sampler linear_sampler;

layout(push_constant, std430) uniform Params {
    float z_near;
    float z_far;
} params;

void main(){
    float z = texture(sampler2D(depth_tex, linear_sampler), uv).r;
    float z_ndc = z * 2.0 - 1.0;
    float zn = params.z_near; float zf = params.z_far;
    float denom = (zf + zn - z_ndc * (zf - zn));
    float z_lin = (2.0 * zn * zf) / max(denom, 1e-6);
    float nz = clamp((z_lin - zn) / max(zf - zn, 1e-6), 0.0, 1.0);
    color = vec4(nz, nz, nz, 1.0);
}
)GLSL";

void DiffusionEffect::_ensure_depth_copy_setup(int p_width, int p_height) {
    RenderingDevice *rd = RD::get_singleton();
    ERR_FAIL_NULL(rd);

    if (depth_copy_sampler.is_null()) {
        RD::SamplerState ss;
        ss.mag_filter = RD::SAMPLER_FILTER_LINEAR;
        ss.min_filter = RD::SAMPLER_FILTER_LINEAR;
        ss.use_anisotropy = false;
        ss.enable_compare = false;
        ss.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        ss.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        depth_copy_sampler = rd->sampler_create(ss);
    }

    if (depth_copy_shader.is_null()) {
        Vector<RD::ShaderStageSPIRVData> stages;
        {
            RD::ShaderStageSPIRVData sd;
            sd.shader_stage = RD::SHADER_STAGE_VERTEX;
            sd.spirv = rd->shader_compile_spirv_from_source(RD::SHADER_STAGE_VERTEX, s_depth_copy_vs, RD::SHADER_LANGUAGE_GLSL);
            stages.push_back(sd);
        }
        {
            RD::ShaderStageSPIRVData sd;
            sd.shader_stage = RD::SHADER_STAGE_FRAGMENT;
            sd.spirv = rd->shader_compile_spirv_from_source(RD::SHADER_STAGE_FRAGMENT, s_depth_copy_fs, RD::SHADER_LANGUAGE_GLSL);
            stages.push_back(sd);
        }
        depth_copy_shader = rd->shader_create_from_spirv(stages, "diffusion_depth_copy");
    }

    // Recreate target/FB if size changed or not present.
    if (depth_copy_target.is_null() || depth_copy_target_size.x != p_width || depth_copy_target_size.y != p_height) {
        if (!depth_copy_framebuffer.is_null()) {
            rd->free(depth_copy_framebuffer);
            depth_copy_framebuffer = RID();
        }
        if (!depth_copy_target.is_null()) {
            rd->free(depth_copy_target);
            depth_copy_target = RID();
        }
        RD::TextureFormat tf;
        tf.texture_type = RD::TEXTURE_TYPE_2D;
        tf.width = p_width;
        tf.height = p_height;
        tf.depth = 1;
        tf.array_layers = 1;
        tf.mipmaps = 1;
        tf.samples = RD::TEXTURE_SAMPLES_1;
        tf.format = RD::DATA_FORMAT_R8_UNORM; // grayscale
        tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
        depth_copy_target = rd->texture_create(tf, RD::TextureView());
        Vector<RID> c;
        c.push_back(depth_copy_target);
        depth_copy_framebuffer = rd->framebuffer_create(c);
        depth_copy_target_size = Size2i(p_width, p_height);
    }

    if (depth_copy_pipeline.is_null()) {
        RD::PipelineDepthStencilState ds; // depth test disabled
        RD::PipelineColorBlendState bs = RD::PipelineColorBlendState::create_disabled(1);
        depth_copy_pipeline = rd->render_pipeline_create(depth_copy_shader, rd->framebuffer_get_format(depth_copy_framebuffer), RD::INVALID_FORMAT_ID, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), ds, bs, 0);
    }
}

bool DiffusionEffect::_blit_depth_to_cpu_gray(const RID &p_depth, int p_width, int p_height, float p_z_near, float p_z_far, PackedByteArray &r_gray_bytes) {
    RenderingDevice *rd = RD::get_singleton();
    ERR_FAIL_NULL_V(rd, false);

    _ensure_depth_copy_setup(p_width, p_height);

    // Build uniforms: depth texture + sampler
    Vector<RD::Uniform> uniforms;
    {
        RD::Uniform u;
        u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
        u.binding = 0;
        u.append_id(p_depth);
        uniforms.push_back(u);
    }
    {
        RD::Uniform u;
        u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
        u.binding = 1;
        u.append_id(depth_copy_sampler);
        uniforms.push_back(u);
    }
    RID uset = rd->uniform_set_create(uniforms, depth_copy_shader, 0);

    // Draw fullscreen pass
    Vector<Color> clear;
    clear.push_back(Color(0, 0, 0, 1));
    RD::DrawListID dl = rd->draw_list_begin(depth_copy_framebuffer, RD::DRAW_CLEAR_COLOR_0, clear, 1.0f, 0, Rect2());
    rd->draw_list_bind_render_pipeline(dl, depth_copy_pipeline);
    rd->draw_list_bind_uniform_set(dl, uset, 0);
    struct PC { float z_near; float z_far; float pad[2]; } pc;  // Pad to 16 bytes
    pc.z_near = p_z_near;
    pc.z_far = p_z_far;
    pc.pad[0] = 0.0f;
    pc.pad[1] = 0.0f;
    rd->draw_list_set_push_constant(dl, &pc, sizeof(PC));
    rd->draw_list_draw(dl, false, 1, 3);
    rd->draw_list_end();

    // Ensure completion, then read back target
    rd->submit();
    rd->sync();

    rd->free(uset);

    PackedByteArray raw = rd->texture_get_data(depth_copy_target, 0);
    // Expect 1 byte per pixel (R8)
    if ((int)raw.size() < p_width * p_height) {
        ERR_PRINT("Diffusion module: depth copy produced too few bytes");
        return false;
    }
    r_gray_bytes = raw;
    return true;
}

void DiffusionEffect::trigger_diffusion() {
	should_trigger_diffusion = true;
	print_line("Diffusion module: Manual trigger requested");
}
