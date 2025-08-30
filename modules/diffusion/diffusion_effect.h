#pragma once

#include "scene/resources/compositor.h"

class DiffusionEffect : public CompositorEffect {
	GDCLASS(DiffusionEffect, CompositorEffect);

protected:
	static void _bind_methods();

public:
	DiffusionEffect();

	void _render_callback(int p_effect_callback_type, const RenderData *p_render_data);
};


