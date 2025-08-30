#include "register_types.h"

#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "scene/resources/compositor.h"
#include "diffusion_effect.h"

void initialize_diffusion_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	ClassDB::register_class<DiffusionEffect>();

	ProjectSettings::get_singleton()->set_custom_property_info("diffusion/enabled", PropertyInfo(Variant::BOOL));
	if (!ProjectSettings::get_singleton()->has_setting("diffusion/enabled")) {
		ProjectSettings::get_singleton()->set_setting("diffusion/enabled", false);
	}
}

void uninitialize_diffusion_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}


