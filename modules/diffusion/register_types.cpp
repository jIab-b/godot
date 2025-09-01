#include "register_types.h"



#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "core/input/input_map.h"
#include "scene/resources/compositor.h"
#include "diffusion_effect.h"

void initialize_diffusion_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	ClassDB::register_class<DiffusionEffect>();

	// Register project setting with proper global default
	GLOBAL_DEF("diffusion/enabled", false);
	ProjectSettings::get_singleton()->set_custom_property_info(
		PropertyInfo(Variant::BOOL, "diffusion/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

	// Add default input action for diffusion generation if it doesn't exist
	if (!InputMap::get_singleton()->has_action("diffusion_generate")) {
		InputMap::get_singleton()->add_action("diffusion_generate");
		
		Ref<InputEventKey> key_event;
		key_event.instantiate();
		key_event->set_keycode(Key::KEY_4);
		InputMap::get_singleton()->action_add_event("diffusion_generate", key_event);
		
		print_line("Diffusion module: Added default input action 'diffusion_generate' bound to key '4'");
	}
}

void uninitialize_diffusion_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}


