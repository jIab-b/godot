extends Node3D

@onready var camera = $Camera3D
@onready var trigger_button = $UI/TriggerButton

var diffusion_effect: DiffusionEffect

func _ready():
	# Get the diffusion effect from the compositor
	var compositor = camera.compositor
	if compositor:
		for effect in compositor.compositor_effects:
			if effect is DiffusionEffect:
				diffusion_effect = effect
				break
	
	if not diffusion_effect:
		print("Warning: No DiffusionEffect found in compositor")
	
	# Connect the button
	trigger_button.pressed.connect(_on_trigger_button_pressed)
	
	# Print setup status
	print("Diffusion Test Scene loaded")
	print("Diffusion enabled: ", ProjectSettings.get_setting("diffusion/enabled", false))

func _on_trigger_button_pressed():
	if diffusion_effect:
		diffusion_effect.trigger_diffusion()
		print("Manually triggered diffusion generation")
	else:
		print("Error: No diffusion effect available")

func _input(event):
	# Show when the action is pressed
	if event.is_action_pressed("diffusion_generate"):
		print("Diffusion action triggered via input!")
