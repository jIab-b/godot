extends CharacterBody3D

@export var speed: float = 1.5  # Reduced speed by 3x (was 5.0)
@export var mouse_sensitivity: float = 0.05  # Reduced sensitivity (default was 0.2)

var camera_angle: float = 0.0

func _ready():
	# Set the camera as current
	var camera = $Camera3D
	if camera:
		camera.make_current()
	
	# Hide and capture mouse
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

func _input(event):
	if event is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		# Rotate player horizontally
		rotation.y -= event.relative.x * mouse_sensitivity * 0.01
		
		# Rotate camera vertically
		camera_angle -= event.relative.y * mouse_sensitivity * 0.01
		# Clamp camera angle to prevent over-rotation
		camera_angle = clamp(camera_angle, -1.5, 1.5)  # ~85 degrees
		
		var camera = $Camera3D
		if camera:
			camera.rotation.x = camera_angle
	# Quit on ESC
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_ESCAPE:
		get_tree().quit()

func _physics_process(delta):
	# Handle movement input relative to camera view direction
	var direction = Vector3.ZERO
	
	# Get the camera
	var camera = $Camera3D
	
	# Create movement vectors based on camera orientation
	var forward = Vector3.ZERO
	var right = Vector3.ZERO
	
	if camera:
		# In Godot, the camera's basis.z points backward (negative Z), so we negate it for forward
		# Create horizontal forward vector (flatten to XZ plane)
		forward = Vector3(-camera.global_transform.basis.z.x, 0, -camera.global_transform.basis.z.z).normalized()
		
		# Create horizontal right vector (flatten to XZ plane)
		right = Vector3(camera.global_transform.basis.x.x, 0, camera.global_transform.basis.x.z).normalized()
	
	# WASD movement (relative to camera view direction)
	if Input.is_action_pressed("ui_right"):
		# Move right relative to camera view
		direction += right
	if Input.is_action_pressed("ui_left"):
		# Move left relative to camera view
		direction -= right
	if Input.is_action_pressed("ui_up"):
		# Move forward relative to camera view
		direction += forward
	if Input.is_action_pressed("ui_down"):
		# Move backward relative to camera view
		direction -= forward
	
	# Up/down movement (world space)
	if Input.is_action_pressed("ui_accept"):  # Spacebar
		direction.y += 1
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
		direction.y -= 1
	
	# Normalize (keep magnitude 0..1)
	if direction != Vector3.ZERO:
		direction = direction.normalized()
	
	# Apply movement using physics
	velocity.x = direction.x * speed
	velocity.z = direction.z * speed
	velocity.y = direction.y * speed
	move_and_slide()
