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
    print("Player: Ready - Camera set as current, mouse captured")
    print("Player: Testing diffusion effect integration")
    
    # Test if we can access the diffusion module
    if ProjectSettings.has_setting("diffusion/enabled"):
        print("Player: Diffusion module setting found: ", ProjectSettings.get_setting("diffusion/enabled"))
    else:
        print("Player: WARNING - Diffusion module setting not found")

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
        print("Player: ESC pressed - quitting")
        get_tree().quit()
    
    # Diffusion generation key (4 key)
    if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_4:
        print("\n=== DIFFUSION GENERATION TRIGGERED (4 key) ===")
        print("Player: 4 key pressed - triggering diffusion generation")
        print("Player: Checking diffusion module availability...")
    
        # Check if DiffusionEffect class exists
        var test_instance = DiffusionEffect.new()
        if test_instance:
            print("Player: DiffusionEffect class available")
            
            # Check if method exists
            if test_instance.has_method("trigger_diffusion"):
                print("Player: trigger_diffusion method found")
                print("Player: Calling diffusion generation...")
                test_instance.trigger_diffusion()
                print("Player: Diffusion trigger completed - check for C++ logs")
            else:
                print("Player: trigger_diffusion method missing")
            
        else:
            print("Player: DiffusionEffect class not available")
    
        # Check project settings
        if ProjectSettings.has_setting("diffusion/enabled"):
            var enabled = ProjectSettings.get_setting("diffusion/enabled")
            if enabled:
                print("Player: Diffusion enabled in project settings")
            else:
                print("Player: Diffusion disabled in project settings")
        else:
            print("Player: diffusion/enabled setting not found")
    
        print("Player: Diffusion key4 test complete")
    
    # Manual diffusion trigger on 'T' key
    if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_T:
        print("Player: Manual diffusion trigger (T key) - testing C++ module integration")
        _trigger_diffusion_comprehensive_test()

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
        direction += right
    if Input.is_action_pressed("ui_left"):
        direction -= right
    if Input.is_action_pressed("ui_up"):
        direction += forward
    if Input.is_action_pressed("ui_down"):
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

func _trigger_diffusion_comprehensive_test():
    print("\n=== COMPREHENSIVE DIFFUSION MODULE DEBUG TEST ===")
    
    # Test 1: Verify Godot scene setup
    print("\n--- GODOT SCENE CONFIGURATION ---")
    var camera = $Camera3D
    if camera:
        print("✓ Camera found: ", camera.name)
        print("  - Camera position: ", camera.global_position)
        print("  - Camera transform: ", camera.global_transform)
        
        # Check viewport
        var viewport = camera.get_viewport()
        if viewport:
            print("✓ Viewport found")
            print("  - Viewport size: ", viewport.size)
            print("  - Viewport transparent: ", viewport.transparent_bg)
        else:
            print("✗ Viewport not found")
    else:
        print("✗ Camera not found")
        return
    
    # Test 2: Check diffusion module registration
    print("\n--- DIFFUSION MODULE REGISTRATION ---")
    print("  - Attempting to create DiffusionEffect instance...")
    var test_instance = DiffusionEffect.new()
    if test_instance:
        print("✓ DiffusionEffect class is registered and accessible")
        
        if test_instance.has_method("trigger_diffusion"):
            print("✓ trigger_diffusion method found")
        else:
            print("✗ trigger_diffusion method missing")
    else:
        print("✗ DiffusionEffect class NOT registered or accessible")
        return
    
    # Test 3: Check project settings
    print("\n--- PROJECT SETTINGS ---")
    if ProjectSettings.has_setting("diffusion/enabled"):
        var enabled = ProjectSettings.get_setting("diffusion/enabled")
        print("✓ diffusion/enabled setting: ", enabled)
        if not enabled:
            print("⚠ WARNING: Diffusion is disabled in project settings")
    else:
        print("✗ diffusion/enabled setting not found")
    
    # Test 4: Check input actions
    print("\n--- INPUT CONFIGURATION ---")
    if InputMap.has_action("diffusion_generate"):
        print("✓ diffusion_generate input action exists")
        var events = InputMap.action_get_events("diffusion_generate")
        print("  - Bound to ", events.size(), " input events")
        for event in events:
            print("    * ", event.as_text())
    else:
        print("✗ diffusion_generate input action missing")
    
    # Test 5: Attempt to create and test DiffusionEffect instance
    print("\n--- DIFFUSION EFFECT INSTANCE TEST ---")
    var diffusion_instance = DiffusionEffect.new()
    if diffusion_instance:
        print("✓ DiffusionEffect instance created")
        
        # Check if it's a CompositorEffect
        if diffusion_instance is CompositorEffect:
            print("✓ Instance is CompositorEffect")
        else:
            print("✗ Instance is NOT CompositorEffect")
        
        # Test method calls
        if diffusion_instance.has_method("trigger_diffusion"):
            print("✓ trigger_diffusion method available, calling...")
            diffusion_instance.trigger_diffusion()
            print("✓ Method called successfully - check console for C++ logs")
        else:
            print("✗ trigger_diffusion method missing")
        
        # Clean up - NO MANUAL FREE NEEDED for RefCounted objects
        print("✓ Instance managed by Godot's garbage collector")
    else:
        print("✗ Could not create DiffusionEffect instance")
    
    # Test 6: Check rendering context
    print("\n--- RENDERING CONTEXT ---")
    var rendering_server = RenderingServer.get_rendering_device()
    if rendering_server:
        print("✓ Rendering device available")
    else:
        print("✗ No rendering device")
    
    print("\n=== DEBUG TEST COMPLETE ===")
    print("Check console output above for any failures marked with ✗")
    print("If all checks pass, the diffusion module should work correctly")
