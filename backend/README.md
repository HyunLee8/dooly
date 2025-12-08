Backend/
├── src/
│   ├── audio_feedback/
│   │   └── stt.py                          ✅ HAVE
│   │
│   ├── computer_vision/
│   │   ├── __init__.py                     ✅ HAVE
│   │   ├── camera_interface.py             ✅ HAVE
│   │   └── object_detection.py (YOLO)      ✅ HAVE
│   │
│   ├── drone_control/
│   │   ├── __init__.py                     ✅ HAVE
│   │   │
│   │   ├── slam/                           ✅ HAVE
│   │   │   ├── __init__.py
│   │   │   ├── orbslam3_wrapper.py
│   │   │   ├── slam_integration.py
│   │   │   ├── drone_slam_demo.py
│   │   │   └── TelloVIO.yaml
│   │   │
│   │   ├── state_estimation/               ✅ HAVE
│   │   │   ├── __init__.py
│   │   │   └── imu_slam_fusion.py
│   │   │
│   │   ├── tello/                          ✅ HAVE
│   │   │   ├── __init__.py
│   │   │   ├── tello_interface.py
│   │   │   └── tello_controller.py
│   │   │
│   │   ├── localization/                   ❌ NEED (NEW!)
│   │   │   ├── __init__.py
│   │   │   ├── aruco_detector.py           # Detect markers in frame
│   │   │   ├── marker_map.py               # Store marker positions
│   │   │   ├── relative_localizer.py       # Calculate position from markers
│   │   │   └── pose_estimator.py           # Combine SLAM + ArUco
│   │   │
│   │   ├── control/                        ❌ NEED
│   │   │   ├── __init__.py
│   │   │   ├── pid_controller.py           # PID loops for X, Y, Z
│   │   │   ├── position_controller.py      # High-level position control
│   │   │   └── velocity_controller.py      # Velocity command interface
│   │   │
│   │   └── planning/                       ❌ NEED
│   │       ├── __init__.py
│   │       ├── waypoint_manager.py         # Queue of positions to visit
│   │       ├── path_planner.py             # Plan paths between waypoints
│   │       └── trajectory_generator.py     # Smooth trajectories
│   │
│   ├── llm_reasoning/                      ✅ HAVE (assuming)
│   │   ├── __init__.py
│   │   ├── llm_interface.py                # LLM API calls
│   │   └── prompt_templates.py             # Prompts for reasoning
│   │
│   ├── exploration/                        ❌ NEED (NEW!)
│   │   ├── __init__.py
│   │   ├── occupancy_grid.py               # Track explored areas
│   │   ├── exploration_manager.py          # Decide where to search
│   │   ├── coverage_tracker.py             # Monitor search progress
│   │   └── marker_based_grid.py            # Grid relative to ArUco markers
│   │
│   ├── mapping/                            ❌ NEED (NEW!)
│   │   ├── __init__.py
│   │   ├── semantic_map.py                 # Map with object labels
│   │   ├── object_database.py              # Store detected objects
│   │   ├── marker_registration.py          # Register ArUco marker locations
│   │   └── spatial_memory.py               # Remember where things are
│   │
│   ├── mission/                            ❌ NEED (NEW!)
│   │   ├── __init__.py
│   │   ├── search_coordinator.py           # Main search orchestration
│   │   ├── task_executor.py                # Execute LLM decisions
│   │   ├── state_machine.py                # Flight state management
│   │   └── safety_monitor.py               # Battery, boundaries, failures
│   │
│   └── integration/                        ❌ NEED (NEW!)
│       ├── __init__.py
│       ├── dooly_brain.py                  # Main coordinator - ties everything together
│       ├── perception_fusion.py            # Combine CV + SLAM + ArUco
│       └── action_bridge.py                # LLM decisions → drone commands
│
├── config/                                 ❌ NEED (NEW!)
│   ├── aruco_markers.yaml                  # Marker IDs and positions
│   ├── room_layout.yaml                    # Known room structure
│   ├── search_params.yaml                  # Search strategy parameters
│   └── safety_limits.yaml                  # Flight boundaries, battery limits
│
├── tests/
│   └── (test files)
│
├── main.py                                 ❌ NEED - Main entry point
└── requirements.txt