Phase 1: Camera Calibration and Camera Localization with Apriltag

Camera Calibration
- Standard camera calibration procedure for webcam
- Details included in calibration.ipynb

Camera Localization with Apriltag
- Test the accuracy and precision of Apriltag detection
- Procedure
  - Select a global coordinate system
  - Measure camera and Apriltag pose in the global coordinate system
  - Compute camera pose in global coordinate system using Apriltag detection results and measured Apriltag pose in global coordinate system
  - Compare computed value and measured value of camera pose
