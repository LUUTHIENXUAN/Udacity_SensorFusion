# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.14.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.14.4/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/luuthienxuan/Downloads/cluster_with_roi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/luuthienxuan/Downloads/cluster_with_roi/build

# Include any dependencies generated for this target.
include CMakeFiles/cluster_with_roi.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cluster_with_roi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cluster_with_roi.dir/flags.make

CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.o: CMakeFiles/cluster_with_roi.dir/flags.make
CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.o: ../src/cluster_with_roi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/luuthienxuan/Downloads/cluster_with_roi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.o -c /Users/luuthienxuan/Downloads/cluster_with_roi/src/cluster_with_roi.cpp

CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/luuthienxuan/Downloads/cluster_with_roi/src/cluster_with_roi.cpp > CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.i

CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/luuthienxuan/Downloads/cluster_with_roi/src/cluster_with_roi.cpp -o CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.s

CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.o: CMakeFiles/cluster_with_roi.dir/flags.make
CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.o: ../src/structIO.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/luuthienxuan/Downloads/cluster_with_roi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.o -c /Users/luuthienxuan/Downloads/cluster_with_roi/src/structIO.cpp

CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/luuthienxuan/Downloads/cluster_with_roi/src/structIO.cpp > CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.i

CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/luuthienxuan/Downloads/cluster_with_roi/src/structIO.cpp -o CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.s

# Object files for target cluster_with_roi
cluster_with_roi_OBJECTS = \
"CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.o" \
"CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.o"

# External object files for target cluster_with_roi
cluster_with_roi_EXTERNAL_OBJECTS =

cluster_with_roi: CMakeFiles/cluster_with_roi.dir/src/cluster_with_roi.cpp.o
cluster_with_roi: CMakeFiles/cluster_with_roi.dir/src/structIO.cpp.o
cluster_with_roi: CMakeFiles/cluster_with_roi.dir/build.make
cluster_with_roi: /usr/local/lib/libopencv_gapi.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_stitching.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_aruco.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_bgsegm.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_bioinspired.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_ccalib.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_dnn_objdetect.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_dpm.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_face.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_freetype.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_fuzzy.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_hfs.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_img_hash.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_line_descriptor.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_quality.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_reg.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_rgbd.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_saliency.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_sfm.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_stereo.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_structured_light.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_superres.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_surface_matching.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_tracking.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_videostab.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_xfeatures2d.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_xobjdetect.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_xphoto.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_shape.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_datasets.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_plot.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_text.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_dnn.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_ml.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_phase_unwrapping.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_optflow.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_ximgproc.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_video.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_objdetect.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_calib3d.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_features2d.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_flann.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_highgui.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_videoio.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_imgcodecs.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_photo.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_imgproc.4.1.0.dylib
cluster_with_roi: /usr/local/lib/libopencv_core.4.1.0.dylib
cluster_with_roi: CMakeFiles/cluster_with_roi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/luuthienxuan/Downloads/cluster_with_roi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable cluster_with_roi"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cluster_with_roi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cluster_with_roi.dir/build: cluster_with_roi

.PHONY : CMakeFiles/cluster_with_roi.dir/build

CMakeFiles/cluster_with_roi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cluster_with_roi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cluster_with_roi.dir/clean

CMakeFiles/cluster_with_roi.dir/depend:
	cd /Users/luuthienxuan/Downloads/cluster_with_roi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/luuthienxuan/Downloads/cluster_with_roi /Users/luuthienxuan/Downloads/cluster_with_roi /Users/luuthienxuan/Downloads/cluster_with_roi/build /Users/luuthienxuan/Downloads/cluster_with_roi/build /Users/luuthienxuan/Downloads/cluster_with_roi/build/CMakeFiles/cluster_with_roi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cluster_with_roi.dir/depend

