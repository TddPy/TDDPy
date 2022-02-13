
C++ dependencies: Boost, LibTorch, Python
Note for Visual Studio development:

Configurations should be reset first, especially the include path and .lib files.

1. An extra environment variable 'PythonPath' should be set to the python interpreter location. Restart VS and try again.
2. An extra environment variable 'Boost' should be set to the C++ Boost library location.
3. An extra environment variable 'libtorch' should be set to the C++ LibTorch library location.