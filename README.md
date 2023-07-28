

# TDDPy2 Documentation

## Project Structures (for Developers)


### Set up the project

This project depends on these libraries: __xtensor__, __Boost__ and __Pybind11__.

Follow these steps to set up the probject. We will demonstrate using __VS Code__.

- Install the package manager __Conda__, and create a virtual environment using:
  ```
  conda create -n tddpy python=3.11
  ```

  Then activate the environment using:
  ```
  conda activate tddpy
  ```
- Install the dependencies using:
  ```
  conda install -c conda-forge boost
  conda install -c conda-forge xtensor-python
  ```
- Clone the GitHub repository from https://github.com/TddPy/TDDPy2 and open the folder using __VS Code__. The settings in .vscode should apply automatically.

- Open _src/CMakePresets.json_ and replace the value __TDDPY_CONDA_PREFIX__ with the path of the virtual environment _tddpy_. Typically it will be something like
```
.../Anaconda3/envs/tddpy
```

- Try buiding the project.

### CMake Presets

See __CMakePresets.json__ for cmake presets of configure and build. Here is the introduction to their purposes.

- __ctdd Backend Test__ : Compile the ctdd project as an executable in the debug model. Intended for testing within C++ codes.
- __TDDPy build__ : Build the TDDPy project.







## Other Information

关于配置预设和构建预设

关于cmake
https://zhuanlan.zhihu.com/p/500002865

在 Visual Studio 中尝试Python/C++联合调试

关于vscode+cmake配置文件
https://zhuanlan.zhihu.com/p/370211322

关于在pip上直接从源代码构建 C++/Python 项目的范例。这难度很高，但允许我们实现从在用户端从源代码构建。
https://github.com/pybind/cmake_example/tree/master