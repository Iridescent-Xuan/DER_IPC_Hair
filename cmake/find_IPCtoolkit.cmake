set(IPC_TOOLKIT_GIT_TAG 5052181f420a9b82aef6cfa5b779b425622b7acd)
message(STATUS "${PROJECT_NAME} fetch IPC toolkit ${IPC_TOOLKIT_GIT_TAG} from source ")
include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
    GIT_TAG ${IPC_TOOLKIT_GIT_TAG}
)
option(IPC_TOOLKIT_BUILD_TESTS "Build unit-tests" OFF)
FetchContent_MakeAvailable(ipc_toolkit)