set(CLI11_VERSION 2.1.2)
find_package(CLI11 ${CLI11_VERSION} QUIET)

if(NOT TARGET CLI11::CLI11)
  message(STATUS "${PROJECT_NAME} Fetch CLI11 ${CLI11_VERSION} from source ")
  include(FetchContent)
  FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11
    GIT_TAG v${CLI11_VERSION}
    GIT_SHALLOW TRUE
    UPDATE_COMMAND ""
  )
  FetchContent_MakeAvailable(CLI11)
else()
  message(STATUS "${PROJECT_NAME} CLI11 ${CLI11_VERSION} FOUND ")
endif(NOT TARGET CLI11::CLI11)