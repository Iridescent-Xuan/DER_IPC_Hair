set(SPDLOG_VERSION 1.11.0)
message(STATUS "${PROJECT_NAME} fetch spdlog ${SPDLOG_VERSION} from source ")
include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v${SPDLOG_VERSION}
  GIT_SHALLOW TRUE
  UPDATE_COMMAND ""
)
FetchContent_MakeAvailable(spdlog)