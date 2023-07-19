set(SPDLOG_VERSION 1.9.2)
find_package(spdlog ${SPDLOG_VERSION} QUIET)

if(NOT TARGET spdlog::spdlog)
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
else()
  message(STATUS "${PROJECT_NAME} spdlog ${SPDLOG_VERSION} FOUND ")
endif(NOT TARGET spdlog::spdlog)